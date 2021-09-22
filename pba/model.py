# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PBA & AutoAugment Train/Eval module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os
import time

import numpy as np
import tensorflow as tf

import pba.data_utils as data_utils
import pba.helper_utils as helper_utils
from pba.bert_model import build_bert_model
from pba.bert_optimization import create_optimizer
from pba.augmentation_utils import ContextNeighborStorage

import six
import json
import re
import collections


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=32,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probabilitiy for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The sttdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    self.current_learning_rate = None


def from_dict(json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
    return config


def config_from_json_file(json_file, model_dropout):
    """Constructs a `BertConfig` from a json file of parameters."""
    with open(json_file, "r") as reader:
      text = reader.read()
    config = from_dict(json.loads(text))
    if model_dropout != -1:
      config.hidden_dropout_prob = model_dropout
      config.attention_probs_dropout_prob = model_dropout
    return config


def build_model(input_ids, input_mask, token_type_ids, num_classes, is_training, hparams, noise_vector):
    """Constructs the vision model being trained/evaled.

    Args:
      inputs: input features being fed to the model build built.
      num_classes: number of output classes being predicted.
      is_training: is the model training or not.
      hparams: additional hyperparameters associated with the model.

    Returns:
    Returns:
      The logits of the model.
    """

    if hparams.model_name == 'bert':
        bert_config_file = os.path.join(hparams.data_path + 'pretrained_models/bert_base/bert_config.json')
        bert_config = config_from_json_file(bert_config_file,-1)
        logits, embd_output = build_bert_model(input_ids, input_mask, token_type_ids, num_classes, is_training, bert_config, noise_vector)

    return logits, embd_output


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name_to_variable[name]
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)

class Model(object):
    """Builds an model."""

    def __init__(self, hparams, num_classes, text_size):
        self.hparams = hparams
        self.num_classes = num_classes
        self.text_size = text_size

    def build(self, mode):
        """Construct the model."""
        assert mode in ['train', 'eval']
        self.mode = mode
        self._setup_misc(mode)
        self._setup_texts_and_labels(self.hparams.dataset) # --- create placeholders
        self._build_graph(self.input_ids, self.input_mask, self.token_type_ids, self.labels, mode, self.noise_vector)


    def _setup_misc(self, mode):
        """Sets up miscellaneous in the model constructor."""

        self.lr_rate_ph = self.hparams.lr
        self.current_learning_rate = self.lr_rate_ph
        self.batch_size = self.hparams.batch_size
        self.dataset = self.hparams.dataset
        self.max_seq_length = self.hparams.max_seq_length

        self.epoch_accuracy = []
        self.matthews_corr = []
        self.loss_history = []
        if mode == 'eval':
            self.batch_size = self.hparams.test_batch_size

    def _setup_texts_and_labels(self, dataset):
        """Sets up text and label placeholders for the model."""
        self.input_ids = tf.placeholder(tf.int32, [None, self.text_size])
        self.input_mask = tf.placeholder(tf.int32,[None, self.text_size])
        self.token_type_ids = tf.placeholder(tf.int32, [None, self.text_size])
        if self.num_classes < 100: # --- classification
            self.labels = tf.placeholder(tf.int32, [None, self.num_classes])
        else: # --- regression
            self.labels = tf.placeholder(tf.float32, [None, 1])

        self.noise_vector = tf.placeholder(tf.float32, [None, None, 768])


    def assign_epoch(self, session, epoch_value):
        session.run(
            self._epoch_update, feed_dict={self._new_epoch: epoch_value})


    def _build_graph(self, input_ids, input_mask, token_type_ids, labels, mode, noise_vector):
        """Constructs the TF graph for the model.

        Args:
          texts: A 2-D text Tensor
          labels: A 2-D labels Tensor.
          mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
        """
        is_training = 'train' in mode
        if is_training:
            self.global_step = tf.train.get_or_create_global_step()

        # texts is placeholder set in _setup_texts_and_labels(data set)
        logits, embd_output = build_model(input_ids, input_mask, token_type_ids, self.num_classes, is_training,
                             self.hparams, noise_vector)

        self.embedding_output = embd_output

        if self.dataset == 'stsb':
            self.predictions = logits
            self.cost = tf.reduce_mean(tf.square(logits - labels))
        else:
            self.predictions, self.cost = helper_utils.setup_loss(logits, labels)

        self._calc_num_trainable_params()

        if is_training:
            self._build_train_op()

        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(max_to_keep=10)

        init_checkpoint = os.path.join(self.hparams.data_path,'pretrained_models', 'bert_base', 'bert_model.ckpt')
        tvars = tf.trainable_variables("bert")

        (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        self.assignment_map = assignment_map
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        self.init = tf.global_variables_initializer()

    def _calc_num_trainable_params(self):
        self.num_trainable_params = np.sum([
            np.prod(var.get_shape().as_list())
            for var in tf.trainable_variables()
        ])
        tf.logging.info('number of trainable params: {}'.format(
            self.num_trainable_params))

    def _build_train_op(self):
        """Builds the train op for the model."""
        hparams = self.hparams
        clip_norm = hparams.gradient_clipping_by_global_norm
        num_train_data = hparams.train_size
        batch_size = hparams.batch_size
        num_epochs = hparams.num_epochs
        num_train_steps = int(np.floor(num_train_data/batch_size) * num_epochs * 0.9)
        num_warmup_steps = int(np.floor(num_train_data/batch_size) * num_epochs * 0.1)
        self.train_op, self.curr_learning_rate_tensor = create_optimizer(self.cost, self.lr_rate_ph, num_train_steps, num_warmup_steps, False, clip_norm, self.global_step)


class ModelTrainer(object):
    """Trains an instance of the Model class."""

    def __init__(self, hparams):
        self._session = None
        self.hparams = hparams

        np.random.seed(0)  # --- Set the random seed to be sure the same validation set is used for each model
        self.data_loader = data_utils.DataSet(hparams)
        np.random.seed()  # --- Put the random seed back to random
        self.data_loader.reset()

        # extra stuff for ray
        self._build_models()
        self._new_session()
        self._session.__enter__()

        self.create_nn_database(self.m, self.session)


    def save_model(self, checkpoint_dir, step=None):
        """Dumps model into the backup_dir.

        Args:
          step: If provided, creates a checkpoint with the given step
            number, instead of overwriting the existing checkpoints.
        """
        model_save_name = os.path.join(checkpoint_dir,'model.ckpt') + '-' + str(step)
        save_path = self.saver.save(self.session, model_save_name)
        tf.logging.info('Saved child model')
        return model_save_name

    def extract_model_spec(self, checkpoint_path):
        """Loads a checkpoint with the architecture structure stored in the name."""
        self.saver.restore(self.session, checkpoint_path)
        tf.logging.warning(
            'Loaded child model checkpoint from {}'.format(checkpoint_path))

    def eval_child_model(self, model, data_loader, mode):
        """Evaluate the child model.

        Args:
          model: image model that will be evaluated.
          data_loader: dataset object to extract eval data from.
          mode: will the model be evaled on train, val or test.

        Returns:
          Accuracy of the model on the specified dataset.
        """
        tf.logging.info('Evaluating child model in mode {}'.format(mode))
        while True:
            try:
                accuracy, matthews_corrcoef, f1_score, pearson, spearman = helper_utils.eval_child_model(
                    self.session, model, data_loader, mode)
                tf.logging.info(
                    'Eval child model accuracy: {}'.format(accuracy))
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info(
                    'Retryable error caught: {}.  Retrying.'.format(e))
        return accuracy, matthews_corrcoef, f1_score, pearson, spearman

    @contextlib.contextmanager
    def _new_session(self):
        """Creates a new session for model m. initialize variables, and save / restore from checkpoint."""
        sess_cfg = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False)
        sess_cfg.gpu_options.allow_growth = True
        self._session = tf.Session('', config=sess_cfg)
        self._session.run(self.m.init)

        return self._session

    def _build_models(self):
        """Builds the text models for train and eval."""
        m = Model(self.hparams, self.data_loader.num_classes, self.data_loader.text_size)
        m.build('train')
        self._num_trainable_params = m.num_trainable_params
        self._saver = m.saver
        self.m = m
        self.meval = m


    def create_nn_database(self, model, session):
        """Create search index for nearest neighbour augmentation from all samples in the train data"""
        if type(self.data_loader.train_texts[0]) == str:
            self.nn_database = ContextNeighborStorage(sentences=self.data_loader.train_texts, n_labels=self.data_loader.train_labels.shape[1], model=model, session=session)
        elif type(self.data_loader.train_texts[0]) == tuple:
            all_sentences = [list(sent_pair) for sent_pair in self.data_loader.train_texts]
            all_sentences_flat = [item for sublist in all_sentences for item in sublist]
            self.nn_database = ContextNeighborStorage(sentences=all_sentences_flat, n_labels=self.data_loader.train_labels.shape[1], model=model, session=session)
        self.nn_database.process_sentences()
        self.nn_database.build_search_index()


    def _run_training_loop(self, curr_epoch):
        """Trains the model `m` for one epoch."""
        start_time = time.time()
        while True:
            try:
                train_accuracy, train_matthews, train_f1_score, train_pearson, train_spearman = helper_utils.run_epoch_training(self.session, self.m, self.data_loader, self.nn_database, curr_epoch)
                break
            except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
                tf.logging.info(
                    'Retryable error caught: {}.  Retrying.'.format(e))
        tf.logging.info('Finished epoch: {}'.format(curr_epoch))
        tf.logging.info('Epoch time(min): {}'.format(
            (time.time() - start_time) / 60.0))

        return train_accuracy, train_matthews, train_f1_score, train_pearson, train_spearman

    def _compute_final_accuracies(self, iteration):
        """Run once training is finished to compute final test accuracy."""
        if (iteration >= self.hparams.num_epochs - 1):
            test_accuracy, test_matthews_corrcoef, test_f1_score, test_pearson, test_spearman = self.eval_child_model(self.m, self.data_loader, 'test')
        else:
            test_accuracy = 0
            test_matthews_corrcoef = 0
            test_f1_score = 0
            test_pearson = 0
            test_spearman = 0
        tf.logging.info('Test Accuracy: {}'.format(test_accuracy))
        tf.logging.info('Test Matthew\' s Corr: {}'.format(test_matthews_corrcoef))
        tf.logging.info('Test F1 Score: {}'.format(test_f1_score))
        tf.logging.info('Test Pearson: {}'.format(test_pearson))
        tf.logging.info('Test Spearman: {}'.format(test_spearman))
        return test_accuracy, test_matthews_corrcoef, test_f1_score, test_pearson, test_spearman

    def run_model(self, epoch):
        """Trains and evalutes the image model."""
        valid_accuracy = 0.
        valid_matthews = 0.
        valid_f1_score = 0.
        valid_pearson = 0.
        valid_spearman = 0.
        training_accuracy, training_matthews, training_f1_score, training_pearson, training_spearman = self._run_training_loop(epoch)
        if self.hparams.validation_size > 0:
            valid_accuracy, valid_matthews, valid_f1_score, valid_pearson, valid_spearman = self.eval_child_model(self.m,
                                                   self.data_loader, 'val')
        tf.logging.info('Train Acc: {}, Valid Acc: {}'.format(
            training_accuracy, valid_accuracy))
        return training_accuracy, training_matthews, training_f1_score, training_pearson, training_spearman, valid_accuracy, valid_matthews, valid_f1_score, valid_pearson, valid_spearman

    def reset_config(self, new_hparams):
        self.hparams = new_hparams
        self.data_loader.reset_policy(new_hparams)
        return

    @property
    def saver(self):
        return self._saver

    @property
    def session(self):
        return self._session

    @property
    def num_trainable_params(self):
        return self._num_trainable_params
