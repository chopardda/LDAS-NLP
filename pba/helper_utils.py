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
"""Helper functions used for training PBA models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from scipy import stats
import math

from autoaugment.helper_utils import setup_loss, cosine_lr
from pba.data_preprocessing import BERT_preprocess
from pba.bert_features import convert_data_to_features
from pba.augmentation_transforms_hp import float_parameter


def eval_child_model(session, model, data_loader, mode):
    """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will evaluate.
    mode: Will `model` either evaluate validation or test data.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
    if mode == 'val':
        texts = data_loader.val_texts
        labels = data_loader.val_labels
    elif mode == 'test':
        texts = data_loader.test_texts
        labels = data_loader.test_labels
    else:
        raise ValueError('Not valid eval mode')
    assert len(texts) == len(labels)
    tf.logging.info('Eval model.batch_size is {}'.format(model.batch_size))
    eval_batches = int(len(texts) / model.batch_size)
    if len(texts) % model.batch_size != 0:
        eval_batches += 1
    correct = 0
    count = 0
    if model.hparams.dataset in ['cola', 'mrpc', 'qqp']:
        confusion_matrix = np.zeros([2, 2])
    elif model.hparams.dataset == 'stsb':
        full_preds = []
        full_labels = []
    for i in range(eval_batches):
        eval_texts = texts[i * model.batch_size:(i + 1) * model.batch_size]
        eval_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]

        if type(eval_texts[0]) == str:
            eval_tokens = BERT_preprocess(eval_texts)
        elif type(eval_texts[0]) == np.str_:
            eval_tokens = BERT_preprocess(eval_texts)
        elif type(eval_texts[0]) == tuple: # --- list of tuples
            eval_tokens_a = BERT_preprocess([texts[0] for texts in eval_texts])
            eval_tokens_b = BERT_preprocess([texts[1] for texts in eval_texts])
            eval_tokens = merge(eval_tokens_a, eval_tokens_b)


        dummy_labels = np.zeros((len(eval_texts),2)) # --- for test data, do not give true label (give label 0)
        features = convert_data_to_features(eval_tokens, dummy_labels, label_list=['neg', 'pos'], max_seq_length=model.hparams.max_seq_length)

        input_ids = []
        input_mask = []
        token_type_ids = []
        for el in features:
            input_ids.append(el.input_ids)
            input_mask.append(el.input_mask)
            token_type_ids.append(el.input_type_ids)

        input_ids = np.array(input_ids, np.float32)
        input_mask = np.array(input_mask, np.float32)
        token_type_ids = np.array(token_type_ids, np.float32)

        noise_vector = np.zeros((len(eval_tokens), model.hparams.max_seq_length, 768))

        preds = session.run(
            model.predictions,
            feed_dict={
                model.input_ids: input_ids,
                model.input_mask: input_mask,
                model.token_type_ids: token_type_ids,
                model.labels: eval_labels,
                model.noise_vector: noise_vector
            })
        correct += np.sum(
            np.equal(np.argmax(eval_labels, 1), np.argmax(preds, 1)))
        count += len(preds)

        if model.hparams.dataset in ['cola', 'mrpc', 'qqp']:
            for sample_idx in range(preds.shape[0]):
                confusion_matrix[
                    np.argmax(eval_labels, axis=1)[sample_idx], np.argmax(preds, axis=1)[sample_idx]] += 1
        elif model.hparams.dataset == 'stsb':
            full_preds.extend(preds)
            full_labels.extend(eval_labels)

    accuracy = correct/count
    tf.logging.info('Eval accuracy: {}'.format(accuracy))
    tf.logging.info('Eval: correct: {}, total: {}'.format(correct, count))
    matthews_corrcoef = 0
    f1_score = 0
    pearson_corr = 0
    spearman_corr = 0
    if model.hparams.dataset == 'stsb':
        # pearson_corr, _ = stats.pearsonr(full_preds, full_labels)
        spearman_corr, _ = stats.spearmanr (full_preds, full_labels)
        tf.logging.info('Eval Pearson: {}'.format(pearson_corr))
        tf.logging.info('Eval Spearman: {}'.format(spearman_corr))
    elif model.hparams.dataset in ['cola', 'mrpc', 'qqp']:
        TP = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        TN = confusion_matrix[0,0]
        FN = confusion_matrix[1,0]
        if model.hparams.dataset == 'cola':
            matthews_corrcoef = ((TP*TN)-(FP*FN))/math.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
            tf.logging.info('Eval Matthew\' Corr: {}'.format(matthews_corrcoef))
        elif model.hparams.dataset in ['mrpc', 'qqp']:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)
            tf.logging.info('Eval F1 Score: {}'.format(f1_score))

    return accuracy, matthews_corrcoef, f1_score, pearson_corr, spearman_corr


def step_lr(learning_rate, epoch):
    """Step Learning rate.

  Args:
    learning_rate: Initial learning rate.
    epoch: Current epoch we are one. This is one based.

  Returns:
    The learning rate to be used for this current batch.
  """
    if epoch < 80:
        return learning_rate
    elif epoch < 120:
        return learning_rate * 0.1
    else:
        return learning_rate * 0.01


def get_lr(curr_epoch, hparams, iteration=None):
    """Returns the learning rate during training based on the current epoch."""
    assert iteration is not None
    batches_per_epoch = int(hparams.train_size / hparams.batch_size)
    if 'svhn' in hparams.dataset and 'wrn' in hparams.model_name:
        lr = step_lr(hparams.lr, curr_epoch)
    elif 'cifar' in hparams.dataset or ('svhn' in hparams.dataset and
                                        'shake_shake' in hparams.model_name):
        lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,
                       hparams.num_epochs)
    else:
        lr = hparams.lr
        tf.logging.log_first_n(tf.logging.WARN, 'Default not changing learning rate.', 1)
    return lr


def merge(list1, list2):
    merged_list = []
    for i in range(max((len(list1), len(list2)))):

        while True:
            try:
                tup = (list1[i], list2[i])
            except IndexError:
                if len(list1) > len(list2):
                    list2.append('')
                    tup = (list1[i], list2[i])
                elif len(list1) < len(list2):
                    list1.append('')
                    tup = (list1[i], list2[i])
                continue

            merged_list.append(tup)
            break
    return merged_list


def run_epoch_training(session, model, data_loader, nn_database, curr_epoch):
    """Runs one epoch of training for the model passed in.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will evaluate.
    curr_epoch: How many of epochs of training have been done so far.

  Returns:
    The accuracy of 'model' on the training set
  """
    steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
    tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
    curr_step = session.run(model.global_step)
    curr_lr = model.current_learning_rate

    tf.logging.info('Initial lr of {} for epoch {}'.format(curr_lr, curr_epoch))

    correct = 0
    count = 0

    loss_list = []
    if model.hparams.dataset in ['cola', 'mrpc', 'qqp']:
        confusion_matrix = np.zeros([2, 2])
    elif model.hparams.dataset == 'stsb':
        full_preds = []
        full_labels = []

    for step in range(steps_per_epoch):

        if step % 20 == 0:
            tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

        train_data, embd_policies_list = data_loader.next_batch(nn_database, curr_epoch)  # list of 32 lists
        train_texts, train_labels = train_data # train_texts is list

        if type(train_texts[0]) == np.str_:
            train_tokens = BERT_preprocess(train_texts)
        elif type(train_texts[0]) == str:
            train_tokens = BERT_preprocess(train_texts)
        elif type(train_texts[0]) == np.ndarray:
            train_tokens_a = BERT_preprocess([texts[0] for texts in train_texts])
            train_tokens_b = BERT_preprocess([texts[1] for texts in train_texts])
            train_tokens = merge(train_tokens_a, train_tokens_b)
        elif type(train_texts[0]) == tuple:
            train_tokens_a = BERT_preprocess([texts[0] for texts in train_texts])
            train_tokens_b = BERT_preprocess([texts[1] for texts in train_texts])
            train_tokens = merge(train_tokens_a, train_tokens_b)

        features = convert_data_to_features(train_tokens, train_labels, label_list=['neg', 'pos'], max_seq_length=model.hparams.max_seq_length)

        input_ids = []
        input_mask = []
        token_type_ids = []
        for el in features:
            input_ids.append(el.input_ids)
            input_mask.append(el.input_mask)
            token_type_ids.append(el.input_type_ids)

        input_ids = np.array(input_ids, np.float32)
        input_mask = np.array(input_mask, np.float32)
        token_type_ids = np.array(token_type_ids, np.float32)

        noise_vector = np.zeros((len(train_tokens), model.hparams.max_seq_length, 768))
        for s_idx in range(len(train_tokens)):
            if embd_policies_list[s_idx] == None:
                continue
            for policy_count in range(len(embd_policies_list[s_idx])):
                name, probability, level = embd_policies_list[s_idx][policy_count]
                if name == 'EmbNormalNoise':
                    sigma = float_parameter(level, 1.0)
                    noise_vector[s_idx] += np.random.normal(0, sigma, noise_vector[s_idx].shape)
                elif name == 'EmbUniformNoise':
                    uu = float_parameter(level, 1.0)
                    noise_vector[s_idx] += np.random.uniform(-uu, uu, noise_vector[s_idx].shape)

        noise_vector = noise_vector * input_mask[:,:,np.newaxis] # take care of padded sequences

        run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        _, loss, step, preds, true_lr, embedding_output = session.run(
                [model.train_op, model.cost, model.global_step, model.predictions, model.curr_learning_rate_tensor, model.embedding_output],
                feed_dict={
                    model.input_ids: input_ids,
                    model.input_mask: input_mask,
                    model.token_type_ids: token_type_ids,
                    model.labels: train_labels,
                    model.noise_vector: noise_vector
                }, options=run_options)


        model.current_learning_rate = true_lr

        loss_list.append(loss)

        correct += np.sum(np.equal(np.argmax(train_labels, 1), np.argmax(preds, 1)))
        count += len(preds)

        if  model.hparams.dataset  in ['cola', 'mrpc', 'qqp']:
            for sample_idx in range(preds.shape[0]):
                confusion_matrix[
                    np.argmax(train_labels, axis=1)[sample_idx], np.argmax(preds, axis=1)[sample_idx]] += 1
        elif model.hparams.dataset == 'stsb':
            full_preds.extend(preds)
            full_labels.extend(train_labels)


    accuracy = correct / count
    tf.logging.info('Train accuracy: {}'.format(accuracy))
    tf.logging.info('train: correct: {}, total: {}'.format(correct, count))
    matthews_corrcoef = 0
    f1_score = 0
    pearson_corr = 0
    spearman_corr = 0
    if model.hparams.dataset == 'stsb':
        spearman_corr, _ = stats.spearmanr (full_preds, full_labels)
        tf.logging.info('Train Pearson: {}'.format(pearson_corr))
        tf.logging.info('Train Spearman: {}'.format(spearman_corr))
    elif model.hparams.dataset in ['cola', 'mrpc', 'qqp']:
        TP = confusion_matrix[1,1]
        FP = confusion_matrix[0,1]
        TN = confusion_matrix[0,0]
        FN = confusion_matrix[1,0]
        if model.hparams.dataset == 'cola':
            matthews_corrcoef = ((TP*TN)-(FP*FN))/math.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
            tf.logging.info('Train Matthew\'s Corr: {}'.format(matthews_corrcoef))
        elif model.hparams.dataset in ['mrpc', 'qqp']:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * (precision * recall) / (precision + recall)
            tf.logging.info('Train F1 Score: {}'.format(f1_score))

    model.loss_history.append(np.mean(np.array(loss_list)))

    model.epoch_accuracy.append(accuracy)
    if model.hparams.dataset == 'cola':
        model.matthews_corr.append(matthews_corrcoef)

    return accuracy, matthews_corrcoef, f1_score, pearson_corr, spearman_corr

