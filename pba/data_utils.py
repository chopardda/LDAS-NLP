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
"""Data utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle
import csv
import random
import os
import numpy as np
import tensorflow as tf

import pba.augmentation_transforms_hp as augmentation_transforms_pba
import pba.augmentation_transforms as augmentation_transforms_autoaug
from pba.utils import parse_log_schedule
from pba.bert_features import convert_to_unicode


def parse_policy(policy_emb, augmentation_transforms):
    policy = []
    num_xform = augmentation_transforms.NUM_HP_TRANSFORM
    xform_names = augmentation_transforms.HP_TRANSFORM_NAMES
    assert len(policy_emb
               ) == 2 * num_xform, 'policy was: {}, supposed to be: {}'.format(
                   len(policy_emb), 2 * num_xform)
    for i, xform in enumerate(xform_names):
        policy.append((xform, policy_emb[2 * i], policy_emb[2 * i + 1]))
    return policy


def shuffle_data(data, labels):
    """Shuffle data using numpy."""
    np.random.seed(0)
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data = data[perm]
    labels = labels[perm]
    return data, labels


def read_tsv(input_file, quotechar=None):
    with open(input_file, 'r') as data_file:
        reader = csv.reader(data_file, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    return lines


class DataSet(object):
    """Dataset object that produces augmented training and eval data."""

    def __init__(self, hparams):
        self.hparams = hparams
        self.epochs = 0
        self.curr_train_index = 0

        self.parse_policy(hparams)
        self.load_data(hparams)

        assert len(self.test_texts) == len(self.test_labels)
        assert len(self.train_texts) == len(self.train_labels)
        assert len(self.val_texts) == len(self.val_labels)
        tf.logging.info('train dataset size: {}, test: {}, val: {}'.format(
            len(self.train_texts), len(self.test_texts), len(self.val_texts)))

    def parse_policy(self, hparams):
        """Parses policy schedule from input, which can be a list, list of lists, text file, or pickled list.

        If list is not nested, then uses the same policy for all epochs.

        Args:
        hparams: tf.hparams object.
        """
        # Parse policy
        if hparams.use_hp_policy: # --- if state == search and use_hp_policy == True
            self.augmentation_transforms = augmentation_transforms_pba

            if isinstance(hparams.hp_policy,
                          str) and hparams.hp_policy.endswith('.txt'):
                if hparams.num_epochs % hparams.hp_policy_epochs != 0:
                    tf.logging.warning(
                        "Schedule length (%s) doesn't divide evenly into epochs (%s), interpolating.",
                        hparams.num_epochs, hparams.hp_policy_epochs)
                tf.logging.info(
                    'schedule policy trained on {} epochs, parsing from: {}, multiplier: {}'
                    .format(
                        hparams.hp_policy_epochs, hparams.hp_policy,
                        float(hparams.num_epochs) / hparams.hp_policy_epochs))
                raw_policy = parse_log_schedule(
                    hparams.hp_policy,
                    epochs=hparams.hp_policy_epochs,
                    multiplier=float(hparams.num_epochs) /
                    hparams.hp_policy_epochs)
            elif isinstance(hparams.hp_policy,
                            str) and hparams.hp_policy.endswith('.p'):
                assert hparams.num_epochs % hparams.hp_policy_epochs == 0
                tf.logging.info('custom .p file, policy number: {}'.format(
                    hparams.schedule_num))
                with open(hparams.hp_policy, 'rb') as f:
                    policy = pickle.load(f)[hparams.schedule_num]
                raw_policy = []
                for num_iters, pol in policy:
                    for _ in range(num_iters * hparams.num_epochs //
                                   hparams.hp_policy_epochs):
                        raw_policy.append(pol)
            else: # --- search
                raw_policy = hparams.hp_policy

            if isinstance(raw_policy[0], list):
                self.policy = []

                for pol in raw_policy:
                    cur_pol = parse_policy(pol, self.augmentation_transforms)
                    self.policy.append(cur_pol)
                tf.logging.info('using HP policy schedule, last: {}'.format(
                    self.policy[-1]))
            elif isinstance(raw_policy, list):
                self.policy = parse_policy(raw_policy, self.augmentation_transforms) # outputs list of NUM_HP_TRANSFORM tuples (str augm name, 0/10, 0)
                tf.logging.info('using HP Policy, policy: {}'.format(
                    self.policy))

        else: # --- if use_hp_policy == False
            self.augmentation_transforms = augmentation_transforms_autoaug
            tf.logging.info('using ENAS Policy or no augmentaton policy')
            self.policy = None


    def reset_policy(self, new_hparams):
        self.hparams = new_hparams
        self.parse_policy(new_hparams)
        tf.logging.info('reset aug policy')
        return


    def populate_data(self, hparams, train_sentences, train_labels, test_sentences, test_labels):
        """Populate data

        Populates:
            self.train_texts: Training text data.
            self.train_labels: Training ground truth labels.
            self.val_texts: Validation/holdout text data.
            self.val_labels: Validation/holdout ground truth labels.
            self.test_texts: Testing text data. (can be dev or test set, depending on haprams.development)
            self.test_labels: Testing ground truth labels. (can be dev or test set, depending on haprams.development)

        Args:
            hparams: tf.hparams object.
            train_sentences: list of str
            train_labels: list of int/float
            test_sentences: list of str
            test_labels: list of int/float
        """
        train_size, val_size = hparams.train_size, hparams.validation_size

        data = list(zip(train_sentences, train_labels))
        random.Random(21).shuffle(data) # --- set seed for reproducibility
        train_sentences, train_labels = zip(*data)

        self.train_texts = list(train_sentences[:train_size])
        self.train_labels = list(train_labels[:train_size])
        if val_size != 0:
            self.val_texts = list(train_sentences[-val_size:])
            self.val_labels = list(train_labels[-val_size:])
        else:
            self.val_texts = []
            self.val_labels = []

        self.test_texts = test_sentences
        self.test_labels = test_labels  # all zero, don't know true labels for test data


    def load_CoLA(self, hparams):
        """ Loader for the CoLA data set (GLUE version)
         based on https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L255
         task: acceptability, metrics: Matthews corr., domain: miscellaneous
        """

        def load_CoLA_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                # --- only the test set has a header
                if set_type == 'test' and i == 0:
                    continue
                if set_type == 'test':
                    text_a = convert_to_unicode(line[1])
                    label = "0"
                else:
                    text_a = convert_to_unicode(line[3])
                    label = convert_to_unicode(line[1])
                sentences.append(text_a)
                labels.append(label)
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'CoLA')
        train_sentences, train_labels = load_CoLA_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 8551 samples
        if hparams.development:
            test_sentences, test_labels = load_CoLA_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_CoLA_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 8551
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2


    def load_MNLIm(self, hparams):
        """ Loader for the MultiNLI data set (GLUE version) based on https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L255 """
        # task: NLI, metrics: matched acc/mismatched acc., domain: miscellaneous

        def load_MNLI_data(data, set_type):
            label_dic = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i==0:
                    continue
                text_a = convert_to_unicode(line[8])
                text_b = convert_to_unicode(line[9])
                if set_type == 'test':
                    label_text = "neutral" # label unknown, dummy variable
                else:
                    label_text = convert_to_unicode(line[-1])
                sentences.append((text_a, text_b))
                labels.append(label_dic[label_text])
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'MNLI')
        train_sentences, train_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 392'702 samples
        if hparams.development:
            test_sentences, test_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'dev_matched.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'test_matched.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 392702
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 3 # {0,1,2}


    def load_MNLImm(self, hparams):
        """ Loader for the MultiNLI data set (GLUE version) based on https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L255 """
        # task: NLI, metrics: matched acc/mismatched acc., domain: miscellaneous

        def load_MNLI_data(data, set_type):
            label_dic = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i==0:
                    continue
                text_a = convert_to_unicode(line[8])
                text_b = convert_to_unicode(line[9])
                if set_type == 'test':
                    label_text = 'neutral' # label unknown, dummy variable
                else:
                    label_text = convert_to_unicode(line[-1])
                sentences.append((text_a, text_b))
                labels.append(label_dic[label_text])
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'MNLI')
        train_sentences, train_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 392'702 samples
        if hparams.development:
            test_sentences, test_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'dev_mismatched.tsv')), 'development') # 9832 samples
        else:
            test_sentences, test_labels = load_MNLI_data(read_tsv(os.path.join(data_dir, 'test_mismatched.tsv')), 'test') # 9847 samples

        assert hparams.train_size + hparams.validation_size <= 392702
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 3 # {0,1,2}


    def load_MRPC(self, hparams):
        """ Loader for the MRPC data set (GLUE version) based on https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_classifier.py#L255 """
        # task: paraphrase, metrics: acc/F1, domain: online news sources

        def load_MRPC_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0:
                    continue
                # guid = "%s-%s" % (set_type, i)
                text_a = convert_to_unicode(line[3])
                text_b = convert_to_unicode(line[4])
                if set_type == 'test':
                    label = "0"
                else:
                    label = convert_to_unicode(line[0])
                sentences.append((text_a, text_b))
                labels.append(label)
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'MRPC')
        train_sentences, train_labels = load_MRPC_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 3668 samples
        if hparams.development:
            test_sentences, test_labels = load_MRPC_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_MRPC_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 3668
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2


    def load_QNLI(self, hparams):

        def load_QNLI_data(data, set_type):
            label_dic = {'not_entailment': 0, 'entailment': 1}
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # all have header
                    continue
                text_a = convert_to_unicode(line[1])
                text_b = convert_to_unicode(line[2])
                if set_type == 'test':
                    label_text = "not_entailment"
                else:
                    label_text = convert_to_unicode(line[3])
                sentences.append((text_a, text_b))
                labels.append(label_dic[label_text])
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'QNLI')
        train_sentences, train_labels = load_QNLI_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 108'436 samples
        if hparams.development:
            test_sentences, test_labels = load_QNLI_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_QNLI_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 108436
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2 # {0,1} ((not) answerable)


    def load_QQP(self, hparams):

        def load_QQP_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0:
                    continue
                if set_type == 'test':
                    text_a = convert_to_unicode(line[1])
                    text_b = convert_to_unicode(line[2])
                    label = 0
                else:
                    try:
                        text_a = convert_to_unicode(line[3])
                        text_b = convert_to_unicode(line[4])
                        label = convert_to_unicode(line[5])
                    except:
                        continue
                sentences.append((text_a, text_b))
                labels.append(label)
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'QQP')
        train_sentences, train_labels = load_QQP_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 363'849 samples
        if hparams.development:
            test_sentences, test_labels = load_QQP_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_QQP_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 363849
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2 # {0,1}


    def load_RTE(self, hparams):
        def load_RTE_data(data, set_type):
            label_dic = {'not_entailment': 0, 'entailment': 1}
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # --- all have a header
                    continue
                text_a = convert_to_unicode(line[1])
                text_b = convert_to_unicode(line[2])
                if set_type == 'test':
                    label_text = "not_entailment"
                else:
                    label_text = convert_to_unicode(line[3])
                sentences.append((text_a, text_b))
                labels.append(label_dic[label_text])
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'RTE')
        train_sentences, train_labels = load_RTE_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 2490 samples
        if hparams.development:
            test_sentences, test_labels = load_RTE_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_RTE_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 2490
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2 # {0,1} does not entail, entail


    def load_SNLI(self, hparams):
        # --- only samples with one of the three labels, these with label "-" already excluded
        def load_SNLI_data(data, set_type):
            label_dic = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # all have header
                    continue
                text_a = convert_to_unicode(line[7])
                text_b = convert_to_unicode(line[8])
                label_text = convert_to_unicode(line[-1])
                sentences.append((text_a, text_b))
                labels.append(label_dic[label_text])
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'SNLI')
        train_sentences, train_labels = load_SNLI_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 549367 samples
        if hparams.development:
            test_sentences, test_labels = load_SNLI_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_SNLI_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 549367
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 3


    def load_SST2(self, hparams):
        def load_SST2_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # --- all have header
                    continue
                if set_type == 'test':
                    text_a = convert_to_unicode(line[1])
                    label_text = '0'
                else:
                    text_a = convert_to_unicode(line[0])
                    label_text = line[1]
                sentences.append(text_a)
                labels.append(int(label_text))
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'SST-2')
        train_sentences, train_labels = load_SST2_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 67349 samples # list of str
        if hparams.development:
            test_sentences, test_labels = load_SST2_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_SST2_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 67349
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2 # {positive,negative}


    def load_STSB(self, hparams):
        def load_STSB_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # all have header
                    continue
                text_a = convert_to_unicode(line[7])
                text_b = convert_to_unicode(line[8])
                if set_type == 'test':
                    label_text = '0'
                else:
                    label_text = line[-1]
                sentences.append((text_a, text_b))
                labels.append(float(label_text))
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'STS-B')
        train_sentences, train_labels = load_STSB_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 5749 samples
        if hparams.development:
            test_sentences, test_labels = load_STSB_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_STSB_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 5749
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)

        # regression
        self.num_classes = 100

    def load_WNLI(self, hparams):
        # only samples with one of the three labels, these with label "-" already excluded
        def load_WNLI_data(data, set_type):
            sentences = []
            labels = []
            for (i, line) in enumerate(data):
                if i == 0: # all have header
                    continue
                text_a = convert_to_unicode(line[1])
                text_b = convert_to_unicode(line[2])
                if set_type == 'test':
                    label_text = '0'
                else:
                    label_text = line[3]
                sentences.append((text_a, text_b))
                labels.append(int(label_text))
            return sentences, labels

        data_dir = os.path.join(hparams.data_path, 'glue_data', 'WNLI')
        train_sentences, train_labels = load_WNLI_data(read_tsv(os.path.join(data_dir, 'train.tsv')), 'train') # 635 samples
        if hparams.development:
            test_sentences, test_labels = load_WNLI_data(read_tsv(os.path.join(data_dir, 'dev.tsv')), 'development') # 1043 samples
        else:
            test_sentences, test_labels = load_WNLI_data(read_tsv(os.path.join(data_dir, 'test.tsv')), 'test') # 1063 samples

        assert hparams.train_size + hparams.validation_size <= 635
        self.populate_data(hparams, train_sentences, train_labels, test_sentences, test_labels)
        self.num_classes = 2 # {0,1}


    def load_data(self, hparams):
        """Load raw data from specified dataset.

        Assumes data is in NCHW format.

        Populates:
            self.train_texts: Training text data.
            self.train_labels: Training ground truth labels.
            self.val_texts: Validation/holdout text data.
            self.val_labels: Validation/holdout ground truth labels.
            self.test_texts: Testing text data.
            self.test_labels: Testing ground truth labels.
            self.num_classes: Number of classes.
            self.num_train: Number of training examples.
            self.text_size: Max sequence length
            self.num_embed_dim: Number of embedding dimensions

        Args:
            hparams: tf.hparams object.
        """
        if hparams.dataset == 'cola':
            self.load_CoLA(hparams)
        elif hparams.dataset == 'sst2':
            self.load_SST2(hparams)
        elif hparams.dataset == 'mrpc':
            self.load_MRPC(hparams)
        elif hparams.dataset == 'stsb':
            self.load_STSB(hparams)
        elif hparams.dataset == 'qqp':
            self.load_QQP(hparams)
        elif hparams.dataset == 'mnli_m':
            self.load_MNLIm(hparams)
        elif hparams.dataset == 'mnli_mm':
            self.load_MNLImm(hparams)
        elif hparams.dataset == 'qnli':
            self.load_QNLI(hparams)
        elif hparams.dataset == 'rte':
            self.load_RTE(hparams)
        elif hparams.dataset == 'wnli':
            self.load_WNLI(hparams)
        elif hparams.dataset == 'imdb_reviews':
            self.load_imdb(hparams)
        elif hparams.dataset == 'test':
            self.load_test(hparams)
        else:
            raise ValueError('unimplemented')

        self.num_train = len(self.train_texts)
        self.text_size = hparams.max_seq_length # 128
        self.dataset = hparams.dataset

        if self.num_classes < 100: # --- classification
            self.train_labels = np.eye(self.num_classes)[np.array(
                self.train_labels, dtype=np.int32)]
            self.val_labels = np.eye(self.num_classes)[np.array(
                self.val_labels, dtype=np.int32)]
            self.test_labels = np.eye(self.num_classes)[np.array(
                self.test_labels, dtype=np.int32)]
        else: # --- regression
            self.train_labels = np.array(self.train_labels, dtype=np.float32)[:, np.newaxis]
            self.val_labels = np.array(self.val_labels, dtype=np.float32)[:, np.newaxis]
            self.test_labels = np.array(self.test_labels, dtype=np.float32)[:, np.newaxis]

        assert len(self.train_texts) == len(self.train_labels)
        assert len(self.val_texts) == len(self.val_labels)
        assert len(self.test_texts) == len(self.test_labels)


    def get_training_data(self):
        return self.train_texts, self.train_labels

    def get_policy(self):
        return self.policy

    def next_batch(self, nn_database, iteration=None):
        """Return the next minibatch of augmented data."""
        next_train_index = self.curr_train_index + self.hparams.batch_size
        if next_train_index > self.num_train:
            epoch = self.epochs + 1
            self.reset()
            self.epochs = epoch

        batched_data = (
            self.train_texts[self.curr_train_index:self.curr_train_index +
                              self.hparams.batch_size],
            self.train_labels[self.curr_train_index:self.curr_train_index +
                              self.hparams.batch_size])

        final_txts = []
        final_emb_policies = []

        dset = self.hparams.dataset
        exp_size = self.hparams.expsize

        texts, labels = batched_data
        for data in texts:
            if not self.hparams.no_aug:
                if not self.hparams.use_hp_policy: # --- train with existing policy
                    # --- apply autoaugment policy
                    epoch_policy = self.good_policies[np.random.choice(
                        len(self.good_policies))]
                    final_txt, embd_policies = self.augmentation_transforms.apply_policy(
                        epoch_policy,
                        data,
                        dset=dset,
                        exp_size=exp_size,
                        nn_database=nn_database
                    )
                else:
                    if isinstance(self.policy[0], list):
                        # --- single policy
                        if self.hparams.flatten:
                            final_txt, embd_policies = self.augmentation_transforms.apply_policy(
                                self.policy[random.randint(
                                    0,
                                    len(self.policy) - 1)],
                                data,
                                self.hparams.aug_policy,
                                dset,
                                exp_size,
                                nn_database=nn_database
                            )
                        else:
                            final_txt, embd_policies = self.augmentation_transforms.apply_policy(
                                self.policy[iteration],
                                data,
                                self.hparams.aug_policy,
                                dset,
                                exp_size=exp_size,
                                nn_database=nn_database
                            )
                    elif isinstance(self.policy, list): # --- search new policy
                        final_txt, embd_policies = self.augmentation_transforms.apply_policy(
                            self.policy,
                            data,
                            self.hparams.aug_policy,
                            dset,
                            exp_size=exp_size,
                            nn_database=nn_database
                        )
                    else:
                        raise ValueError('Unknown policy.')
            else: # -- no augmentation
                embd_policies = None
                final_txt = data

            final_txts.append(final_txt)
            final_emb_policies.append(embd_policies)

        batched_data = (final_txts, labels)
        self.curr_train_index += self.hparams.batch_size
        return batched_data, final_emb_policies

    def reset(self):
        """Reset training data and index into the training data."""
        self.epochs = 0
        data = list(zip(self.train_texts, self.train_labels))
        random.shuffle(data) # --- shuffle the training data
        train_texts_perm, train_labels_perm = zip(*data)
        self.train_texts = list(train_texts_perm)
        self.train_labels = np.array(train_labels_perm)

        self.curr_train_index = 0
