# Based on code from UDA [https://github.com/google-research/uda/tree/master/text/bert]

# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
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
"""Functions and classes related to optimization (weight updates).

Part of the code is from https://github.com/google-research/bert
"""

import collections
import numpy as np
import os

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def wordpiece_tokenize(text, bert_vocab):
    """Tokenizes a piece of text into its word pieces. From UDA [https://github.com/google-research/uda/tree/master/text/bert]

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """
    max_input_chars_per_word = 100
    unk_token = "[UNK]"

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > max_input_chars_per_word:
        output_tokens.append(unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in bert_vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input. From https://github.com/google-research/bert"""
    if isinstance(text, str):
      return text
    # elif isinstance(text, bytes):
    #   return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

def convert_tokens_to_ids(vocab, tokens):
  """Converts a sequence of tokens into ids using the vocab."""
  ids = []
  for token in tokens:
    ids.append(vocab[token])
  return ids

def truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop(0) # remove first token (keep end of sequence)
    else:
      tokens_b.pop(0) # remove fist token (keep end of sequence)

def convert_data_to_features(tokenized_data, labels, label_list, max_seq_length):
    # from UAD [https://github.com/google-research/uda/tree/master/text/bert]
    bert_vocab_file = os.path.join(os.path.realpath('../../..'),'datasets','pretrained_models','bert_base','vocab.txt')
    bert_vocab = load_vocab(bert_vocab_file)

    features = []

    ii = 0
    for text in tokenized_data:
        if type(text) == list:
            text_a = text
            text_b = []
        elif type(text) == tuple:
            text_a = text[0]
            text_b = text[1]

        tokens_a = []
        tokens_b = []
        for token in text_a:
            tokens_a += wordpiece_tokenize(token, bert_vocab)
        if text_b:
            for token in text_b:
                tokens_b += wordpiece_tokenize(token, bert_vocab)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                # keep only end of sequence
                tokens_a = tokens_a[-(max_seq_length - 2):]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = convert_tokens_to_ids(bert_vocab, tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids) # list of ones of size (len(input_ids))

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          input_type_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(input_type_ids) == max_seq_length

        if labels.shape[1] > 1: # classification
            label_id = np.argmax(labels, 1) # extract label from label matrix (first column, first label, etc)
        elif labels.shape[1] == 1: # regression
            label_id = labels

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                label_id=label_id))

        ii+=1

    return features


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, input_type_ids, label_id):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.input_type_ids = input_type_ids
    self.label_id = label_id
