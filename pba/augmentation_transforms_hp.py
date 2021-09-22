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
"""Transforms used in the PBA Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import copy
import collections
import inspect
import random
import csv
import os

import numpy as np
from nltk.corpus import wordnet
from googletrans import Translator

from pba.augmentation_transforms import TransformFunction
from pba.data_preprocessing import BERT_preprocess

PARAMETER_MAX = 1  # --- The max value of the magnitude parameter


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
  Args:
    level: Level of the operation that will be between [0, `PARAMETER_MAX`].
    maxval: Maximum value that the operation can have. This will be scaled
      to level/PARAMETER_MAX.
  Returns:
    A float that results from scaling `maxval` according to `level`.
  """
    return float(level) * maxval / PARAMETER_MAX


def apply_policy(policy, txt, aug_policy, dset, exp_size, nn_database, verbose=False):
    """Apply the `policy` to the numpy `txt`.

  Args:
    policy: A list of tuples with the form (name, probability, level) where
      `name` is the name of the augmentation operation to apply, `probability`
      is the probability of applying the operation and `level` is what strength
      the operation to apply.
    txt: List of size (batch_size,)---each element being a string---that will have `policy` applied to it.
    aug_policy: Augmentation policy to use.
    dset: Name of dataset (for contextual augm)
    exp_size: Size of experiment (for contextual augm)
    verbose: Whether to print applied augmentations.

  Returns:
    The result of applying `policy` to `txt`.
  """
    if aug_policy == 'default_aug_policy':
        count = np.random.choice([0, 1, 2, 3], p=[0.2, 0.3, 0.5, 0.0])
    else:
        raise ValueError('Unknown aug policy.')
    if count != 0:
        policy = copy.copy(policy)
        random.shuffle(policy)

        # --- Sort policies into two categories: those that must be applied directly on text, and those that must be applied later on embeddings
        dataaugm_text_first = ['ContextualAugmentation', 'Backtranslation']
        dataaugm_texts = ['RandomSwap', 'RandomDeletion', 'RandomInsertionSyn', 'RandomInsertionVocab',
                          'SynonymReplacement', 'HypernymReplacement', 'NearestNeighbour']
        dataaugm_embs = ['EmbNormalNoise', 'EmbUniformNoise']

        not_modified_flag = True
        embd_policies = []
        for xform in policy:
            assert len(xform) == 3
            name, probability, level = xform
            assert 0. <= probability <= 1.
            assert 0 <= level <= PARAMETER_MAX
            if name in dataaugm_text_first and not_modified_flag:
                xform_fn = NAME_TO_TRANSFORM[name].array_transformer(
                    probability, level, dset, exp_size, nn_database)
                if type(txt) == str:
                    txt, res = xform_fn(txt) # --- apply data augmentation on single sentence
                elif type(txt) == tuple:
                    sent_1, res_1 = xform_fn(txt[0]) # --- apply data augmentation on each of the two sentences
                    sent_2, res_2 = xform_fn(txt[1]) # --- apply data augmentation on each of the two sentences
                    txt = (sent_1, sent_2)
                    res = res_1 and res_2
                if verbose and res:
                    print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
                count -= res  # count-=1 if data augmentation successful
                not_modified_flag = False
            elif name in dataaugm_texts:
                xform_fn = NAME_TO_TRANSFORM[name].array_transformer(
                    probability, level, dset, exp_size, nn_database)
                if type(txt) == str:
                    txt, res = xform_fn(txt)  # apply data augmentation on text
                elif type(txt) == tuple:
                    sent_1, res_1 = xform_fn(txt[0]) # --- apply data augmentation on each of the two sentences
                    sent_2, res_2 = xform_fn(txt[1]) # --- apply data augmentation on each of the two sentences
                    txt = (sent_1, sent_2)
                    res = res_1 and res_2
                if verbose and res:
                    print("Op: {}, Magnitude: {}, Prob: {}".format(name, level, probability))
                count -= res  # count-=1 if data augmentation successful
                not_modified_flag = False
            elif name in dataaugm_embs:
                embd_policies.append(xform)
                count -= 1
            assert count >= 0
            if count == 0:
                break

        return txt, embd_policies

    else:  # if count==0, i.e. do not apply any augmentation functions but transform to embeddings anyway

        embd_policies = None
        return txt, embd_policies

class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def array_transformer(self, probability, level, dset, exp_size, nn_database):
        """Builds augmentation function which returns resulting text and whether augmentation was applied."""

        def return_function(txt):
            res = False
            if random.random() < probability:
                if 'nn_database' in inspect.getfullargspec(self.xform).args:
                    txt = self.xform(txt, level, nn_database)  # apply xform on txt
                elif 'dataset' in inspect.getfullargspec(self.xform).args:
                    txt = self.xform(txt, level, dset, exp_size)  # apply xform on txt
                else:
                    txt = self.xform(txt, level)  # apply xform on txt
                res = True  # augmentation was applied
            return txt, res

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)

    def str(self):
        return self.name

#stop words list
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
			'ours', 'ourselves', 'you', 'your', 'yours',
			'yourself', 'yourselves', 'he', 'him', 'his',
			'himself', 'she', 'her', 'hers', 'herself',
			'it', 'its', 'itself', 'they', 'them', 'their',
			'theirs', 'themselves', 'what', 'which', 'who',
			'whom', 'this', 'that', 'these', 'those', 'am',
			'is', 'are', 'was', 'were', 'be', 'been', 'being',
			'have', 'has', 'had', 'having', 'do', 'does', 'did',
			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
			'because', 'as', 'until', 'while', 'of', 'at',
			'by', 'for', 'with', 'about', 'against', 'between',
			'into', 'through', 'during', 'before', 'after',
			'above', 'below', 'to', 'from', 'up', 'down', 'in',
			'out', 'on', 'off', 'over', 'under', 'again',
			'further', 'then', 'once', 'here', 'there', 'when',
			'where', 'why', 'how', 'all', 'any', 'both', 'each',
			'few', 'more', 'most', 'other', 'some', 'such', 'no',
			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
			'very', 's', 't', 'can', 'will', 'just', 'don',
			'should', 'now', '']

# ################## Transform Functions ##################
import re


def reg_tokenize(text):
    # note this does not separate contractions ('I'm', 'didn't', ...)
    tokens = re.findall(r"[\w']+|[.,!?;]", text)
    return tokens

########################################################################
# Random swap
# Randomly swap two words in the sentence n times
########################################################################

def swap_word(tokens_list):
    """Randomly swap two words from list of tokens `tokens_list` n times [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    tokens_list: A list of str

  Returns:
    The result of randomly swapping tokens in `tokens_list` with given level of magnitude.
  """
    random_idx_1 = random.randint(0, len(tokens_list)- 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:  # multiple attempts (up to 3) to get different values for the two indices to swap
        random_idx_2 = random.randint(0, len(tokens_list) - 1)
        counter += 1
        if counter > 3:
            return tokens_list
    tokens_list[random_idx_1], tokens_list[random_idx_2] = tokens_list[random_idx_2], tokens_list[random_idx_1]
    return tokens_list

def _random_swap_impl(txt, level):
    """Randomly swap two words from sentence `txt` n times [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    txt: A str that corresponds to one sample
    level: the magnitude of the augmentation (percentage of words to swap)

  Returns:
    The result of randomly swapping words in `txt` with given level of magnitude.
  """
    tokens = txt.split() # --- transform sentence to list of tokens (including punctuation)

    # --- level is percentage of swaps (proportional to number of tokens)
    level = float_parameter(level, 0.25) # --- scales level between 0 and 0.2
    n = int(level * len(tokens)) # --- n originally = int(0.1*len(tokens))

    for _ in range(n):
        tokens = swap_word(tokens)
    new_txt = " ".join(tokens)
    assert (type(new_txt) == str)

    return new_txt

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def _random_deletion_impl(txt, level):
    """Randomly delete words from the sentence `txt`[based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    txt: A str that corresponds to one sample
    level: the magnitude of the augmentation (percentage of words to delete, always keep one word at the minimum)

  Returns:
    The result of randomly deleted a level-percentage of words in `txt`.
  """
    tokens = txt.split()  # from sentence to list of tokens

    # --- if there's only one word, don't delete it
    if len(tokens) == 0:
        print('+++++ Random deletion : len(tokens) == 0')
        print(txt)
        return txt  # return the only word as a str

    # --- if there's only one word, don't delete it
    if len(tokens) == 1:
        return txt

    # --- randomly delete words with probability p
    p = float_parameter(level, 0.25)  # --- scale level between 0 and 0.25 (probability of deleting a word)

    new_tokens = []
    for token in tokens:
        r = random.uniform(0, 1)
        if r > p: # --- originally p is 0.1
            new_tokens.append(token)

    # if all words end up being deleted, just return a random word [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]
    if len(new_tokens) == 0:
        rand_int = random.randint(0, len(tokens)-1)
        return tokens[rand_int]  # return the only word as str

    assert (type(" ".join(new_tokens)) == str)

    return " ".join(new_tokens)  # return the remaining words as a str


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def add_word_syn(tokens):
    """Find a random synonym of a random word in the sentence that is not a stop word.
    Insert that synonym into a random position in the sentence. Do this n times.
     [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    tokens: list of str

  Returns:
   Tokens with randomly inserted synonyms
  """
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = tokens[random.randint(0, len(tokens)-1)]
        synonyms = get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return # -- if cannot find synonym after 10 attempts
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(tokens)-1)
    tokens.insert(random_idx, random_synonym)


def _random_insertion_syn_impl(txt, level):
    """Randomly insert synonyms into txt [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    txt: str
    level: magnitude parameter

  Returns:
    Augmented txt with randomly inserted synonyms
  """

    tokens = txt.split()
    scaled_level = float_parameter(level, 0.25)  # --- scale level between 0 and 0.25 (percentage of words to delete)

    n = int(scaled_level * len(tokens)) # --- n originally equals int(0.1*len(tokens))
    for ii in range(n):
        add_word_syn(tokens)

    assert (type(" ".join(tokens)) == str)

    return " ".join(tokens)


# --- Find a random token from BERT vocab that does not start with '##'. Insert that token into a random position in the sentence. Do this n times
def add_word_vocab(tokens):
    # [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]
    bert_vocab_file = os.path.join(os.path.realpath('../../..'),'datasets','pretrained_models','bert_base','vocab.txt')

    vocab = load_vocab(bert_vocab_file) #TODO: improve
    new_word = []
    while len(new_word) < 1:
        random_word_idx = random.randint(1996,29611) # 999:1066 (punct, digits, ...) + 1996:29611 ("words")
        random_word = list(vocab.items())[random_word_idx][0]
        if '##' in random_word:
            continue
        else:
            new_word = random_word
    random_idx = random.randint(0, len(tokens)-1)
    tokens.insert(random_idx, random_word)


def _random_insertion_vocab_impl(txt, level):
     '''based on [https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py] '''
     tokens = txt.split()
     scaled_level = float_parameter(level, 0.25)  # --- scale level between 0 and 0.25 (percentage of words to delete)

     n = int(scaled_level * len(tokens)) # --- n originally equals int(0.1*len(tokens))
     for ii in range(n):
         add_word_vocab(tokens)

     assert (type(" ".join(tokens)) == str)
     return " ".join(tokens)

########################################################################
# Synonym and hypernym replacement
# Replace n words in the sentence with synonyms/hypernyms from WordNet
########################################################################
def get_synonyms(word):
    """Get the list of synonyms for word from WordNet [from https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    word: str

  Returns:
    A list of synonyms of word
  """
    synonyms = set()
    for syn in wordnet.synsets(word):     # for syn in wordnet.synsets(word, pos=pos):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def get_hypernyms(word):
    """Get the list of hypernyms for word from WordNet [based on https://github.com/jasonwei20/eda_nlp/blob/master/code/eda.py]

  Args:
    word: str

  Returns:
    A list of hypernyms of word
  """

    hypernyms = set()
    try:
        for hyp in wordnet.synsets(word)[0].hypernyms():
            for l in hyp.lemmas():
                hypernym = l.name().replace("_", " ").replace("-", " ").lower()
                hypernym = "".join([char for char in hypernym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                hypernyms.add(hypernym)
        if word in hypernyms:
            hypernyms.remove(word)
        return list(hypernyms)
    except:
        return []


def _synonym_replacement_impl(txt, level):
    """Randomly replace up to n words in str `txt` with one of their synonyms [based on https://github.com/dsfsi/textaugment/blob/master/textaugment/wordnet.py]

  Args:
    txt: A str (one sample)
    level: the magnitude of the augmentation (percentage of words to replace with synonym)

  Returns:
    The result of randomly replacing 0.25*level percent of the tokens in 'txt' with one of their synonyms
  """
    tokens = txt.split()
    new_tokens = tokens.copy()

    scaled_level = float_parameter(level, 0.25)
    n = int(scaled_level*len(tokens))  # --- n originally int(0.1*len(tokens))

    if n > 0:  # --- if number of tokens to replace is zero, do not replace any word
        random_word_list = list(set([word for word in tokens if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms))
                new_tokens = [synonym if word == random_word else word for word in new_tokens]
                num_replaced += 1
            if num_replaced >= n:  # --- only replace up to n words
                break

    new_sentence = ' '.join(new_tokens)
    assert(type(new_sentence) == str)

    return new_sentence


def _hypernym_replacement_impl(txt, level):
    """Randomly replace up to n words in str `txt` with one of their hypernyms [based on https://github.com/dsfsi/textaugment/blob/master/textaugment/wordnet.py]

  Args:
    txt: A str (one sample)
    level: the magnitude of the augmentation (percentage of words to replace)

  Returns:
    The result of randomly replacing 0.25*level percent of the tokens in 'txt' with one of their hypernyms
  """

    tokens = txt.split()
    new_tokens = tokens.copy()

    scaled_level = float_parameter(level, 0.25)
    n = int(scaled_level*len(tokens))  # --- n originally int(0.1*len(tokens))

    if n > 0: # --- if number of tokens to replace is zero, do not replace any word
        random_word_list = list(set([word for word in tokens if word not in stop_words]))
        random.shuffle(random_word_list)
        num_replaced = 0
        for random_word in random_word_list:
            hypernyms = get_hypernyms(random_word)
            if len(hypernyms) >= 1:
                synonym = random.choice(list(hypernyms))
                new_tokens = [synonym if word == random_word else word for word in new_tokens]
                num_replaced += 1
            if num_replaced >= n:  # --- only replace up to n words
                break

    new_sentence = ' '.join(new_tokens)
    assert(type(new_sentence) == str)

    return new_sentence

def _nearest_neighbour_impl(txt, level, nn_database):
    """Randomly replace up to n words in str `txt` with one of their 10 nearest neighbours

  Args:
    txt: A str (one sample or one sequence of one sample)
    level: the magnitude of the augmentation (percentage of words to replace)

  Returns:
    The result of randomly replacing 0.25*level percent of the tokens in 'txt' with one of their 10 nearest neighbours
  """
    tokens_list = BERT_preprocess([txt])[0]
    new_tokens = tokens_list.copy()

    scaled_level = float_parameter(level, 0.25)
    n = int(scaled_level*len(tokens_list))

    if n > 0: # --- only replace if n>0
        random_token_list = list(set([word for word in tokens_list]))
        random.shuffle(random_token_list)

        num_replaced = 0
        count = 0
        for random_token in random_token_list:
            try:
                distances, neighbours, _ = nn_database.query(query_sent=txt, query_word=random_token, k=10,
                                                               filter_same_word=True, different_neighbours=True)
                sorted_neighbours = [x for _, x in sorted(zip(distances, neighbours), reverse=True)]
                if sorted_neighbours != []:
                    neighbours_idx = list(np.random.geometric(p=0.5, size=len(sorted_neighbours)) == 1)
                    nearest_neighbour = sorted_neighbours[neighbours_idx.index(True)] if True in neighbours_idx else random.choice(sorted_neighbours)
                    new_tokens = [nearest_neighbour if word == random_token else word for word in new_tokens]
                    num_replaced += 1
                if num_replaced >= n:  # --- only replace up to n words
                    break
            except:
                count += 1
                if count > 10:
                    return ' '.join(new_tokens)

    new_sentence = ' '.join(new_tokens)

    return new_sentence


def read_tsv(input_file, quotechar=None):
    with open(input_file, 'r', encoding="ISO-8859-1") as data_file:
        reader = csv.reader(data_file, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
    return lines

def _contextual_aug_impl(txt, level, dataset, exp_size):
    """Replace txt with the pre-generated contextual augmented version depending on level

  Args:
    txt: A str (one sample or one sequence of one sample)
    level: the magnitude of the augmentation

  Returns:
    The result of randomly masking 0.25*level percent of the tokens in 'txt' and using the fine-tuned conditional model to predict replacement
  """
    scaled_magn = round(level*10)

    if int(scaled_magn) == 0:
        return txt

    data_dir = os.path.join(os.path.realpath('../../..'),'datasets','augmented_data','contextual_bert') #'/home/c.c1851936/pba/datasets/augmented_data/contextual_bert'
    aug_file = os.path.join(data_dir, dataset, exp_size, 'train_aug_magn_{}.tsv'.format(scaled_magn))
    aug_data = read_tsv(aug_file)

    no_punct_samples = [x[0].replace(' ', '').replace('"""', '') for x in aug_data]
    for sample_idx in range(len(no_punct_samples)):
        if no_punct_samples[sample_idx].replace(' ', '') == txt.lower().replace(' ', '') or \
                txt.lower().replace(' ', '').find(no_punct_samples[sample_idx].replace(' ', '')) != -1:
            break
    aug_txt = random.choice(aug_data[sample_idx][1:])

    return aug_txt

def back_translation(sentence, language, translator):
  mid_translation = translator.translate(sentence, src='en', dest=language)
  return translator.translate(mid_translation.text, src=language, dest='en').text

magnitudes_languages = {1: ['pt', 'it', 'fr', 'cs', 'sv'],
                        2: ['nl', 'mt', 'pl', 'ro', 'ru'],
                        3: ['af', 'be', 'sk', 'da', 'id'],
                        4: ['de', 'sq', 'bg', 'ja', 'es'],
                        5: ['zh-cn', 'zh-tw', 'hr', 'fi', 'lv', 'ar', 'ml'],
                        6: ['el', 'ko', 'no', 'sr', 'tr', 'cy'],
                        7: ['gl', 'is', 'sl', 'vi'],
                        8: ['ca', 'et', 'tl', 'hu', 'sw'],
                        9: ['ga', 'th', 'iw', 'uk', 'fa'],
                        10: ['lt', 'mk', 'yi', 'hi']}

def _backtranslation_impl(txt, level, dataset, exp_size):
    """Replace txt with the pre-generated contextual augmented version depending on level

  Args:
    txt: A str (one sample or one sequence of one sample)
    level: the magnitude of the augmentation

  Returns:
    The result of randomly masking tokens in 'txt' and using the fine-tuned conditional model to predict replacement
  """
    scaled_magn = round(level*10)

    if int(scaled_magn) == 0:
        return txt

    translator = Translator()

    flag = True
    counter = 0 # --- up to 10 tries
    while flag and counter < 10:
        lang = random.choice(magnitudes_languages[scaled_magn])
        try:
            aug_txt = back_translation(txt, lang, translator) # to catch error in the rare case where translation not unique [TypeError: 'NoneType' object is not iterable]
            flag = False
        except:
            counter += 1
            time.sleep(0.8)

    if counter == 10: # --- if did not work, return original sample
        aug_txt = txt

    return aug_txt

def _emb_normal_noise_impl(txt_emb, level):
    """ Dummy function """
    return txt_emb

def _emb_uniform_noise_impl(txt_emb, level):
    """ Dummy function """
    return txt_emb

random_swap = TransformT('RandomSwap', _random_swap_impl)
random_deletion = TransformT('RandomDeletion', _random_deletion_impl)
random_insertion_syn = TransformT('RandomInsertionSyn', _random_insertion_syn_impl)
random_insertion_vocab = TransformT('RandomInsertionVocab', _random_insertion_vocab_impl)
synonym_replacement = TransformT('SynonymReplacement', _synonym_replacement_impl)
hypernym_replacement = TransformT('HypernymReplacement', _hypernym_replacement_impl)
nearest_neighbour = TransformT('NearestNeighbour', _nearest_neighbour_impl)
contextual_aug = TransformT('ContextualAugmentation', _contextual_aug_impl)
backtranslation_aug = TransformT('Backtranslation', _backtranslation_impl)
emb_normal_noise = TransformT('EmbNormalNoise', _emb_normal_noise_impl)
emb_uniform_noise = TransformT('EmbUniformNoise', _emb_uniform_noise_impl)

HP_TRANSFORMS = [
    random_swap,
    random_deletion,
    random_insertion_syn,
    random_insertion_vocab,
    synonym_replacement,
    hypernym_replacement,
    nearest_neighbour,
    contextual_aug,
    backtranslation_aug,
    emb_normal_noise,
    emb_uniform_noise
]

NAME_TO_TRANSFORM = collections.OrderedDict((t.name, t) for t in HP_TRANSFORMS)
HP_TRANSFORM_NAMES = NAME_TO_TRANSFORM.keys()
NUM_HP_TRANSFORM = len(HP_TRANSFORM_NAMES)
