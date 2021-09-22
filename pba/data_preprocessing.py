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

import unicodedata

def is_whitespace(char):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def is_control_char(char):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    # """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def cleanup_text(text):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or is_control_char(char):
        continue
      if is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

def whitespace_tokenize(text):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def strip_accents(text):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)

def is_punctuation(char):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
          (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def split_on_punc(text):
    # from UDA [https://github.com/google-research/uda/tree/master/text/bert
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]


def BERT_preprocess(data):
    # based on code from UDA [https://github.com/google-research/uda/tree/master/text/bert
    # (data must be a list of samples)

    do_lower_case = True # parameter

    # loop over samples
    clean_data = []
    for text in data:
        # clean text (remove invalid character and cleanup whitespaces)
        text = cleanup_text(text)
        # tokenize (whitespace separator)
        orig_tokens = whitespace_tokenize(text)
        # lowercase and strip accents
        split_tokens = []
        for token in orig_tokens:
            if do_lower_case:
                token = token.lower()
                token = strip_accents(token)  # strip accents (e.g., "'", "^", "`", ...)
            split_tokens.extend(split_on_punc(token))  # split punctuation to separate token
        output_tokens = whitespace_tokenize(" ".join(split_tokens)) # list of tokens ['"', 'brain', 'dead', '"', 'makes', 'no', 'sense', 'whatsoever', '.', ...]
        clean_data.append(output_tokens)

    return clean_data

