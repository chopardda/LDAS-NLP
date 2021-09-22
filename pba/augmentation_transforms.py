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
"""Transforms used in the Augmentation Policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import random


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func, name):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, txt):
        return self.f(txt) # --- apply function func on txt


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name, xform_fn):
        self.name = name
        self.xform = xform_fn

    def array_transformer(self, probability, level, txt_size):
        """Builds augmentation function which returns resulting image and whether augmentation was applied."""

        def return_function(im):
            if random.random() < probability:
                if 'text_size' in inspect.getargspec(self.xform).args:
                    im = self.xform(im, level, txt_size) # apply xform on im
                else:
                    im = self.xform(im, level) # apply xform on im
                res = True # --- augmentation was correctly applied
            return im

        name = self.name + '({:.1f},{})'.format(probability, level)
        return TransformFunction(return_function, name)