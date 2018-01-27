#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 11:16:15 2017

@author: mohsin
"""
from __future__ import print_function, unicode_literals

import os
import pandas as pd
import numpy as np
np.random.seed(786)
import random
import sys
import hashlib
import string
import unicodedata
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics

import gc
from collections import defaultdict, Counter
from itertools import chain

import keras
from keras.models import Sequential, Model
from keras.layers import PReLu, LeakyReLu
from keras.layers.embedding impor Embedding
from keras.optimizers import RMSprop, Adam


#Functions for tokenizing data
#class Tokenizer():

def unicodeToAscii(s):
    return  unicodedata.normalize('NFKC', s)

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"'", r"", s)
    s = re.sub(r"[.!?:;,]", r" ", s)
    s = re.sub(r"-", r"", s)
    s = re.sub(r"[^0-9a-zA-Z.!?]+", r" ", s)
    return s

def ngram_tokenize(sent, ngrams):
    input_list = normalizeString(sent).split()
    #input_list = [word for word in input_list if word not in stop_words]
    s = input_list.copy()
    for i in range(2, ngrams+1):
        s += [' '.join(input_list[j:j+i]) for j in range(len(input_list)-i + 1)]
        #s += list((zip(*[input_list[j:] for j in range(i)])))
    return s

def remove_stops(tokens, stop_list):
    return [token for token in tokens if token not in stop_list]


