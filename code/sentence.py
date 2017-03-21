from pprint import pprint
import re
from math import log10
from collections import defaultdict
import globals
import numpy as np
import sys

def get_sentence_words(sentence):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.strip().lower()

    return string.split()


def get_normalized_numbers(l_words):
    norm_words = []

    for word in l_words:
        if contains_digits(word):
            norm_words.append(re.sub("\d", "0", word))

    return norm_words