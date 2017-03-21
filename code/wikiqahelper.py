import sys
import cPickle as pickle
import copy
from os.path import exists
from os import makedirs
from pprint import pprint
from wordrepresentation import word2vecmodel as w2vec
from question import Question
from sentence import get_sentence_words, get_normalized_numbers
from collections import defaultdict
import numpy as np
import logging
import re
import sys  

import globals

def load_questions_from_file(mode, q_limit, vocabulary=None):
    questions = []
    n = 0
    idf = defaultdict(float)
    if vocabulary is None:
        vocabulary = defaultdict(float)
    else:
        print("Vocabulary passed")

    with open(globals.input_files.get(mode)) as f:
        question_text = None
        question = None
        parsed_questions = 0
        answers_count = 0

        for line in f:
            line = line.decode('latin-1')
            split_line = line.rstrip().split('\t')
            # If new question (but not the first one)
            if question_text is not None and question_text != split_line[0]:
                is_new_question = True
                questions.append(question)
                parsed_questions += 1
            else:
                is_new_question = False

            # If there was a limit, break if reached
            if -1 < parsed_questions == q_limit:
                break

            question_text = split_line[0]

            # Number of samples/documents
            n += 1

            # Add to vocabulary
            words_set = set(get_sentence_words(split_line[0]))
            words_set.update(get_sentence_words(split_line[1]))
            for word in words_set:
                vocabulary[word] += 1

            # If Word2Vec will use normalized numbers (0000),
            # update vocabulary with them
            if globals.normalize_numbers is True:
                    for i in get_normalized_numbers(words_set):
                        vocabulary[i] += 1

            # If new question entity
            if is_new_question or question is None:
                answers_count = 0
                question = Question(split_line[0], split_line[1])
            else:
                question.add_answer(split_line[1])

            # Add answer if found
            if split_line[2] == "1":
                question.add_correct_answer(answers_count)

            answers_count += 1

    # Calculate idf
    for k, v in vocabulary.items():
        idf[k] = np.log(n / vocabulary[k])

    return questions, vocabulary, idf

def buildQAPairs(dataset):
    #Construct Question Answer Pairs
    questions_answer_pairs = []
    for k, test_q_k in enumerate(dataset):
        q = test_q_k.question
        for i, a_i in enumerate(test_q_k.answers):
            is_correct = 1 if i in test_q_k.correct_answer else 0
            questions_answer_pairs += [(q, a_i, is_correct)]
    return questions_answer_pairs
