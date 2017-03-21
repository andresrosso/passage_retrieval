from gensim.parsing.preprocessing import STOPWORDS
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from nltk.corpus import wordnet
from nltk.tokenize import RegexpTokenizer
import nltk
from scipy import spatial
import numpy as np
import sys

class Word2vectUtils:
    def __init__(self, w2v_path):
        self.not_found_word = 0
        self.w2v_model = Doc2Vec.load_word2vec_format(w2v_path, binary=True)
    
    def def_noword_zeros(self, word):
        self.not_found_word += 1
        return np.zeros(300)
    
    def def_noword_random(self, word):
        self.not_found_word += 1
        np.random.seed(sum([ord( x )for x in word]))
        return np.random.rand(300)
    
    def transform2Word2Vect(self, sentence, def_noword_function=def_noword_random, MAX_WORDS=50):
        w2vect = []
        for i in range(MAX_WORDS):
            if i+1 > len(sentence):
                break
            else:
                w2vect.append(self.w2v_model[sentence[i]] if sentence[i] in self.w2v_model else def_noword_function(self, sentence[i]) )
        return w2vect
    
    def getWord2VectModel(self):
        return self.w2v_model

PREPROCESS_STEPS = ['stop_words_removal']

def to_lowercase(data):
    return data.lower()

def tokenize(data):
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer.tokenize(data)

def remove_stopwords(data):
    return [word for word in data if word not in STOPWORDS]

def data_preprocess(data, steps):
    data = tokenize(data.lower())
    if 'stop_words_removal' in steps:
        data = remove_stopwords(data)
    return data
    
def qa_preprocessing(question, answers, steps):
    question = data_preprocess(question, steps)
    for sentence in answers:
        answer_list = tokenize(sentence.lower())
        answers_tag_list.append([word for word in answer_list if word not in STOPWORDS])
    return question, answers_tag_list
    
def getQaPairAsWord2Vect(qaPair, w2v, def_noword_function, MAX_WORDS=50):
    question = qaPair[0]
    answer = qaPair[1]
    q_vect = []
    a_vect = []
    for i in range(MAX_WORDS):
        q_vect.append( transform2Word2Vect(def_noword_function, question[i]) )
        a_vect.append( transform2Word2Vect(def_noword_function, answer[i]) )
    label = qaPair[2]
    return q_vect, a_vect, label

def word2vect_sum_representation(list1, list2, w2v_model):
    sum_list1 = np.zeros(300)
    sum_list2 = np.zeros(300)
    mult_vector = np.ones(300)
    for wq in list1:
        try:
            sum_list1 += w2v_model[wq]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+wq)
    for aq in list2: 
        try:
            sum_list2 += w2v_model[aq]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+aq)
    return sum_list1, sum_list2

def word2vect_sum_representation(list1, list2, w2v_model):
    list1 = []
    list2 = []
    for wq in list1:
        try:
            list1 += [w2v_model[wq]]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+wq)
    for aq in list2: 
        try:
            list2 += [w2v_model[aq]]
        except Exception as e:
            logger.debug("Word not in word2vect vocabulary "+aq)
    return list1, list2









