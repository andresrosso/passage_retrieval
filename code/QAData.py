import numpy as np
import random
from random import shuffle

class QAPair():
    def __init__(self, qi, q, ai, a, l):
        self.qi = qi
        self.q = q
        self.ai = ai
        self.a = a
        self.l = l

    def __repr__(self):
        return 'qi('+str(self.qi)+') '+'ai('+str(self.ai)+')'+' '+str(self.l)

class QADataSet(object):

    def __init__(self, name):
        self.name = name
        self.patitions = []
        self.questions = {}

    def get_stats(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def build_qa_pairs(self, dataset):
        raise NotImplementedError("Subclass must implement abstract method")


    def get_random_samples(self, dataset, samples, positive_rate=0.5):
        num_pos_samples = int(samples*(positive_rate))
        positiveSamples = [ q for q in dataset if q.l==1 ]
        negativeSamples = [ q for q in dataset if q.l==0 ]
        data = random.sample(positiveSamples, num_pos_samples)+random.sample(negativeSamples, samples-num_pos_samples)
        shuffle(data)
        return data

#WikiQA Imports
import data_stats
from wikiqahelper import load_questions_from_file
import QAData

class WikiQADataSet(QADataSet):

    def __init__(self):
        QADataSet.__init__(self,'WikiQA')
        self.patitions = ['train','validate','test']
        self.questions['train'], vocabulary, idf = load_questions_from_file('train', -1)
        self.questions['validate'], vocabulary, idf = load_questions_from_file('validate', -1)
        self.questions['test'], vocabulary, idf = load_questions_from_file('test', -1)

    def get_stats(self):
        return data_stats.getStats()

    '''
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        questions_answer_pairs = []
        for k, test_q_k in enumerate(dataset):
            q = test_q_k.question
            for i, a_i in enumerate(test_q_k.answers):
                is_correct = 1 if i in test_q_k.correct_answer else 0
                questions_answer_pairs += [QAPair(k+1, q, i, a_i, is_correct)]
        return questions_answer_pairs


#WikiQA Imports
import trecqajakanahelper as trec

class TrecDataSet(QADataSet):

    def __init__(self):
        QADataSet.__init__(self,'TrecDataSet')
        self.patitions = ['train','validate','test']
        self.questions['train'] = ( trec.load_data(trec.datasets['train']) )
        self.questions['validate'] = ( trec.load_data(trec.datasets['validate']) )
        self.questions['test'] = ( trec.load_data(trec.datasets['test']) )

    def get_stats(self):
        return 'Train: '+str(len(self.questions['train'][0]))+', Test: '+str(len(self.questions['test'][0]))+', Validate: '+str(len(self.questions['validate'][0]))

    '''
    Return a tuple of (quesion_id, question, answer_id, answer, label)
    '''
    def build_qa_pairs(self, dataset):
        #Construct Question Answer Pairs
        idx_ques, questions, idx_ans, answers, labels = dataset
        questions_answer_pairs = trec.buildQAPairs( idx_ques, questions, idx_ans, answers, labels )
        return questions_answer_pairs


class TrecDataSet_TrainAll(TrecDataSet):

    def __init__(self):
        TrecDataSet.__init__(self)
        self.name = 'TrecDataSet_TrainAll'
        self.questions['train'] = ( trec.load_data(trec.datasets['train-all']) )

class DataSetFactory():
    @staticmethod
    def loadDataSet(targetclass):
        return globals()[targetclass]()
