import QAData
from os import listdir
from os.path import isfile, join
import PassageRetrieval as pr
from json2html import *
import matplotlib
#matplotlib.use('Agg')
import pylab as plt
import nlp_utils
from keras_utils import MAPCallback
import time
import os
import json
import logging
from json_utils import JSONConnector
from QAData import DataSetFactory
import models
from passrtv_models import PassageRetrievalModel
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from keras.layers import Convolution1D, Convolution2D
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import Sequential
from keras.layers import MaxPooling2D, Convolution2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Activation, Input, Dense, merge, Dropout, LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, SimpleRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import Callback
from itertools import groupby
import sys
import pickle
import nlp_utils
import numpy as np
import threading
import models as models
from models import threadsafe_generator
from scipy import spatial
import nltk
import random
from random import shuffle

"""
Convolutional NN with 1d input tensor cosine sim input matrix element wise multiplied by salience matrix
"""
class KerasConvNetModel_8(models.PassageRetrievalModel):

    def __init__(self,init_params):
        super(KerasConvNetModel_8, self).__init__('KerasConvNetModel_8', init_params['runid'])
        self.w2vutil = init_params['w2v']
        self.w2v = self.w2vutil.getWord2VectModel()
        self.params = init_params['params']
        self.max_words = self.params['method_params']['max_words']
        self.positive_rate = self.params['method_params']['positive_rate']
        self.prep_step = [ str(step) for step in init_params['params']['preprocess_steps'] ]

    def getSalienceScore(self, qv, av, maxterms=40):
            score = 0
            imp_postag = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'NN', 'NNS', 'NNP', 'NNPS', 'JJ'])
            #imp_postag = set(['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP', 'WRB', 'NN', 'NNS', 'NNP', 'NNPS', 'MD'])
            #imp_postag = set(['WRB','VB', 'VBD', 'VBG', 'VBN', 'VBP','VBZ', 'WDT', 'WP', 'WRB', 'NN', 'NNS', 'NNP', 'NNPS', 'MD'])
            pq = nltk.pos_tag(qv)
            pa = nltk.pos_tag(av)
            out_m = np.zeros( (maxterms, maxterms) )
            if len(pq)>maxterms:
                pq = pq[0:maxterms]
            if len(pa)>maxterms:
                pa = pa[0:maxterms]
            wq_m = np.zeros((maxterms,maxterms))
            wa_m = np.zeros((maxterms,maxterms))

            pq_l = [len(set([qt[1]]).intersection(imp_postag))+1 for qt in pq]
            pa_l = [len(set([at[1]]).intersection(imp_postag))+1 for at in pa]

            wq_m[0:len(pa_l) , 0:len(pq_l)]=pq_l
            wa_m[0:len(pq_l) , 0:len(pa_l)]=pa_l

            out_m = (wq_m.T + wa_m)/4
            return out_m[0:maxterms,0:maxterms]

    def buildCosineSimMatrix(self, questions_answer_pairs, ordered_matrix=1, salience_weight=0, max_terms=40):
        #Construct Question Answer Matrix Pairs
        x = []
        y = []
        for pair in questions_answer_pairs:
            #Question Processin
            q_list = nlp_utils.data_preprocess(pair.q,self.prep_step)
            q_vect = self.w2vutil.transform2Word2Vect(q_list)
            #Answer processing
            a_list = nlp_utils.data_preprocess(pair.a,self.prep_step)
            a_vect = self.w2vutil.transform2Word2Vect(a_list)
            #Get cosine distance
            #distance = np.absolute( spatial.distance.cdist(q_vect[0:max_terms], a_vect[0:max_terms], 'cosine') )
            ''' with that param the MAP and Loss are  highly correlated
                and the improvement in MAP is very fast, but in test the results are almost equal
                cos_matrix = 1 - (1/(1+np.exp(-distance*3)))
                it changes a bit with
                cos_matrix = 1 - (1/(1+np.exp(-distance*2)))

            distance = spatial.distance.cdist(q_vect[0:max_terms], a_vect[0:max_terms],
              lambda u, v: (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                    if (np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))) >= 0
                    else -1.0*(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))) )
                    '''
            if len(q_vect) < 1 or len(a_vect) < 1:
                cos_matrix = np.zeros( (max_terms,max_terms) )
            else:
                distance = spatial.distance.cdist(q_vect[0:max_terms], a_vect[0:max_terms], 'cosine')
                #cos_matrix = 1 - distance
                cos_matrix = 1 - distance/2
            shape_cos_matrix = cos_matrix.shape
            cos_matrix = np.pad(cos_matrix, ((0,max_terms-shape_cos_matrix[0]),(0,max_terms-shape_cos_matrix[1])), mode='constant')
            if np.isnan(cos_matrix).any():
                print 'ERROR IS NAN: ',pair
            if salience_weight == 1:
                #Get salience score
                sal_matrix = self.getSalienceScore(q_list,a_list,max_terms)
                cos_matrix = np.multiply(cos_matrix,sal_matrix)
            #x.append( np.expand_dims( np.multiply(cos_matrix,sal_matrix) ,0) )
            if ordered_matrix == 1:
                cos_matrix.sort()
                cos_matrix = cos_matrix[::-1]
                cos_matrix = cos_matrix[:,::-1]
            y.append( pair.l )
            x.append( np.expand_dims( cos_matrix,0 ) )
        return np.array(x), np.array(y)

    def load_model(self):
        self.model = Sequential()
        mp = self.params['method_params']
        self.model.add(Convolution2D(
            nb_filter=mp['convolution_2d']['nb_filter'],
            nb_row=mp['convolution_2d']['nb_row'],
            nb_col=mp['convolution_2d']['nb_col'],
            subsample=(mp['convolution_2d']['subsample'], mp['convolution_2d']['subsample']),
            border_mode=mp['convolution_2d']['border_mode'],
            activation=mp['convolution_2d']['activation'],
            input_shape=(1, self.max_words, self.max_words)))
        self.model.add(Activation(mp['activation_2nd_Layer']))
        #self.model.add(MaxPooling2D((5,5)))
        self.model.add(GlobalMaxPooling2D())
        #self.model.add(GlobalAveragePooling2D())
        self.model.add(Dense(mp['dense_4th_Layer']))
        self.model.add(Dropout(mp['dropout']))
        self.model.add(Dense(mp['dense_6th_layer']))
        self.model.add(Activation(mp['end_layer_activation']))
        return self.model

    @threadsafe_generator
    def generateXYBatches(self, samples_xy, num_samples, positive_rate=0.5):
        num_pos_samples = int(num_samples*(positive_rate))
        positiveSamples = [ q for q in samples_xy if q[1]==1 ]
        negativeSamples = [ q for q in samples_xy if q[1]==0 ]
        samples_xy = random.sample(positiveSamples, num_pos_samples)+random.sample(negativeSamples, num_samples-num_pos_samples)
        shuffle(samples_xy)
        while 1:
            x,y = zip(*samples_xy)
            #x, y = self.buildCosineSimMatrix(samples,max_terms=self.max_words)
            yield ( np.array(x), np.array(y) )

    def train(self, ds, qa_pair):
        self.model = self.load_model()
        self.best_params=self.params['working_folder']+self.params['expriment_id'].replace('$runid',self.runid)+"_best.hdf5"
        map_callback = MAPCallback(qa_pair['validate'], self.max_words, self.buildCosineSimMatrix, self.best_params)
        self.use_salience = self.params['method_params']['use_salience']

        train_qxa, train_l = self.buildCosineSimMatrix(qa_pair['train'], self.use_salience, max_terms=self.max_words )
        val_qxa, val_l   = self.buildCosineSimMatrix(qa_pair['validate'], self.use_salience, max_terms=self.max_words )

        epochs_number = self.params['method_params']['epochs']
        batch_size = self.params['method_params']['batch_size']
        validation_size = self.params['method_params']['validation_size']

        self.model.compile(
                      loss=self.params['method_params']['loss'],
                      optimizer=self.params['method_params']['optimizer'],
                      metrics=['accuracy'])

        history = self.model.fit_generator(
                    self.generateXYBatches( zip(train_qxa, train_l ), batch_size, positive_rate=self.positive_rate),
                    samples_per_epoch=batch_size,
                    validation_data=self.generateXYBatches( zip(val_qxa,val_l), validation_size, positive_rate=self.positive_rate),
                    nb_val_samples=validation_size,
                    nb_epoch=epochs_number,
                    callbacks=[map_callback]
                    )

        #Add MAP and MRR to history
        history.history['map'] = map_callback.map_score
        history.history['mrr'] = map_callback.mrr_score
        return history

    def test(self, ds, qa_pairs):
        #reload best weights
        self.model.load_weights(self.best_params)
        #Construct Test dataset
        test_qxa, test_l = self.buildCosineSimMatrix(qa_pairs, self.use_salience, max_terms=self.max_words)
        predictions = self.model.predict(np.array(test_qxa))
        return predictions
