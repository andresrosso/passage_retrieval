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
from keras.layers import GlobalMaxPooling2D, Activation, Input, Dense, merge, Dropout, LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution1D, Convolution2D, GlobalMaxPooling2D, Activation, Input, Dense, merge, Dropout
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

    def buildCosineSimMatrix(self, questions_answer_pairs, max_terms=40):
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
            #Get salience score
            #sal_matrix = self.getSalienceScore(q_list,a_list,max_terms)
            #Get cosine distance
            cos_matrix = 1-spatial.distance.cdist(q_vect[0:max_terms], a_vect[0:max_terms], 'cosine')
            shape_cos_matrix = cos_matrix.shape
            cos_matrix = np.pad(cos_matrix, ((0,max_terms-shape_cos_matrix[0]),(0,max_terms-shape_cos_matrix[1])), mode='constant')
            if np.isnan(cos_matrix).any():
                print 'ERROR IS NAN: ',pair
            #x.append( np.expand_dims( np.multiply(cos_matrix,sal_matrix) ,0) )
            x.append( np.expand_dims( cos_matrix,0 ) )
            y.append( pair.l )
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
        self.model.add(GlobalMaxPooling2D())
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

        test_qxa, test_l = self.buildCosineSimMatrix(qa_pair['test'], max_terms=40)
        val_qxa, val_l   = self.buildCosineSimMatrix(qa_pair['validate'], max_terms=40)

        epochs_number = self.params['method_params']['epochs']
        batch_size = self.params['method_params']['batch_size']
        validation_size = self.params['method_params']['validation_size']

        self.model.compile(loss='mean_squared_error',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        history = self.model.fit_generator(
                    self.generateXYBatches( zip(test_qxa,test_l ), batch_size, positive_rate=self.positive_rate),
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
        test_qxa, test_l = self.buildCosineSimMatrix(qa_pairs, max_terms=self.max_words)
        predictions = self.model.predict(np.array(test_qxa))
        return predictions
