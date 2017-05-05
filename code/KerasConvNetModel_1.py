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
import sys
import nlp_utils
import numpy as np
import threading
import models as models
from models import threadsafe_generator
from scipy import spatial

"""
Lstm network (4 inputs <300,300,1>) with shared weights of q,a input in word2vect and other input cosine similarity scalar
"""
class KerasConvNetModel_1(models.PassageRetrievalModel):

    def __init__(self,init_params):
        super(KerasConvNetModel_1, self).__init__('KerasConvNetModel_1', init_params['runid'])
        self.w2vutil = init_params['w2v']
        self.w2v = self.w2vutil.getWord2VectModel()
        self.params = init_params['params']
        self.max_words = self.params['method_params']['max_words']
        self.positive_rate = self.params['method_params']['positive_rate']
        self.prep_step = [ str(step) for step in init_params['params']['preprocess_steps'] ]

    def buildCosineSimMatrix(self, questions_answer_pairs, max_terms=20):
        #Construct Question Answer Matrix Pairs
        x = []
        y = []
        for pair in questions_answer_pairs:
            q_vect = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(pair.q,self.prep_step))
            #print q
            cos_matrix = []
            for i, q_i in enumerate(q_vect):
                if i==max_terms:
                    break
                sim_qi_a = []
                #print q_i
                #a = nlp_utils.data_preprocess(pair.a,self.prep_step)
                a_vect = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(pair.a,self.prep_step))
                #print a
                for k, a_k in enumerate(a_vect):
                    if k==max_terms:
                        break
                    #print a_k
                    sim_qi_a += [spatial.distance.cosine(q_i, a_k)]
                cos_matrix += [sim_qi_a]
            cos_matrix = np.array(cos_matrix)
            shape_cos_matrix = cos_matrix.shape
            cos_matrix = np.pad(cos_matrix, ((0,max_terms-shape_cos_matrix[0]),(0,max_terms-shape_cos_matrix[1])), mode='constant')
            if np.isnan(cos_matrix).any():
                print 'ERROR IS NAN: ',pair
            x.append( np.expand_dims(cos_matrix,0) )
            y.append( pair.l )
        return x, y

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
    def generateXYBatches(self, ds, dataset, num_samples, positive_rate=0.5):
        samples = ds.get_random_samples(dataset, num_samples, self.positive_rate)
        while 1:
            x, y = self.buildCosineSimMatrix(samples,max_terms=self.max_words)
            yield ( np.array(x), np.array(y))

    def train(self, ds, qa_pair):
        self.model = self.load_model()
        # checkpoints
        self.best_params=self.params['working_folder']+self.params['expriment_id'].replace('$runid',self.runid)+"_best.hdf5"
        #output the model weights each time an improvement is observed during training
        checkpoint = ModelCheckpoint(self.best_params, monitor=self.params['method_params']['monitor'], verbose=self.params['method_params']['verbose'], save_best_only=True, mode='auto')
        #stops if the model is not learning at any point
        earlyStopping=EarlyStopping(monitor=self.params['method_params']['monitor'], patience=self.params['method_params']['patience'], verbose=self.params['method_params']['verbose'], mode='auto')
        epochs_number = self.params['method_params']['epochs']
        batch_size = self.params['method_params']['batch_size']
        validation_size = self.params['method_params']['validation_size']

        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        history = self.model.fit_generator(
                    self.generateXYBatches(ds, qa_pair['train'], batch_size, positive_rate=self.positive_rate),
                    samples_per_epoch=batch_size,
                    validation_data=self.generateXYBatches(ds, qa_pair['validate'], validation_size, positive_rate=self.positive_rate),
                    nb_val_samples=validation_size,
                    nb_epoch=epochs_number,
                    callbacks=[checkpoint, earlyStopping])

        return history


    def test(self, ds, qa_pairs):
        #reload best weights
        self.model.load_weights(self.best_params)
        #Construct Test dataset
        test_qxa, test_l = self.buildCosineSimMatrix(qa_pairs, max_terms=self.max_words)
        predictions = self.model.predict(np.array(test_qxa))
        return predictions
