from passrtv_models import PassageRetrievalModel 
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from keras.layers import Convolution1D, Convolution2D
from keras.layers import Input, Embedding, merge, Flatten, SimpleRNN
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.models import Sequential
from keras.layers import GlobalMaxPooling2D, Activation, Input, Dense, merge, Dropout, LSTM
import nlp_utils
import numpy as np
import threading
import models as models
from models import threadsafe_generator
from scipy import spatial

"""
Lstm network (3 inputs <300,300,1>) with shared weights of q,a input in word2vect and other input cosine similarity scalar
"""
class KerasLstmModel_3(models.PassageRetrievalModel):
    
    def __init__(self,init_params):
        super(KerasLstmModel_3, self).__init__('KerasLstmModel_3', init_params['runid'])
        self.w2vutil = init_params['w2v']
        self.w2v = self.w2vutil.getWord2VectModel()
        self.params = init_params['params']
        self.prep_step = [ str(step) for step in init_params['params']['preprocess_steps'] ]
    
    def proc_third_input(self, q, a):
        q_w2v = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(q,self.prep_step)) \
                if len(self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(q,self.prep_step)))>0 else [np.zeros(300)]
        a_w2v = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(a,self.prep_step)) \
                if len(self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(a,self.prep_step)))>0 else [np.zeros(300)]
        #res = np.concatenate((sum(np.array(q_w2v)), sum(np.array(a_w2v)) ))
        res = spatial.distance.cosine( sum(np.array(q_w2v)), sum(np.array(a_w2v)) )
        return res
    
    def load_model(self):
        self.timesteps = self.params['method_params']['time_steps']
        question_input = Input(shape=(self.timesteps, self.params['method_params']['q_input_size']))
        answer_input = Input(shape=(self.timesteps, self.params['method_params']['a_input_size']))
        third_input = Input(shape=(self.params['method_params']['third_input_size'],))
        shared_lstm = LSTM(self.params['method_params']['shared_lstm_units'])
        encoded_q = shared_lstm(question_input)
        encoded_a = shared_lstm(answer_input)
        merged_vector = merge([encoded_q, encoded_a], mode='concat', concat_axis=-1)
        sim = Dense(self.params['method_params']['merged_vector_units'], activation='sigmoid')(merged_vector)
        sim_w2v = Dense(self.params['method_params']['merged_vector_units'], activation='sigmoid')(third_input)
        merged_vector_2 = merge([sim, sim_w2v], mode='concat', concat_axis=-1)
        sim2 = Dense(self.params['method_params']['final_merge_units'], activation='sigmoid')(merged_vector_2)
        drop = Dropout(self.params['method_params']['final_dropout'])(sim2)
        sim3 = Dense(self.params['method_params']['final_dense_units'], activation='sigmoid')(drop)
        predictions = Dense(1, activation='sigmoid')(sim3)
        model = Model(input=[question_input, answer_input, third_input], output=predictions)
        model.compile(optimizer=self.params['method_params']['optimizer'],
                      loss=self.params['method_params']['loss'],# kullback_leibler_divergence mean_squared_error
                      metrics=['accuracy'])
        #SVG(model_to_dot(model).create(prog='dot', format='svg'))
        print model.summary()
        return model
    
    @threadsafe_generator
    def generateXYBatches(self, qadata, dataset, samples, prep_step, proc_func, positive_rate=0.5):
        samples = qadata.get_random_samples(dataset, samples, positive_rate)
        while 1:
            train_X1 = []
            train_X2 = []
            train_X3 = []
            train_Y = []
            for qa_pair in samples:
                
                q_vect = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(qa_pair.q,prep_step))
                if len(q_vect)==0:
                    q_vect = q_vect + [np.zeros(300)]
                q_vect = np.array(q_vect)
                q_vect = np.pad(q_vect, ((0,self.timesteps-q_vect.shape[0]),(0,0)), mode='constant')
                
                a_vect = self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(qa_pair.a,prep_step))
                if len(a_vect)==0:
                    a_vect = a_vect + [np.zeros(300)]
                a_vect = np.array(a_vect)
                a_vect = np.pad(a_vect, ((0,self.timesteps-a_vect.shape[0]),(0,0)), mode='constant')
                
                train_X1.append(q_vect)
                train_X2.append(a_vect)
                train_X3.append(proc_func(qa_pair.q, qa_pair.a))
                train_Y.append( np.array(qa_pair.l) )
            #print "   Samples generated = ", len(train_X1), '  -  Validation(',validation, ')'
            #yield (train_X, train_Y)
            yield ([np.array(train_X1), np.array(train_X2), np.array(train_X3)], np.array(train_Y))
    
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
        
        history = self.model.fit_generator( 
                    self.generateXYBatches(ds, qa_pair['train'], batch_size, self.prep_step, self.proc_third_input, positive_rate=self.params['method_params']['positive_rate']), 
                    samples_per_epoch=batch_size, 
                    validation_data=self.generateXYBatches(ds, qa_pair['validate'], validation_size, self.prep_step, self.proc_third_input, positive_rate=self.params['method_params']['positive_rate']), 
                    nb_val_samples=validation_size, 
                    nb_epoch=epochs_number, 
                    callbacks=[checkpoint, earlyStopping],nb_worker=10)
        return history
    
    
    def test(self, ds, qa_pair):
        #reload best weights
        self.model.load_weights(self.best_params)
        #Construct Test dataset 
        test_X1 = []
        test_X2 = []
        test_X3 = []
        test_Y = []           
        for qa in ds.build_qa_pairs(qa_pair):
            q_vect = np.array(self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(qa.q,self.prep_step)))
            q_vect = np.pad(q_vect, ((0,self.timesteps-q_vect.shape[0]),(0,0)), mode='constant')
            a_vect = np.array(self.w2vutil.transform2Word2Vect(nlp_utils.data_preprocess(qa.a,self.prep_step)))
            a_vect = np.pad(a_vect, ((0,self.timesteps-a_vect.shape[0]),(0,0)), mode='constant')
            test_X1.append(q_vect)
            test_X2.append(a_vect)
            test_X3.append(self.proc_third_input(qa.q, qa.a))
            test_Y.append( qa.l )
        test_X1 = np.array(test_X1)
        test_X2 = np.array(test_X2)
        test_X3 = np.array(test_X3)
        test_Y = np.array(test_Y)
        #loss, acc = model.evaluate([test_X1, test_X2, test_X3], test_Y, 50)
        #loss, acc = model.evaluate(test_X, test_Y, batch_size, show_accuracy=True)
        #print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
        predictions = self.model.predict([test_X1, test_X2, test_X3])
        return predictions
