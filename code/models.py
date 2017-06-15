import nlp_utils
import numpy as np
import threading

class PassageRetrievalModel(object):

    def __init__(self, name, runid):
        self.name = name
        self.runid = runid

    def train(self, ds, datasetv):
        raise NotImplementedError("Subclass must implement abstract method")

    def load_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def test(self, ds, dataset):
        raise NotImplementedError("Subclass must implement abstract method")

    def gen_trec_eval_file(self,predictions, rank_file, qa_pairs):
        idx_pred = 0
        with open(rank_file, 'wb') as text_file:
            for qa in qa_pairs:
                label = predictions[idx_pred][0]
                str_out = str(qa.qi) + ' 0 ' + str(qa.ai) + ' 0 ' + str(label) + ' 0\n'
                idx_pred += 1
                text_file.write(str_out)


class PassRtvModelFactory():
    @staticmethod
    def load_model2(targetclass, params):
        return globals()[targetclass](params)

    @staticmethod
    def load_model(module_name, class_name, params):
        """Constructor"""
        module = __import__(module_name)
        my_class = getattr(module, class_name)
        instance = my_class(params)
        return instance


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()




def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g
