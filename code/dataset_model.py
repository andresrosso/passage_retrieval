#WikiQa imports
import data_stats
import wordrepresentation
from wikiqahelper import load_questions_from_file
import QaData

class WikiQA(QaData):
    
    def __init__(self):
        QaData.__init__()
        #WikiQA, self).__init__()
        self.patitions = ['train','validate','test']     
        self.questions['train'], vocabulary, idf = load_questions_from_file('train', -1)
        self.questions['validate'], vocabulary, idf = load_questions_from_file('validate', -1)
        self.questions['test'], vocabulary, idf = load_questions_from_file('test', -1)
        
    def get_stats(self):
        return data_stats.getStats()
    
    '''
    Return a tuple of (quesion_id, question, answer_is, answer, label)
    '''
    def build_qa_pairs(dataset):
        #Construct Question Answer Pairs
        questions_answer_pairs = []
        for k, test_q_k in enumerate(dataset):
            q = test_q_k.question
            for i, a_i in enumerate(test_q_k.answers):
                is_correct = 1 if i in test_q_k.correct_answer else 0
                questions_answer_pairs += [(k, q, i, a_i, is_correct)]
        return questions_answer_pairs
    
    