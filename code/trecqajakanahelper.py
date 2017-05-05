import re
import os
import numpy as np
import cPickle
import subprocess
from QAData import QAPair

from collections import defaultdict

UNKNOWN_WORD_IDX = 0
DATA_PATH = '/home/aerossom/datasets/jacana-qa-naacl2013-data-results/'

'''
train-less-than-40.manual-edit.xml: TRAIN in paper
train2393.cleanup.xml.gz:           TRAIN-ALL in paper
dev-less-than-40.manual-edit.xml:   DEV in paper
test-less-than-40.manual-edit.xml:  TEST in paper
The dataset was first released and then organized by the following papers:
Mengqiu Wang, Noah A. Smith, and Teruko Mitamura. 2007. What is the Jeopardy model? a quasi-synchronous grammar for QA.
In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational
Michael Heilman and Noah A. Smith. 2010. Tree edit models for recognizing textual entailments, paraphrases, and answers to questions.

Then for the task of answer extraction, it was processed in the following steps:
1. exact numbers are recovered (in the original data numbers are replaced with <num>).
2. each positive sentence is matched against the TREC pattern file to identify the answer fragments
3. if multiple match, then manually inspect it.
'''

datasets = {'train': DATA_PATH+'train-less-than-40.manual-edit.xml',
            'train-all': DATA_PATH+'train2393.cleanup.xml',
            'validate': DATA_PATH+'dev-less-than-40.manual-edit.xml',
            'test': DATA_PATH+'test-less-than-40.manual-edit.xml'
             }

def load_data(fname):
  lines = open(fname).readlines()
  qids, idx_ques, questions, idx_ans, answers, labels = [], [], [], [], [], []
  num_skipped = 0
  prev = ''
  qid2num_answers = {}

  k_q, k_a = -1, 0
  startq_answers = False

  for i, line in enumerate(lines):
    line = line.strip()

    qid_match = re.match('<QApairs id=\'(.*)\'>', line)

    if qid_match:
      qid = qid_match.group(1)
      qid2num_answers[qid] = 0

    if prev and prev.startswith('<question>'):
      question = line.lower().split('\t')
      k_a = 0
      startq_answers = True

    label = re.match('^<(positive|negative)>', prev)
    if label:
      if startq_answers:
          k_q = k_q + 1
          startq_answers = False

      label = label.group(1)
      label = 1 if label == 'positive' else 0
      answer = line.lower().split('\t')
      if len(answer) > 60:
        num_skipped += 1
        continue
      labels.append(label)
      answers.append(answer)
      idx_ans.append(k_a)
      k_a = k_a + 1

      questions.append(question)
      idx_ques.append(k_q)

      qids.append(qid)
      qid2num_answers[qid] += 1
    prev = line
  # print sorted(qid2num_answers.items(), key=lambda x: float(x[0]))
  # print 'num_skipped', num_skipped
  return idx_ques, questions, idx_ans, answers, labels

def buildQAPairs(idx_ques, questions, idx_ans, answers, labels):
    #Construct Question Answer Pairs
    questions_answer_pairs = []
    for k, test_q_k in enumerate(idx_ques):
        #questions_answer_pairs += [(' '.join(questions[k]), ' '.join(answers[k]), labels[k])]
        #questions_answer_pairs += [QAPair(k, q, i, a_i, is_correct)]
        questions_answer_pairs += [QAPair(str(idx_ques[k]+1)+'.0', ' '.join(questions[k]), idx_ans[k], ' '.join(answers[k]), labels[k])]
    return questions_answer_pairs
