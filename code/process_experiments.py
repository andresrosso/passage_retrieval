#!/usr/bin/env python
#!/home/aerossom/.conda/envs/py2/bin

import QAData
from os import listdir
from os.path import isfile, join
import PassageRetrieval as pr
from json2html import *
import matplotlib
#matplotlib.use('Agg')
import pylab as plt
import nlp_utils
import time
import os
import logging
from json_utils import JSONConnector

exp_in_path = '/home/aerossom/passage-retrieval/conf_experiments/'
exp_out_path = '/var/www/html/aerossom/exp_result/'
exp_done_path = '/home/aerossom/passage-retrieval/conf_experiments_done/'
w2v_path = '/home/aerossom/datasets/word2vect/GoogleNews-vectors-negative300.bin'
w2v_util = nlp_utils.Word2vectUtils(w2v_path)

logging.basicConfig(filename="messages.log", level=logging.INFO, format='%(filename)s: %(levelname)s: %(funcName)s():  %(lineno)d:\t %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def start_try():
    onlyfiles = [f for f in listdir(exp_in_path) if isfile(join(exp_in_path, f))]
    logger.info( 'Experimentation files' + str(onlyfiles) )
    m_recip_rank, m_map = 0,0
    for exp_file in onlyfiles:
        logger.info( 'process model file ' + str(exp_file) )
        run_id = str(int(time.time()))
        js = JSONConnector(exp_in_path+exp_file)
        with open(exp_in_path+exp_file) as f:
            jshtml = f.read()
            jshtml = json2html.convert(json = jshtml)
            report, m_map, m_recip_rank = pr.run_experiment(js, w2v_util, run_id, exp_in_path+exp_file, jshtml)
        #with open(exp_out_path+js.data['dataset']+'_map_'+str(m_map)+'_'+exp_file+'_'+run_id, 'wb') as report_file:
        with open(exp_out_path+js.data['model']+'_map_'+str(m_map)+'_'+js.data['dataset']+'_'+run_id, 'wb') as report_file:
            report_file.write(report)
            os.rename(exp_in_path+exp_file, exp_done_path+exp_file)
        logger.info( 'end process file ' + str(exp_file) )
        #time.sleep(10)
    return 0

def queryRepeatedly():
    num_try = 0
    print '###################################################################'
    while num_try < 20:
        print '######## PROCESS EXPERIMENTATION STARTS TRY (',num_try,') #############'
        try:
            start_try()
        except Exception as e:
            logger.exception( "Error evaluating model : " + str(e) )
            logger.error(e)
            print "Error: ", e
        num_try = num_try + 1
        time.sleep(10)
    print '############ PROCESS EXPERIMENTATION ENDS #################'
    print '###################################################################'

queryRepeatedly()
