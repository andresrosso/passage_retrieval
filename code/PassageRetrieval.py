#!/usr/bin/env python
from subprocess import call, Popen, PIPE
from time import gmtime, strftime
from jinja2 import Template
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import pylab as plt
import sys
import pickle
import logging
from json_utils import JSONConnector
from utils import ObjectFactory
import QAData
from QAData import *
import nlp_utils
import models as models
import time

template = """
<html>
<head>
<title>Model <b>{{ model }}<b></title>
</head>
<body>
<table>
<tr>
<td colspan="2"><h2>Model {{model}}</h2></td>
</tr>
<tr>
<td>Model description</td><td>{{modeldes}}</td>
</tr>
<tr>
<td>Configuration File</td><td>{{cfile}}</td>
</tr>
<tr>
<td>Running Date</td><td>{{date}}</td>
</tr>
<tr>
<td>Running ID</td><td>{{runid}}</td>
</tr>
<tr>
<td>MAP</td><td>{{mmap}}</td>
</tr>
<tr>
<td>MRR</td><td>{{mmrr}}</td>
</tr>
<tr>
<td>Accuracy Curve</td><td><img src="data:image/png;base64,{{learn_acc_plot}}"/></td>
</tr>
<tr>
<td>Loss Curve</td><td><img src="data:image/png;base64,{{learn_loss_plot}}"/></td>
</tr>

<tr>
<td>Parameters</td><td>{{model_vars}}</td>
</tr>

</table>
</body>
</html>
"""

def run_experiment(jsparams, w2v_util, run_id, cfile, html_model_params):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    start_time = time.time()
    params = jsparams.data
    html_params = {}
    ds = DataSetFactory.loadDataSet(params['dataset'])
    html_params['stats'] = ds.get_stats()
    html_params['dsname'] = ds.name
    workfolder = params["working_folder"]
    #Get train qa pairs
    logger.info("Loading the dataset "+str(ds.name)+" !")
    qa_pair = {}
    for partition in params['dataset_partitions']:
        qa_pair[str(partition)] = ds.build_qa_pairs(ds.questions[str(partition)])
    #Train and test the model
    model_params = {"w2v": w2v_util, "runid":run_id,  "params":params }
    logger.info("Loading the model, " + str(params['model']))
    p_model = models.PassRtvModelFactory.load_model(params['model'],params['model'],model_params)
    logger.info("Trainning the model")
    history = p_model.train(ds, qa_pair)
    #Save history object in pickel
    save_history(params, history, run_id)
    logger.info("Testing the model")
    predictions = p_model.test(ds, qa_pair['test'])
    trec_eval_path = params['trec_eval_path']
    rank_file = params['out_folder']+params['expriment_id'].replace('$runid',str(run_id))+".rank"
    ground_truth_file = params['ground_truth_file']
    p_model.gen_trec_eval_file(predictions,rank_file,ds.build_qa_pairs(ds.questions['test']))
    # Extract the score from treceval
    p = Popen([trec_eval_path, '-c', ground_truth_file, rank_file], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    rc = p.returncode
    out = output.split("\n")
    m_map = [s for s in out if "map" in s and "gm_map" not in s]
    m_map = m_map[0].split("\t")[2]
    m_recip_rank = [s for s in out if "recip_rank" in s]
    m_recip_rank = m_recip_rank[0].split("\t")[2]
    logger.info("Model performance:, map: " + str(m_map) + ", mrr: "+str(m_recip_rank))
    strftime("%Y-%m-%d %H:%M:%S", gmtime())
    html_params['model'] = params['model']
    html_params['modeldes'] = params['modeldes']
    html_params['expriment_id'] = params['expriment_id']
    html_params['date'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_params['map'] = m_map
    html_params['mrr'] = m_recip_rank
    html_params['run_id'] = run_id
    html_params['model_vars'] = html_model_params
    html_params['cfile'] = cfile

    html_report = gen_html_report(html_params, history, workfolder)

    logger.info("End model evaluation, total time: " + str(time.time() - start_time) )
    return html_report, m_map, m_recip_rank

def gen_html_report(params, history, workfolder):
    t = Template(template)
    return t.render(model=params['model'],runid=params['run_id'],date=params['date'],mmap=params['map'],mmrr=params['mrr'], learn_acc_plot=gen_acc_plot(history, workfolder),learn_loss_plot=gen_error_plot(history, workfolder),model_vars=params['model_vars'], cfile=params['cfile'], modeldes=params['modeldes'])

def save_history(params, history, run_id):
    logger = logging.getLogger()
    out_file = params['working_folder']+params['expriment_id'].replace('$runid',run_id)+"_history.pkl"
    with open(out_file, 'wb') as output:
        pickle.dump(history.history, output, pickle.HIGHEST_PROTOCOL)
    logger.info("History saved at: " + out_file )

def gen_acc_plot(history, workfolder):
    # Print learning history
    # summarize history for accuracy
    plt.figure(figsize=(5,5))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(workfolder+'learn_acc.png')
    plt.close()
    learn_acc_string = ''
    with open(workfolder+'learn_acc.png', "rb") as f:
        data = f.read()
        learn_acc_string = data.encode("base64")
    return learn_acc_string

def gen_error_plot(history, workfolder):
    # summarize history for loss
    plt.figure(figsize=(5,5))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
    plt.savefig(workfolder+'learn_loss.png')
    plt.close()
    learn_loss_string = ''
    with open(workfolder+'learn_loss.png', "rb") as f:
        data = f.read()
        learn_loss_string = data.encode("base64")
    return learn_loss_string
