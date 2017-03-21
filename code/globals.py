import sys

dataset_folder = '/home/aerossom/datasets/WikiQACorpus'

normalize_numbers = False

input_files = {'train': dataset_folder+'/WikiQASent-train.txt',
               'test': dataset_folder+'/WikiQASent-test.txt',
               'validate': dataset_folder+'/WikiQASent-dev.txt',
               'all': dataset_folder+'/WikiQA.tsv'}

def printAll():
    print "dataset_folder: ", dataset_folder
    print "input_files: ", input_files