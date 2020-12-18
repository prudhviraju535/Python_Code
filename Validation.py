#!/usr/bin/env python
# coding: utf-8

# In[1]:


n = 4


# In[2]:


import os
os.getcwd()


# In[3]:


# -*- coding: utf-8 -*-

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re

import tensorflow as tf

print(tf.config.experimental.list_physical_devices('GPU'))
# After training the model, re-run the environment but run this code in first, then predict.
tfe = tf.contrib.eager
tfe.enable_eager_execution()
Modes = tf.estimator.ModeKeys

# Config
from decode_t2t_funcs import encode, decode
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import t2t_model
import datetime

#from ADLS_access import access_file_from_directory
#from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

#from utils import process_data
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics
import numpy as np
import pandas as pd
import os
from tensor2tensor.data_generators import text_problems

## Required Folder Creation

HOME_PATH = "/data/home/users/pnadim64/notebooks/Raju/Translation/T2t/pml-sig-mlapp/model_output/"

data_dir = os.path.expanduser(HOME_PATH + "data")  # This folder contain the data
tmp_dir = os.path.expanduser(HOME_PATH + "tmp")  # Ths folder contains temp data if any
train_dir = os.path.expanduser(HOME_PATH + "train")  # This folder contain the model
export_dir = os.path.expanduser(HOME_PATH + "export")  # This folder contain the exported model for production
translations_dir = os.path.expanduser(HOME_PATH + "translation")  # This folder contain  all translated sequence
event_dir = os.path.expanduser(HOME_PATH + "event")  # Test the BLEU score
usr_dir = os.path.expanduser(HOME_PATH + "user")  # This folder contains our data that we want to add
checkpoint_dir = os.path.expanduser(HOME_PATH + "checkpoints")
# %%

from tensor2tensor.utils.trainer_lib import create_hparams

## Model name and Parameters selection
PROBLEM = "sig_translator"  # Custom ESIG Translation Problem
MODEL = "transformer"  # Our model
HPARAMS = "transformer_h16"  # Hyperparameters for the model by default


# If you have a one gpu, use transformer_big_single_gpu

from tensor2tensor.utils import usr_dir
from tensor2tensor import problems


# In[ ]:


#dsvm config
#usr_dir.import_usr_dir('~/varshini_esig/T2T_test_project/export_files')

usr_dir.import_usr_dir('/data/home/users/pnadim64/notebooks/Raju/Translation/T2t/pml-sig-mlapp/')


t2t_problem = problems.problem(PROBLEM)

# Copy the vocab file locally so we can encode inputs and decode model outputs
vocab_name = "vocab.sig_translator.32768.subwords"
vocab_file = os.path.join(data_dir, vocab_name)

print(vocab_file)

# Get the encoders from the problem
encoders = t2t_problem.feature_encoders(data_dir)


def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))

# Predict
hparams = trainer_lib.create_hparams(HPARAMS, data_dir=data_dir, problem_name=PROBLEM)
hparams.batch_size = 4096
hparams.learning_rate_warmup_steps = 4000
hparams.learning_rate = .2
translate_model = registry.model(MODEL)(hparams, Modes.PREDICT)
#ckpt_path = tf.train.latest_checkpoint(os.path.join(train_dir))

#averaged ckpt path
ckpt_path = HOME_PATH+'/checkpoints/averaged.ckpt-0'

print('checkpoint path:', ckpt_path)

def translate(inputs):
    encoded_inputs = encode(inputs)
    with tfe.restore_variables_on_create(ckpt_path):
        t1 = datetime.datetime.now()
        model_out = translate_model.infer(encoded_inputs)
        t2 = datetime.datetime.now()
        time_taken_for_response = int((t2 - t1).total_seconds() * 1000)
        model_output = model_out["outputs"]
        val = model_out["scores"]

    return {'model_output': decode(model_output), 'log_likelihood': val, 'time': time_taken_for_response}



def calculate_bleu(test_sig):
    bleu_score_1_gram = []
    bleu_score_2_gram = []

    for i in tqdm(range(len(test_sig))):
        bleu_score_1_gram.append(
            sentence_bleu([test_sig.IC_Pharmacist_SIG.iloc[i].split()], test_sig.ml_translation_prediction.iloc[i].split(),
                          weights=(1, 0, 0, 0)))
        bleu_score_2_gram.append(
            sentence_bleu([test_sig.IC_Pharmacist_SIG.iloc[i].split()], test_sig.ml_translation_prediction.iloc[i].split(),
                          weights=(0.5, 0.5, 0, 0)))

    test_sig['translation_BLEU_score_1_gram'] = bleu_score_1_gram
    test_sig['translation_BLEU_score_2_gram'] = bleu_score_2_gram

    test_sig['translation_BLEU_score_1_gram'] = round(test_sig['translation_BLEU_score_1_gram'], 2)
    test_sig['translation_BLEU_score_2_gram'] = round(test_sig['translation_BLEU_score_2_gram'], 2)

    return test_sig


def normalized_log_loss(test_sig):
    test_sig['ML_translation_Prediction_Len'] = test_sig.ml_translation_prediction.map(lambda x: len(x.split()))
    test_sig['translation_log_likelihood'] = np.around(test_sig['translation_log_likelihood'].astype(np.double), 2)
    test_sig['translation_ML_Logloss_Normalize'] = test_sig.translation_log_likelihood / test_sig.ML_translation_Prediction_Len
    test_sig['translation_ML_Logloss_Normalize'] = np.around(test_sig['translation_ML_Logloss_Normalize'].astype(np.double), 2)

    return test_sig



#validation

data_path = '/data/home/users/pnadim64/notebooks/Raju/Translation/T2t/Data/CV_Validation/'

#sig_data = access_file_from_directory("sig-erx", data_path, "test_Data_OTHERS_OTHERS_groups_14JULY20VER1.csv")

sig_data = pd.read_csv(data_path +'OR_OTH_CV_VALIDATION_{}.csv'.format(n))
print(sig_data.shape)


#err = ['SIG001E','SIG002E','SIG003E','SIG005E','SIG006E','SIG012E','SIG013E','SIG015E','SIG016E','SIG017E']

#sig_data = sig_data[sig_data.Error_CD.isin(err)]


#tqdm.pandas()
#sig_data['score'] = sig_data.Standardized_SIG.progress_apply(lambda x: translate(x))

ml_prediction = []
log_likelihood = []
bleu_score = []

for rx in tqdm(sig_data.Standardized_SIG.tolist()):
    out = translate(rx)
    ml_prediction.append(re.sub(r'\b(\w+)( \1\b)+', r'\1', out['model_output']))
    log_likelihood.append(out['log_likelihood'].numpy()[0])

sig_data['ml_translation_prediction'] = ml_prediction
sig_data['translation_log_likelihood'] = log_likelihood


# In[ ]:


sig_data = calculate_bleu(sig_data)

sig_data = normalized_log_loss(sig_data)

sig_data.to_csv('OR_OTH_CV_VALIDATION_{}_P.csv'.format(n), index=False)


# In[ ]:




