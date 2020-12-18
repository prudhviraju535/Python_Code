# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:53:18 2020

@author: Prudhvi

train.py

Translate Standardise SIG to Pharmacist SIG
Data use: entire one year data on GROUP1 and 20% success data
"""

import re
import json
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils.trainer_lib import create_run_config, create_experiment
from tensor2tensor.utils.trainer_lib import create_hparams
from tensor2tensor import models
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics_hook, metrics, t2t_model
from tensorflow.python.client import device_lib
from tensor2tensor import problems
from tensor2tensor.bin import t2t_eval, t2t_bleu
from pathlib import Path
import shutil
import os
import numpy as np
import pandas as pd
import tensorflow as tf

# call model registration package
from sig_translator import SigTranslator


print(tf.__version__)
tf.logging.set_verbosity(tf.logging.INFO)

# Enable TF Eager execution
tfe = tf.contrib.eager
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

# %%
print('************')
print(tf.config.experimental.list_physical_devices('GPU'))
print('************')

# %%
## Required Folder Creation
print('**********Creating Data path ************')
HOME_PATH = os.getcwd()
HOME_PATH = os.path.join(HOME_PATH, "model_output")
data_dir = os.path.join(HOME_PATH, "data")  # This folder contain the data
tmp_dir = os.path.join(HOME_PATH, "tmp")  # Ths folder contains temp data if any
train_dir = os.path.join(HOME_PATH, "train")  # This folder contain the model
export_dir = os.path.join(HOME_PATH, "export")  # This folder contain the exported model for production
translations_dir = os.path.join(HOME_PATH, "translation")  # This folder contain  all translated sequence
event_dir = os.path.join(HOME_PATH, "event")  # Test the BLEU score
usr_dir = os.path.join(HOME_PATH, "user")  # This folder contains our data that we want to add
checkpoint_dir = os.path.join(HOME_PATH, "checkpoints")

## Creating folders
print('**********Creating folders ************')

list_of_dirs_to_create = [data_dir, tmp_dir, export_dir, translations_dir, train_dir, event_dir, usr_dir,
                          checkpoint_dir]
folder_names = ['data', 'tmp', 'export', 'translation', 'train', 'event', 'user', 'checkpoints']

for directory_path in list_of_dirs_to_create:
    tf.io.gfile.makedirs(directory_path)


PROBLEM = "sig_translator"  # Custom ESIG Translation Problem
MODEL = "transformer"  # Our model
HPARAMS = "transformer_big"  # Hyperparameters for the model by default

# If you have a one gpu, use transformer_big_single_gpu


print('Generating data...........')
problem_definition = SigTranslator()
t2t_problem = problems.problem(PROBLEM)
t2t_problem.generate_data(data_dir, tmp_dir)

print("Data Generated successfully!!")

# Init Hparams object from T2T Problem
hparams = create_hparams(HPARAMS)

# Make Changes to Hparams
hparams.batch_size = 4048
hparams.learning_rate_warmup_steps = 40000
hparams.learning_rate = .2
save_checkpoints_steps = 10000

#keep_checkpoint_max = 100

# Can see all Hparams with code below
print(json.loads(hparams.to_json()))

# Init Run Config for Model Training
RUN_CONFIG = create_run_config(
    model_dir=train_dir,
    model_name=MODEL,
    num_gpus=2,
    #keep_checkpoint_max=keep_checkpoint_max,
    save_checkpoints_steps=save_checkpoints_steps  # Location of where model file is store
    # More Params here in this fucntion for controling how noften to tave checkpoints and more.
)

# # Create Tensorflow Experiment Object
tensorflow_exp_fn = create_experiment(
    run_config=RUN_CONFIG,
    hparams=hparams,
    model_name=MODEL,
    problem_name=PROBLEM,
    data_dir=data_dir,
    schedule="train_and_evaluate",
    #eval_early_stopping_steps=5000,
    min_eval_frequency=1000,
    train_steps=90000,  # Total number of train steps for all Epochs
    eval_steps=100  # Number of steps to perform for each evaluation
)

# Kick off Training
print('Training started.....')

#file = open("Model_Training_Progress.txt", "w")
#file.close()

#with open("Model_Training_Progress.txt", "a") as f:
#    f.write(print(tensorflow_exp_fn.train_and_evaluate()))

tensorflow_exp_fn.train_and_evaluate()

print('Training completed successfully!')

