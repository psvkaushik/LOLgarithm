import torch
from transformers import BertTokenizer
import os
import time
import pandas as pd
from datasets import load_metric
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np
import csv
import tensorflow as tf
import keras
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import sklearn
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

model = keras.models.load_model("/home/tkhuran3/LOLgarithm/colbert-trained")

with open('colbert_inputs.pkl', 'rb') as f:
    colbert_inputs = pickle.load(f)
with open('colbert_output.pkl', 'rb') as f:
    colbert_outputs = pickle.load(f)

model.compile(optimizer=Adam(3e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  
              metrics=['accuracy'])

history = model.fit(colbert_inputs, colbert_outputs, batch_size=64, epochs=2, validation_split=0.2)
model.save("colbert_newtrained")

# test_scores = model.evaluate(colbert_inputs, colbert_outputs, verbose=2)

with open('sample_input.pkl', 'rb') as f:
    x_test = pickle.load(f)
with open('sample_output.pkl', 'rb') as f:
    y_true = pickle.load(f)

y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(int)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print('Acc', accuracy, 'Prec', precision, 'Rec', recall, 'F1',f1)