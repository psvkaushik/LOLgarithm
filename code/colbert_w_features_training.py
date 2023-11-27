import torch
from transformers import TFBertModel, BertTokenizer
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
from tensorflow.keras.layers import Input, Dense, Concatenate, Add, Dropout
from tensorflow.keras.models import Model

model = keras.models.load_model("/home/tkhuran3/LOLgarithm/colbert-trained")

with open('colbert_inputs.pkl', 'rb') as f:
    colbert_inputs = pickle.load(f)
with open('colbert_output.pkl', 'rb') as f:
    colbert_outputs = pickle.load(f)

features_df= pd.read_csv("/home/tkhuran3/LOLgarithm/dataset/combined-features.csv")
features_df = features_df.iloc[:, 1:]
features_df = features_df.values.tolist()
additional_features_array = np.array(features_df)

# Modify model architecture

layers = [l for l in model.layers]

for i in range(0,43):
    layers[i].trainable = False
    concatenated_features_1 = Concatenate()([
    model.get_layer('dense_16').output,
    model.get_layer('dense_18').output,
    model.get_layer('dense_20').output,
    model.get_layer('dense_22').output,
    model.get_layer('dense_24').output,
    model.get_layer('dense_26').output])

additional_features_input = Input(shape=(33), name="additional_features")
additional_dense = Dense(104, activation='relu')(additional_features_input)
concatenated_features_2 = Concatenate()([concatenated_features_1, additional_dense])

x = Dense(128, activation='relu')(concatenated_features_2)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

result_model = Model(inputs=[model.input, additional_features_input], outputs=output)
print(result_model.summary())

result_model.compile(optimizer=Adam(3e-5),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),  
              metrics=['accuracy'])

history = result_model.fit([colbert_inputs, additional_features_array] ,colbert_outputs, batch_size=64, epochs=10)
result_model.save("colbert_trained_w_features")
