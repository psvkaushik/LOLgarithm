""" Returns the vectors which contain syntactical information obtained using NRCEmotion
Lexicon
The size of each vector is 10 - representing each emotion present in the lexicon
"""

import re
import pandas as pd
from nrclex import NRCLex
import tqdm
from sklearn.model_selection import train_test_split

emotion_map = {
    'fear' : 0,
    'anger': 1,
    'anticip':2,
    'anticipation':2,
    'trust':3,
    'surprise':4,
    'positive':5,
    'negative':6,
    'sadness':7,
    'disgust':8,
    'joy':9
}


# Read the dataset
dataset = pd.read_csv("D:/Work/MS COURSES/COURSES FALL 23/CSC 791 Natural Language Processing/LOLgorithm/LOLgarithm/dataset/dataset.csv")
# Separate the X(jokes) and the Y(is_humor or not)

jokes = list(dataset['text'])
labels = list(dataset['humor'])

PATTERN = r'[^A-Za-z0-9\s]'

op = []
for joke in tqdm.tqdm(jokes):
    processed_joke = re.sub(PATTERN, '', joke)
    words = joke.split(' ')
    temp = [0]*10
    for word in words:
        lexs = NRCLex(word).top_emotions
        for emotion, val in lexs:
            temp[emotion_map[emotion]]+=val
    temp = [i/len(words) for i in temp]
    op.append(temp)

print(len(op))
nrclex_columns = ['fear','anger','anticipation','trust','surprise','positive','negative','sadness','disgust','joy']

nrclex_features_df = pd.DataFrame(op)
nrclex_features_df.columns = nrclex_columns
nrclex_features_df.to_csv('D:/Work/MS COURSES/COURSES FALL 23/CSC 791 Natural Language Processing/LOLgorithm/LOLgarithm/dataset/nrclex-features.csv')

# train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(op, labels, test_size=0.3)
