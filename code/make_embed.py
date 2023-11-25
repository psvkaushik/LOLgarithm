from tqdm import tqdm
import pandas as pd
from gensim.models import Word2Vec
from SSE import get_syntactic_features
from data_NRC import get_emotion_features
from semantic_features import clean_data, get_pos_tagged_sentences,sense_combination, path_similarity, incongruity, get_alliteration_rhyme_features
import numpy as np

data = pd.read_csv(r"C:\Users\psvka\Downloads\data_with_distfeatures.csv", index_col=0)
print(data.head())

def make_embeddings(data: pd.DataFrame, emotions_list : list = ['all'], sse_list : list = ['all'], semantic_list : list = ['all'], SKIPGRAM: int = 1):

    emotion_map = {
    'fear' : 0,
    'anger': 1,
    'anticipation':2,
    'trust':3,
    'surprise':4,
    'positive':5,
    'negative':6,
    'sadness':7,
    'disgust':8,
    'joy':9
    }

    sse_feature_map = { 'np_count' : 0,
                        'vp_count': 1,
                        'pp_count': 2,
                        'sbar_count': 3,
                'np_ratio':4, 'vp_ratio':5, 'pp_ratio':6, 
                'phrase_length_ratios_VP':7,
                'phrase_length_ratios_NP':8, 'phrase_length_ratios_PP':9,
                'avg_NP_len':10, 'avg_VP_len':11, 'avg_PP_len':12, 'rpnv': 13}
    

    jokes = data['text'][:100]
    emotion_features = np.array(get_emotion_features(jokes))
    sse_features = np.array(get_syntactic_features(jokes))
    ### semantic features

    # Incongruity
    words_list = clean_data(jokes)
    wv_model = Word2Vec(words_list, min_count = 1, vector_size = 100, window = 5, sg = SKIPGRAM)
    disconnection_list, repetition_list = np.array(incongruity(wv_model, words_list))

    # Ambiguity
    pos_sentences = get_pos_tagged_sentences(words_list)
    sense_combination_list = sense_combination(pos_sentences)

    # Phonetic Style
    phonetic_style_features, eps, sps = get_alliteration_rhyme_features(jokes)
    phonetic_style_features = np.array(phonetic_style_features)
    
    semantic_map = {
        'disconnection': disconnection_list,
        'repitition' : repetition_list,
        'sense_combination': sense_combination_list,
        'num_alliteration' : phonetic_style_features[:, 0],
        'num_rhymes' : phonetic_style_features[:, 1],
        'max_alliteration' : phonetic_style_features[:, 2],
        'max_rhymes' : phonetic_style_features[:, 3],
        'closest_path': list(data['closest']),
        'farthest_path': list(data['farthest'])
    }
    filtered_emotion_features = []
    if emotions_list == ['all']:
        filtered_emotion_features = emotion_features
    else:
        indices = []
        for i in emotions_list:
            indices.append(emotion_map[i])
        filtered_emotion_features = emotion_features[:, indices]
    
    filtered_sse_features = []
    if sse_list == ['all']:
        filtered_sse_features = sse_features
    else:
        indices = []
        for i in sse_list:
            indices.append(sse_feature_map[i])
        filtered_sse_features = sse_features[:, indices]
    filtered_semantic_features = []
    if semantic_list == ['all']:
        for i in semantic_map.keys():
            filtered_semantic_features.append(semantic_map[i])
    else:
        for i in semantic_list:
            filtered_semantic_features.append(semantic_map[i])
    return filtered_emotion_features, filtered_sse_features, filtered_semantic_features


# emotions_list = {
#     'fear' : 0,
#     'anger': 1,
#     'anticipation':2,
#     'trust':3,
#     'surprise':4,
#     'positive':5,
#     'negative':6,
#     'sadness':7,
#     'disgust':8,
#     'joy':9
# }
emotions_list=['surprise', 'trust', 'anticipation']

# sse_feature_names = ['np_count', 'vp_count', 'pp_count', 'sbar_count',
#                 'np_ratio', 'vp_ratio', 'pp_ratio', 'phrase_length_ratios_VP',
#                 'phrase_length_ratios_NP', 'phrase_length_ratios_PP',
#                 'avg_NP_len', 'avg_VP_len', 'avg_PP_len', 'rpnv']
sse_list=['vp_count', 'avg_VP_len', 'rpnv', 'phrase_length_ratios_NP', 'phrase_length_ratios_VP']

# TODO : Vikram fill this out
# NOTE : for filtered_semantic_features, the output is of dimension len(semantic_list) x 2,00,000 x whatever the dimension that feature has, so 
# make sure to unpack(contd.....)
# NOTE: (contd....) for below example filtered_semantic_features is unpacked into those five variables.
semantic_list = ['disconnection', 'sense_combination', 'num_alliteration', 'max_alliteration', 'closest_path']
filtered_emotion_features, filtered_sse_features, filtered_semantic_features= make_embeddings(data, emotions_list, sse_list, semantic_list)
disconnection, sense_combination, num_alliteration, max_alliteration, closest_path = filtered_semantic_features[0], filtered_semantic_features[1], filtered_semantic_features[2], filtered_semantic_features[3], filtered_semantic_features[4]
print(np.shape(filtered_emotion_features))
print(np.shape(filtered_sse_features))
print(np.shape(disconnection))
print(np.shape(sense_combination))
print(np.shape(num_alliteration))
print(np.shape(max_alliteration))
print(len(closest_path))

