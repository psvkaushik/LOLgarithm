from make_embed import make_embeddings
from joke_scraper import scraped_jokes
import pandas as pd
import numpy as np


data = pd.DataFrame({'text' : scraped_jokes})
semantic_list = ['disconnection', 'repitition', 'sense_combination', 'num_alliteration', 'num_rhymes', 'max_alliteration', 'max_rhymes']

filtered_emotion_features, filtered_sse_features, filtered_semantic_features= make_embeddings(data, semantic_list = semantic_list)
disconnection, repitition, sense_combination, num_alliteration, num_rhymes, max_alliteration, max_rhymes = filtered_semantic_features[0], filtered_semantic_features[1], filtered_semantic_features[2], filtered_semantic_features[3], filtered_semantic_features[4], filtered_semantic_features[5], filtered_semantic_features[6]

print(np.array(filtered_semantic_features).shape)