import pandas as pd
import numpy as np
from nltk import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import wordnet as wn, cmudict

import gensim
from gensim.models import Word2Vec

import re
import math
from tqdm import tqdm

import warnings
warnings.filterwarnings(action = 'ignore')

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('cmudict')

# Function to clean data - remove special characters if any
def clean_data(jokes):
    """
    """
    PATTERN = r'[^A-Za-z0-9\s]'

    words_list = []
    for joke in jokes:
        joke = joke.lower()
        processed_joke = re.sub(PATTERN, '', joke)
        words = processed_joke.split(' ')
        words_list.append(words)

    return words_list


"""""""""""INCONGRUITY"""""""""""

# Function of Incongruity: Calculate disconnection (max meaning distance) and repetition (min meaning distance)
def incongruity(word2vec_model, words_list):
  """
  """
  max_threshold = 1  # Set maximum threshold as 1 because distance of word from itself will be 1
  disconnection_list = []
  repetition_list = []
  for sentence in tqdm(words_list, desc="Processing Sentences"):
      sentence_word_distances = []
      for i in range(len(sentence)):
          for j in range(i + 1, len(sentence)):
              distance = word2vec_model.wv.similarity(sentence[i], sentence[j])
              if distance < max_threshold:
                  sentence_word_distances.append(distance)
      # Check if the sentence_word_distances list is not empty before calculating disconnection and repetition
      if sentence_word_distances:
          disconnection = max(sentence_word_distances)
          repetition = min(sentence_word_distances)
      else:
          disconnection = None
          repetition = None
      disconnection_list.append(disconnection)
      repetition_list.append(repetition)
  return disconnection_list, repetition_list


"""""""""""AMBIGUITY"""""""""""

# Function to get Sentences along with POS Tags
def get_pos_tagged_sentences(words_list):
    """
    """
    tagged_sentences = []
    pos_tagged_sentences = []

    print(f"Getting POS Tags for each sentence")
    for sentence in tqdm(words_list):
        tagged_sentence = pos_tag(sentence)
        tagged_sentences.append(tagged_sentence)

    print(f"Getting POS Tags Lists each sentence")
    for tagged_words in tqdm(tagged_sentences):
        pos_words = {'NOUN': [], 'VERB': [], 'ADJ': [], 'ADV': [], 'DET': [], 'NUM': []}
        for word, pos in tagged_words:
            if pos.startswith('N'):
                pos_words['NOUN'].append(word)
            elif pos.startswith('V'):
                pos_words['VERB'].append(word)
            elif pos.startswith('J'):
                pos_words['ADJ'].append(word)
            elif pos.startswith('R'):
                pos_words['ADV'].append(word)
        pos_tagged_sentences.append(pos_words)

    return pos_tagged_sentences

# Get sense combiination score
def sense_comination(pos_tagged_sentences):
    sense_combination_list = []
    for sentence in tqdm(pos_tagged_sentences):
        sense_combination = 0
        for pos, words in sentence.items():
            for word in words:
                synsets = wn.synsets(word, pos=pos[0].lower())
                if synsets:
                    num_senses = len(synsets)
                    sense_combination += math.log(num_senses) ##################### CHECK FORMULA ###################
        sense_combination = math.exp(sense_combination)
        sense_combination_list.append(sense_combination)

    return sense_combination_list


# Function to get Path Similarity features - farmost and closest path
def path_similarity(pos_tagged_sentences):
    sense_farmost_list = []
    sense_closest_list = []

    for sentence in tqdm(pos_tagged_sentences):
        path_similarities = []

        for words in sentence.values():
            for word in words:
                synsets = wn.synsets(word)
                if synsets:
                    # for each sense of same word, find similarity
                    for synset in synsets:
                    # Compare the similarity of our word sense with other word senses (of same word)
                        similarities = [synset.path_similarity(other) for other in synsets if other != synset and other.path_similarity(synset)]
                        if similarities:
                            path_similarities.extend(similarities)

        sense_farmost = max(path_similarities) if path_similarities else None
        sense_closest = min(path_similarities) if path_similarities else None
        sense_farmost_list.append(sense_farmost)
        sense_closest_list.append(sense_closest)

    return sense_farmost_list, sense_closest_list


"""""""""""PHONETIC STYLE"""""""""""
# Load CMU Pronouncing Dictionary
d = cmudict.dict()


# Function to get Phonetic representations of word
def get_phonemes(word):
    """
    Get phonetic representation of a word from CMU Pronouncing Dictionary
    """
    return d[word][0] if word in d else None


# Function to get alliteration and rhyme
def detect_alliteration_rhyme(words):
    alliteration_chains = 0
    max_alliteration_chain_length = 0

    rhyme_chains = 0
    max_rhyme_chain_length = 0

    prev_phoneme = None
    alliteration_chain_length = 1
    rhyme_chain_length = 1

    for word in words:
        phonemes = get_phonemes(word)

        if phonemes:
            first_phoneme = phonemes[0]
            last_phoneme = phonemes[-1]

            if prev_phoneme and first_phoneme == prev_phoneme:
                alliteration_chain_length += 1
            else:
                max_alliteration_chain_length = max(max_alliteration_chain_length, alliteration_chain_length)
                if alliteration_chain_length > 1:
                    alliteration_chains += 1
                alliteration_chain_length = 1

            if prev_phoneme and last_phoneme == prev_phoneme:
                rhyme_chain_length += 1
            else:
                max_rhyme_chain_length = max(max_rhyme_chain_length, rhyme_chain_length)
                if rhyme_chain_length > 1:
                    rhyme_chains += 1
                rhyme_chain_length = 1

            prev_phoneme = last_phoneme

    return alliteration_chains, max_alliteration_chain_length, rhyme_chains, max_rhyme_chain_length


# Get Phonetic Style
def phonetic_style(words_list):
  alliteration_list = []
  max_alliteration_list = []
  rhyme_list = []
  max_rhyme_list = []

  for sentence in tqdm(words_list):
    alliteration_chains, max_alliteration_chain_length, rhyme_chains, max_rhyme_chain_length = detect_alliteration_rhyme(sentence)

    alliteration_list.append(alliteration_chains)
    max_alliteration_list.append(max_alliteration_chain_length)
    rhyme_list.append(rhyme_chains)
    max_rhyme_list.append(max_rhyme_chain_length)

  return alliteration_list, max_alliteration_list, rhyme_list, max_rhyme_list
