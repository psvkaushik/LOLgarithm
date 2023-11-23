""" Performs Statistics of Structural Elements and returns an embedding which represents
    the SSE of a sentence
"""

import pandas as pd
import numpy as np
import nltk
from nltk import pos_tag, word_tokenize
from nltk.chunk import RegexpParser
from collections import defaultdict
import tqdm
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')



dataset = pd.read_csv(r"C:\Users\psvka\OneDrive\Desktop\fall23\csc791\LOLgarithm\dataset\dataset.csv")
sentences = list(dataset['text'])
labels = list(dataset['humor'])

# Function to extract phrases
def extract_phrases(sentence):
    """
    This function extracts all the phrases, their counts, and also calculates one
    ratio of NP/PP to VP in a sentence.
    """
    # First we tokenize the sentence and get  the POS tags
    words = word_tokenize(sentence)
    pos_tags = pos_tag(words)

    # This is the default grammar we've chosen to work with.
    grammar = r"""
        NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}  # Noun Phrase
        VP: {<VB.*><NP|PP|RB>*}          # Verb Phrase
        PP: {<IN><NP>}                    # Prepositional Phrase
        S: {<NP><VP>}                     # Simple Sentence
        SBAR: {<IN|DT|RB><S>}             # Subordinate Clause
    """

    # Create a chunk parser with the chosen grammar
    chunk_parser = RegexpParser(grammar)

    # Apply the parser to the part-of-speech tagged words
    tree = chunk_parser.parse(pos_tags)

    # Extract the phrases in a sentence adn store them
    phrases = defaultdict(list)
    phrases_count = defaultdict(int)
    for subtree in tree.subtrees():
        if subtree.label() in ['NP', 'VP', 'PP', 'S', 'SBAR']:
            phrases[subtree.label()].append(' '.join(word for word, _ in subtree.leaves()))
            phrases_count[subtree.label()]+=1

    # The next lines of code calculate thr RPNV measure
    vp_len = 0
    np_or_pp_len = 0
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'VP'):
        vp_len += len(subtree.leaves())
        for child in subtree:
            try: 
                if child.label() in ['NP', 'PP']:
                    np_or_pp_len += len(child.leaves())
            except:
                continue
    if vp_len:
        rpnv = np_or_pp_len/vp_len
    else:
        rpnv = 0
    return phrases, phrases_count, rpnv

def get_syntactic_features(sentences):
    """
    This Functions Calculate 5 of the 7 Statistical features and return their embeddings
    """
    sse_features = []

    for sentence in tqdm.tqdm(sentences):
        #Get the phrases and the counts, and also First Complexity Metric - RPNV
        phrases, phrases_count, rpnv = extract_phrases(sentence)

        #Second complexity metric - Phrase Counts
        np_count = phrases_count['NP']
        vp_count = phrases_count['VP']
        pp_count = phrases_count['PP']
        sbar_count = phrases_count['SBAR']

        #Third complexity metric - Phrase Ratio
        total = phrases_count['NP'] + phrases_count['VP'] + phrases_count['PP'] + phrases_count['SBAR']
        if total:
            np_ratio = phrases_count['NP']/total
            vp_ratio = phrases_count['VP']/total
            pp_ratio = phrases_count['PP']/total
        else:
            np_ratio, pp_ratio, vp_ratio = 0, 0, 0

        #Fourth Complexity Metric -  Phrase Length Ratio
        words_VP = 0
        words_NP = 0
        words_PP = 0
        max_len_VP = 0
        max_len_NP = 0
        max_len_PP = 0
        for phrase in phrases['VP']:
            words_VP += len(phrase.split(' '))
            max_len_VP = max(max_len_VP, len(phrase.split(' ')))
        for phrase in phrases['NP']:
            words_NP += len(phrase.split(' '))
            max_len_NP = max(max_len_NP, len(phrase.split(' ')))
        for phrase in phrases['PP']:
            words_PP += len(phrase.split(' '))
            max_len_PP = max(max_len_PP, len(phrase.split(' ')))
        length_sentence = len(sentence.split(' '))
        if length_sentence:
            phrase_length_ratios_VP = words_VP/length_sentence
            phrase_length_ratios_NP = words_NP/length_sentence
            phrase_length_ratios_PP = words_PP/length_sentence
        else:
            phrase_length_ratios_VP, phrase_length_ratios_NP, phrase_length_ratios_PP = 0, 0, 0

        # Fifth Complexity Metric - Average Phrase Length Ratio - Considering the second case(len = max(phrase))
        if max_len_VP:
            avg_VP_len = words_VP/max_len_VP
        else:
            avg_VP_len = 0
        if max_len_NP:
            avg_NP_len = words_NP/max_len_NP
        else:
            avg_NP_len = 0
        if max_len_PP:
            avg_PP_len = words_PP/max_len_PP
        else:
            avg_PP_len = 0

        sse_features.append([np_count, vp_count, pp_count, sbar_count,
                np_ratio, vp_ratio, pp_ratio, phrase_length_ratios_VP,
                phrase_length_ratios_NP, phrase_length_ratios_PP,
                avg_NP_len, avg_VP_len, avg_PP_len, rpnv])
    return sse_features

        
sse_features = get_syntactic_features(sentences)

train_data_X, test_data_X, train_data_Y, test_data_Y = train_test_split(sse_features, labels, test_size=0.3)
