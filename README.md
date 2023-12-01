## CSC791 - Natural Language Processing

### Team 
- Kaushik Pillalamarri 
- Tanisha Khurana 
- Vikram Pande
  
# LOLgorithm
Integrating Semantic, Syntactic, and Contextual Elements for Humor Classification

## Table of contents
* [Approach](#Approach)
* [Installation](#Installation)
* [Folder Structure](#Folder_Structure)
* [Run Instructions](#Run_Instructions)




## Approach
In this work, we formulate humor recognition as a classification task in which we distinguish between humorous and non-humorous instances.
Exploring the syntactical structure involves leveraging Lexicons to capture sentiment counts within a sentence, while Statistics of Structural Elements (SSE) encapsulates the statistical insights of Noun phrases, Word phrases, and more. Unveiling the semantic layers of humor delves into Word2Vec embeddings, analyzing incongruity, ambiguity, and phonetic structures within sentences. Additionally, contextual information is harnessed through ColBERT embeddings. For each latent structure, we design a set of
features to capture the potential indicators of humor. 

## Installation
### Dependencies required:
Set up the following environment
* python 
* tensorflow 
* scikit-learn
* pandas
* numpy
* NLTK
* shap 
* seaborn 
* matplotlib
* graphviz
* pickle
* transformers
* regex
* nrclex
* tqdm
* scipy
* gensim
  
```
conda create -n lolgorithm
conda activate lolgorithm
pip install tensorflow
pip install scikit-learn
pip install pandas
pip install numpy
pip install --user -U nltk
pip install shap
pip install seaborn
pip install -U matplotlib
pip install graphviz
pip install transformers
pip install NRCLex
pip install tqdm
pip install scipy
pip install --upgrade gensim

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('cmudict')
```
## Folder_Structure
##### Following are the modules we built for the project:
- Code: Contains Modular Python Files
  1.  ```data_NRC.py```: contains the functions to generate NRCLex features (Part of Syntactical features).
  2.  ```SSE.py```: contains the functions to generate Statistics of Structural Elements features (Part of Syntactic features).
  3.  ```semantic_features.py```: contains the functions to generate Semantic features.
  4.  ```joke_scraper.py```: script to scrape jokes as unseen data.
  5.  ```test_features.py```: script to generate features for unseen data.
  6.  ```make_embed.py```: script to generate combined features - NRCLex, SSE, Semantic.
  7.  ```baseline_model.py```: Baseline Decision Tree model for Feature Engineering.
  8.  ```Colbert_training.py```: Script to train Colbert only with contextual embeddings.
  9.  ```Colbert_w_training.py```: Script to train Colbert only with contextual and hand-crafted features.
- experiments: Contains Notebooks of Experiments performed. 
  1.  ```Colbert_dataset.ipynb```: contains experiments with the Colbert dataset to get contextual embeddings.
  2.  ```baseline_book.ipynb```: contains Decision Tree analysis on NRCLex features.
  3.  ```feature_engg_notebook.ipynb```: contains feature engineering with Decision Trees and Gradient Boost with SHAP for all 4 features - nrclex, syntactic, semantic, combined.
  4.  ```final.ipynb```: Notebook for inference on unseen data.
  5.  ```semantic-word2vec-expts_v2.ipynb```: semantic feature experiments v2.
  6.  ```semantic-word2vec-expts.ipynb```: semantic feature experiments.
  7.  ```sse_book.ipynb```: Baseline Decision Tree model on structural symantic elements.
  8.  ```Colbert_train.ipynb```: Experiments to train Colbert with contextual and hand-crafted features.
- dataset: Contains Data files.
  1. ```combined-features.csv```: contains combined NRCLex, SSE, and Semantic features (200000, 33)
  2. ```dataset.csv```: ColBERT dataset containing jokes and labels.
  3. ```nrclex-features.csv```: NRCLex features
  4. ```syntactic-features.csv```: Statistics of Structural Elements (SSE) features
  5. ```semantic-features.csv```: Semantic features - Incongruity, Ambiguity, Phonetic Style
  6. ```sample_input.csv```: Sample ColBERT model input to accept.
  7. ```reddit_test_features.csv```: Unseen scraped dataset
- figures: Contains Figures and Graphs.
  - Figures of Decision Trees and SHAP analysis for feature engineering on NRCLex, Syntactic, Semantic, and Combined features to find important features for decision.
- models: Contains Saved Models.
  - Decision Tree, GradientBoost models for feature engineering on NRCLex, Syntactic, Semantic, and Combined features.

## Run_Instructions
#### How to run? Steps for running the code:
1. The directory contains all the necessary files, download/clone the repository.
2. Copy the data in the same  directory.
3. Run ```bert_combined_feats.ipynb``` used for predictions of unseen test data + Hand crafted features with BERT.
