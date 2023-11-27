## CSC791 - Natural Language Processing

### Team 

- Kaushik Pillalamari 
- Tanisha Khurana 
- Vikram Pande
  
# LOLgarithm
Integrating Semantic, Syntactic, and Contextual Elements for Humor Classification

In this work, we formulate humor recognition as a classification task in which we distinguish between humorous and non-humorous instances.
Exploring the syntactical structure involves leveraging Lexicons to capture sentiment counts within a sentence, while Statistics of Structural Elements (SSE) encapsulates the statistical insights of Noun phrases, Word phrases, and more. Unveiling the semantic layers of humor delves into Word2Vec embeddings, analyzing incongruity, ambiguity, and phonetic structures within sentences. Additionally, contextual information is harnessed through ColBERT embeddings. For each latent structure, we design a set of
features to capture the potential indicators of humor. 

This readme contains the directory structure. 

##### Following are the modules we built for the project:
- Code
  1.  ```data_NRC.py```: contains the functions to generate NRCLex features (Part of Syntactical features).
  2.  ```SSE.py```: contains the functions to generate Statistics of Structural Elements features (Part of Syntactic features).
  3.  ```semantic_features.py```: contains the functions to generate Semantic features.
  4.  ```joke_scraper.py```: script to scrape jokes as unseen data.
  5.  ```test_features.py```: script to generate features for unseen data.
  6.  ```make_embed.py```: script to generate combined features - NRCLex, SSE, Semantic.
  7.  ```baseline_model.py```: Baseline Decision Tree model for Feature Engineering.
- experiments
  1.  ```Colbert_dataset.ipynb```: contains experiments with the Colbert dataset to get contextual embeddings.
  2.  ```baseline_book.ipynb```: contains Decision Tree analysis on NRCLex features.
  3.  ```feature_engg_notebook.ipynb```: contains feature engineering with Decision Trees and Gradient Boost with SHAP for all 4 features - nrclex, syntactic, semantic, combined.
  4.  ```final.ipynb```: Notebook for inference on unseen data.
  5.  ```semantic-word2vec-expts_v2.ipynb```: semantic feature experiments v2.
  6.  ```semantic-word2vec-expts.ipynb```: semantic feature experiments.
  7.  ```sse_book.ipynb```: Baseline Decision Tree model on structural symantic elements.
- dataset
  1. ```combined-features.csv```: contains combined NRCLex, SSE, and Semantic features (200000, 33)
  2. ```dataset.csv```: ColBERT dataset containing jokes and labels.
  3. ```nrclex-features.csv```: NRCLex features
  4. ```syntactic-features.csv```: Statistics of Structural Elements (SSE) features
  5. ```semantic-features.csv```: Semantic features - Incongruity, Ambiguity, Phonetic Style
  6. ```sample_input.csv```: Sample ColBERT model input to accept.
  7. ```reddit_test_features.csv```: Unseen scraped dataset
- figures
  - Figures of Decision Trees and SHAP analysis for feature engineering on NRCLex, Syntactic, Semantic, and Combined features to find important features for decision.
- models
  - Decision Tree, GradientBoost models for feature engineering on NRCLex, Syntactic, Semantic, and Combined features.

#### How to run? Steps for running the code:
1. The directory contains all the necessary files, download/clone the repository.
2. Copy the data in the same  directory.
3. Run ```notebook.ipynb``` that generates the prediction files.
