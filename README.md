## CSC791 - Natural Language Processing

### Team 

- Kaushik Pillalamari 
- Tanisha Khurana 
- Vikram Pande
  
# LOLgarithm
Integrating Semantic, Syntactic, and Contextual Elements for Humor Classification

This readme contains the directory structure. 

##### Following are the modules we built for the project:
- Code
  1.  ```data_NRC.py```: contains the functions to generate NRCLex features (Part of Syntactical features).
  2.  ```SSE.py```: contains the functions to generate Statistics of Structural Elements features (Part of Syntactic features).
  3.  ```semantic_features.py```: contains the functions to generate Semantic features.
  4.  ```joke_scraper.py```: script to scrape jokes as unseen data.
  5.  ```test_features.py```: script to generate features for unseen data.
  6.  ```make_embed.py```: script to generate combined features - NRCLex, Syntactic, Semantic.
  7.  ```baseline_model.py```: Baseline Decision Tree model for Feature Engineering.
- experiments
- dataset
- figures
- models

#### How to run? Steps for running the code:
1. The directory contains all the necessary files, download/clone the repository.
2. Copy the data in the same  directory.
3. Run ```notebook.ipynb``` that generates the prediction files.
