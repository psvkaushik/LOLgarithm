import pandas as pd

ambiguity_features = pd.read_csv('D:/Work/MS COURSES/COURSES FALL 23/CSC 791 Natural Language Processing/LOLgorithm/LOLgarithm/dataset/ambi_features_full.csv')

ambiguity_features_pths = pd.DataFrame({
    'farmost_path': ambiguity_features['farthest'],
    'closest_path': ambiguity_features['closest']
})

ambiguity_features_pths.to_csv('D:/Work/MS COURSES/COURSES FALL 23/CSC 791 Natural Language Processing/LOLgorithm/LOLgarithm/dataset/ambiguity_features_pths.csv')
