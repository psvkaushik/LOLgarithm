"""
Contains the code for baseline DecisionTreeClassifier on SSE
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from SSE import train_data_X, train_data_Y, test_data_X, test_data_Y, feature_names
import matplotlib.pyplot as plt

baseline_model = DecisionTreeClassifier(max_depth=4)

baseline_model.fit(train_data_X, train_data_Y)

baseline_model.score(train_data_X, train_data_Y)

preds = baseline_model.predict(test_data_X)

print(classification_report(preds, test_data_Y))
tree.plot_tree(baseline_model, feature_names=feature_names, class_names=['True', 'False'], filled=True, fontsize=10)
plt.show()
