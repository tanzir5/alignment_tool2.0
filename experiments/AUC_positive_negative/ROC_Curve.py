from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

def get_y(sim):
  df_pos = pd.read_csv('data/experiments/AUC_positive_negative/positive/positive_sw_scores_'+sim+'.csv')
  pos_scores = df_pos['score'].tolist()
  pos_labels = np.ones(len(pos_scores))
  
  df_neg = pd.read_csv('data/experiments/AUC_positive_negative/negative/negative_sw_scores_'+sim+'.csv')
  neg_scores = df_neg['score'].tolist()
  neg_labels = np.zeros(len(neg_scores))

  labels = list(pos_labels) + list(neg_labels)
  scores = list(pos_scores) + list(neg_scores)
  return labels, scores

plt.rcParams.update({'font.size': 13.5})
fig, ax = plt.subplots()

for sim in ['sbert', 'glove', 'jaccard', 'tf_idf']:
  y_label, y_scores = get_y(sim)
  RocCurveDisplay.from_predictions(y_label, y_scores, ax=ax, name=sim)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()