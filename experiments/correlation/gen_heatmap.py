import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import string


def get_story_breaks_ticks(path):
  df = pd.read_csv(path)
  story_breaks = df['start'].tolist()
  locations = story_breaks
  labels = list(string.ascii_lowercase)[:len(locations)]
  return story_breaks, locations, labels 

book1 = '30127' # Dickens
book2 = '10748' # Classics
dp_matrix = np.load(
  '../../data/experiments/correlation/classics_collection_'+book1+"_"+book2+'.npy')
sw_scores = []
dp_2d = np.zeros((dp_matrix.shape[0], dp_matrix.shape[1]))
for i in range(dp_matrix.shape[0]):
  for j in range(dp_matrix.shape[1]):
    dp_2d[i][j] = dp_matrix[i][j][0][0]

waterman_matrix = dp_2d

dickens_story_breaks, dickens_locations, dickens_labels = get_story_breaks_ticks(
  '../../data/experiments/correlation/Dickens_story_breaks.csv')
classics_story_breaks, classics_locations, classics_labels = get_story_breaks_ticks(
  '../../data/experiments/correlation/Classics_story_breaks.csv')
ax = sns.heatmap(
  waterman_matrix, 
  xticklabels=False, 
  yticklabels=False, 
  cmap = sns.cm.rocket_r)
print(np.max(dickens_story_breaks), *ax.get_ylim())
print(np.max(classics_story_breaks), *ax.get_xlim())
print(waterman_matrix.shape)
exit(0)
ax.hlines(dickens_story_breaks, *ax.get_xlim(), linestyles='dashed')
ax.vlines(classics_story_breaks, *ax.get_ylim(), linestyles='dashed')
plt.yticks(ticks=dickens_locations, labels=dickens_labels)
plt.show()
