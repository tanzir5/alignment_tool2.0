import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import string


def get_story_breaks_ticks(path):
  df = pd.read_csv(path)
  story_breaks = df['start'].tolist()
  locations = list(np.array(story_breaks) + 30)
  story_breaks.append(np.max(df['end'].tolist()))
  print(len(story_breaks))
  labels = list(string.ascii_lowercase)[:len(locations)]
  return story_breaks, locations, labels 

book1 = '10748' # Classics
book2 = '30127' # Dickens

dp_matrix = np.load(
  '../../data/experiments/correlation/10748_30127_align_score.npy')
print(np.max(dp_matrix))


sw_scores = []
dp_2d = np.zeros((dp_matrix.shape[0], dp_matrix.shape[1]))
for i in range(dp_matrix.shape[0]):
  for j in range(dp_matrix.shape[1]):
    dp_2d[i][j] = dp_matrix[i][j][0][0]

waterman_matrix = dp_2d
classics_story_breaks, classics_locations, classics_labels = get_story_breaks_ticks(
  '../../data/experiments/correlation/Classics_story_breaks.csv')
dickens_story_breaks, dickens_locations, dickens_labels = get_story_breaks_ticks(
  '../../data/experiments/correlation/Dickens_story_breaks.csv')

ax = sns.heatmap(
  waterman_matrix, 
  xticklabels=False, 
  yticklabels=False, 
  cmap=sns.cm.rocket_r,
)
print(len(dickens_story_breaks), "YEAH")
print(dickens_story_breaks)
ax.vlines(dickens_story_breaks, *ax.get_ylim(), linestyles='dashed')
ax.hlines(classics_story_breaks, *ax.get_xlim(), linestyles='dashed')
plt.xticks(ticks=dickens_locations, labels=dickens_labels)
plt.yticks(ticks=classics_locations, labels=classics_labels)
plt.show()
