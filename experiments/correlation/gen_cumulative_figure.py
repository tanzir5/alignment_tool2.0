import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import glob 
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
from combine_pvalues_discrete import CTR, combine
from scipy import stats

MAX = 20
ids1 = []
ids2 = []
corrs = []
pvalues = []
results = []

def get_book_names(names):
  names = names.split('/')[-1]
  names = names[:-4]
  names = names.split("_")
  return names

print(len(glob.glob('../../data/positive_pairs/dp_matrices/*')))
for fname in tqdm(glob.glob('../../data/positive_pairs/dp_matrices/*')):
  dp_matrix = np.load(fname)
  sw_scores = []
  for i in range(dp_matrix.shape[0]):
    for j in range(dp_matrix.shape[1]):
      if dp_matrix[i][j][0][0] > 0:
        sw_scores.append((dp_matrix[i][j][0][0], i, j))

  assert(len(sw_scores) > 20)
  sw_scores.sort(reverse=True)
  x = []
  y = []
  for k in range(MAX):
    if k < len(sw_scores): 
      x.append(sw_scores[k][1])
      y.append(sw_scores[k][2])
    else:
      x.append(-1)
      y.append(-1)
  spearman_corr = spearmanr(x, y, alternative='greater')
  corrs.append(spearman_corr.correlation)
  pvalues.append(spearman_corr.pvalue)
  names = get_book_names(fname)
  ids1.append(names[0])
  ids2.append(names[1])
  results.append(CTR.spearmanr(x, y, alternative="greater"))
  #if len(results) == 3:
  #  break
df = pd.DataFrame({'id1':ids1, 'id2':ids2, 'correlation':corrs, 'pvalue':pvalues})
df.to_csv('../../data/positive_pairs/correlations.csv')


print(combine(results))
print(stats.combine_pvalues(pvalues, method='mudholkar_george'))

x = np.sort(corrs)
  
# get the cdf values of y
y = np.arange(1, len(corrs)+1)
plt.xlabel('correlation')
plt.ylabel('cumulative frequency') 
plt.plot(x, y, marker='o')
plt.show()