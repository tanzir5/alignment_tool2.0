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
import heapq
import pickle

def create_plot_correlation_from_pickle(root_path):
  corrs = []
  pvalues = []
  original = pickle.load(open(root_path+"temp.p", "rb"))
  combined_pvalues_corr_entries = []
  for (x, y) in tqdm(original):
    spearman_corr = spearmanr(x, y, alternative='greater')
    corrs.append(spearman_corr.correlation)
    pvalues.append(spearman_corr.pvalue)
    combined_pvalues_corr_entries.append(CTR.spearmanr(x, y, alternative="greater"))

  print(np.mean(pvalues), np.median(pvalues))
  #print(combine(combined_pvalues_corr_entries))
  #print(stats.combine_pvalues(pvalues, method='mudholkar_george'))
  x = np.sort(corrs)
  y = np.arange(1, len(corrs)+1)
  return x, y

def create_correlation_pickle(root_path):
  ret = []
  for fname in tqdm(glob.glob(root_path+'dp_matrices/*')):
    dp_matrix = np.load(fname)
    sw_scores = []
    for i in range(dp_matrix.shape[0]):
      for j in range(dp_matrix.shape[1]):
        if dp_matrix[i][j][0][0] > 0:
          sw_scores.append((dp_matrix[i][j][0][0], i, j))

    assert(len(sw_scores) > 20)
    largest_scores = heapq.nlargest(MAX, sw_scores, key = lambda x: x[0])
    x = []
    y = []
    for k in range(MAX):
      if k < len(largest_scores): 
        x.append(largest_scores[k][1])
        y.append(largest_scores[k][2])
      else:
        x.append(-1)
        y.append(-1)
    ret.append((x,y))
  pickle.dump(ret, open(root_path+'temp.p', "wb"))

pos_x, pos_y = create_plot_correlation_from_pickle('data/positive_pairs/')
neg_x, neg_y = create_plot_correlation_from_pickle('data/negative_pairs/')
plt.xlabel('Correlation')
plt.ylabel('Cumulative frequency') 
plt.plot(pos_x, pos_y, marker=".", label="related")
plt.plot(neg_x, neg_y, marker=".", label="unrelated")
plt.legend(loc="upper left")
plt.show()


'''
For related pairs,
Discrete combined_P_Value(pvalue=9.9999990000001e-08, std=9.9999990000001e-08)
continuous Combined_P_Value(statistic=415.59727959218685, pvalue=7.288915253647748e-93)
mean = 0.09773091109259377 
median = 0.002850140147225905
For unrelated pairs,
Discrete Combined_P_Value(pvalue=1.599999840000016e-06, std=3.8729800541708124e-07)
Continuous Combined_P_Value(statistic=61.6886421873683, pvalue=1.4644438017404043e-06)
mean = 0.4597758698430436 
median = 0.4062815613729491
'''