import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
import numpy as np

# show z vs recall graph for individual types for jaccard and for sbert
# show z vs recall graph for all together for jaccard and for sbert
# show best average recall value

root_dir = 'data/experiments/pan13/outputs/'
for sim in ["jaccard", "sbert"]:
  f, ax = plt.subplots(1, 1)
  avg_f1 = {}
  for i in range(2, 6):
    if i == 5:
      df_path = root_dir + "/" + sim + "_0" + str(i) + "_para/" + "eval_stats.csv"
    else:
      df_path = root_dir + "/" + sim + "_0" + str(i) + "_sent/" + "eval_stats.csv"
    df = pd.read_csv(df_path)
    print(np.max(df['f1'].tolist()), np.argmax(df['f1'].tolist()), df['z'].iloc[np.argmax(df['f1'].tolist())])
    #sns.lineplot(ax=ax, data=df, x="z", y="f1")
    for j in range(len(df)):
      z = df['z'].iloc[j]
      f1 = df['f1'].iloc[j]
      if z not in avg_f1:
        avg_f1[z] = 0
      avg_f1[z] += f1/4

  sns.lineplot(x = list(avg_f1.keys()), y = list(avg_f1.values()))

  #ax.legend(handles=ax.lines, labels=[str(i) for i in range(2, 6)], loc='upper left')
  plt.show()
  plt.close()

df = pd.read_csv('data/experiments/pan13/thresh_perf.csv')

plt.plot(df['Threshold'].tolist(), df['Recall'].tolist(), color='r', label='recall')
plt.plot(df['Threshold'].tolist(), df['Precision'].tolist(), color='g', label='Precision')
plt.show()