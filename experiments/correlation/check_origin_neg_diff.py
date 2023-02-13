import pandas as pd

positive_df = pd.read_csv('data/positive_pairs/positive_pairs_dataset.csv')
negative_df = pd.read_csv('data/negative_pairs/negative_pairs_dataset.csv')

for i in range(len(negative_df)):
  id1 = negative_df['Id1'].iloc[i]
  id2 = negative_df['Id2'].iloc[i]
  id1_origin = None
  id2_origin = None
  if (positive_df['Id1'] == id1).any():
    id1_origin = list(positive_df[positive_df['Id1']==id1]['Origin Lang'])[0]
  else:
    id1_origin = list(positive_df[positive_df['Id2']==id1]['Origin Lang'])[0]
    

  if (positive_df['Id1'] == id2).any():
    id2_origin = list(positive_df[positive_df['Id1']==id2]['Origin Lang'])[0]
  else:
    id2_origin = list(positive_df[positive_df['Id2']==id2]['Origin Lang'])[0]
    
  print(id1_origin, id2_origin)
  assert(id1_origin != id2_origin)  