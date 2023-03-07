import glob
from tqdm import tqdm
import pandas as pd
from aligners.align_pipeline import align_sequences
from aligners.preprocessor import Preprocessor
from aligners.smith_waterman import Aligner
import os
import numpy as np
import multiprocessing as mp
import sys

save_df_busy = False

def check_already_done(df_path, book1, book2):
  global save_df_busy
  if os.path.exists(df_path) == False:
    return False
  while(save_df_busy):
    time.sleep(0.1)
  save_df_busy = True
  df = pd.read_csv(df_path)
  save_df_busy = False
  if len(df[(df['Id1'] == book1) & df['Id2'] == book2]) > 0:
    return True
  else:
    return False

def save_in_df(df_path, book1, book2, max_sw_score):
  global save_df_busy
  while(save_df_busy):
    time.sleep(0.1)
  save_df_busy = True
  df = pd.DataFrame(
        {'Id1':[book1], 'Id2':[book2], 'max_sw_score':[max_sw_score]}
      )
  df.to_csv(df_path, mode='a', header=os.path.exists(df_path)==False)
  save_df_busy = False

def get_texts_from_csv(path):
  if os.path.exists(path) is False:
    return None
  df = pd.read_csv(path)
  return df['para'].tolist()

def process_two_books(
  book1, book2, sim_func, z_thresh, save_df_path, text_path, emb_path
):
  book1 = str(book1)
  book2 = str(book2)
  if sim_func == 'sbert':
    seq1 = emb_path + '/' + book1 + '.npy'
    seq2 = emb_path + '/' + book2 + '.npy'
    if os.path.exists(seq1) is False or os.path.exists(seq2) is False:
      return
    unit1 = 'embedding_path'
    unit2 = 'embedding_path'
  else:
    seq1 = get_texts_from_csv(text_path+'/'+book1+'.csv')
    seq2 = get_texts_from_csv(text_path+'/'+book2+'.csv')
    if seq1 is None or seq2 is None:
      return
    unit1 = 'paragraph'
    unit2 = 'paragraph'

  if check_already_done(save_df_path, book1, book2):
    return
  '''
  aligner = align_sequences(
    seq1, 
    seq2, 
    unit1=unit1, 
    unit2=unit2, 
    sim=sim_func, 
    z_thresh=z_thresh,
    return_aligner=True)['aligner']
  '''
  sim_config = {}
  sim_config['func'] = sim_func
  sim_config['threshold'] = z_thresh
  preprocessor = Preprocessor(
    seq1, 
    seq2, 
    size_a=unit1,
    size_b=unit2,
    sim_config=sim_config,
    clip_length=None,
    save_emb_dirs=None,
    no_gap=False,
    double_break_for_paragraphs=True
  )
  aligner = Aligner(
    preprocessor.sim_matrix, 
    ignore=None,
    no_gap=False
  )
  aligner.compute_smith_waterman()
  max_sw_score = np.max(aligner.dp)
  save_in_df(save_df_path, book1, book2, max_sw_score)

def process_books(
  Ids1, Ids2, sim_func, z_thresh, save_df_path, text_path, emb_path
):
  pool = mp.Pool(mp.cpu_count())
  jobs = []
  for k in range(len(Ids1)):
    book1, book2 = Ids1[k], Ids2[k]
    job = pool.apply_async(
      process_two_books, 
      (book1, book2, sim_func, z_thresh, save_df_path, text_path, emb_path)
    )
    jobs.append(job)
    
  print("length of jobs", len(jobs))
  for i, job in tqdm(enumerate(jobs)): 
    job.get()
    print(i, "is done")

  pool.close()
  pool.join()


if __name__ ==  '__main__':
  '''
  z_thresh = 2
  sim_func = 'sbert'
  emb_path = 'data/positive_pairs/embs'
  save_path_root_positive = 'data/positive_pairs/dp_matrices/'
  save_path_root_negative = 'data/negative_pairs/dp_matrices'
  '''

  text_path = 'data/positive_pairs/text_csv'
  emb_path = 'data/positive_pairs/embs'
  z_thresh = 2
  sim_func = sys.argv[1]
  positive_sw_df_path = ('data/experiments/AUC_positive_negative/positive_' + 
                          sim_func + 
                          "_sw_scores.csv")
  df = pd.read_csv('data/positive_pairs/positive_pairs_dataset.csv')
  Ids1 = df['Id1'].to_list()
  Ids2 = df['Id2'].to_list()
  process_books(
    Ids1, 
    Ids2, 
    sim_func, 
    z_thresh, 
    positive_sw_df_path, 
    text_path=text_path,
    emb_path=emb_path 
  )
  
  df = pd.read_csv('data/negative_pairs/negative_pairs_dataset.csv')
  Ids1 = df['Id1'].to_list()
  Ids2 = df['Id2'].to_list()
  negative_sw_df_path = ('data/experiments/AUC_positive_negative/negative_' + 
                          sim_func + 
                          "_sw_scores.csv")
  process_books(
    Ids1, 
    Ids2, 
    sim_func, 
    z_thresh, 
    negative_sw_df_path, 
    text_path=text_path,
    emb_path=emb_path 
  )
