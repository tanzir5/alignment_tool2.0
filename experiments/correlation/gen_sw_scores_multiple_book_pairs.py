import glob
from tqdm import tqdm
import pandas as pd
from aligners.align_pipeline import align_sequences
import os
import numpy as np
import multiprocessing as mp

def process_two_books(
  book1, book2, sim_func, z_thresh, emb1_path, emb2_path, save_path_root
):

  #i, j = args[0], args[1]
  book1 = str(book1)
  book2 = str(book2)
  save_path = save_path_root + "/" + book1+"_"+book2 + ".npy"
  if os.path.exists(save_path):
    return
  seq1 = np.load(emb1_path + '/' + book1 + '.npy')
  seq2 = np.load(emb2_path + '/' + book2 + '.npy')
  print("size and books:")
  print(seq1.shape, seq2.shape)
  print(book1, book2)
  aligner = align_sequences(
    seq1, 
    seq2, 
    unit1='embedding', 
    unit2='embedding', 
    sim=sim_func, 
    z_thresh=z_thresh,
    return_aligner=True)['aligner']
  dp = aligner.dp
  np.save(save_path, dp)

def process_books(
  id1, id2, sim_func, z_thresh, emb1_path, emb2_path, save_path_root
):
  pool = mp.Pool(mp.cpu_count())
  jobs = []
  for k in range(len(id1)):
    i, j = id1[k], id2[k]
    print("starting job for", i, j)
    job = pool.apply_async(
      process_two_books, 
      (i, j, sim_func, z_thresh, emb1_path, emb2_path, save_path_root)
    )
    jobs.append(job)
    
  print("length of jobs", len(jobs))
  for i, job in tqdm(enumerate(jobs)): 
    job.get()
    print(i, "is done")

  pool.close()
  pool.join()


if __name__ ==  '__main__':
  z_thresh = 2
  sim_func = 'sbert'
  emb_path = 'data/positive_pairs/embs'
  
  df = pd.read_csv('data/positive_pairs/positive_pairs_dataset.csv')
  id1 = df['Id1'].to_list()
  id2 = df['Id2'].to_list()
  save_path_root = 'data/positive_pairs/dp_matrices'
  process_books(
    id1, id2, sim_func, z_thresh, emb_path, emb_path, save_path_root)
  
  df = pd.read_csv('data/negative_pairs/negative_pairs_dataset.csv')
  id1 = df['Id1'].to_list()
  id2 = df['Id2'].to_list()
  save_path_root = 'data/negative_pairs/dp_matrices'
  process_books(
    id1, id2, sim_func, z_thresh, emb_path, emb_path, save_path_root)


