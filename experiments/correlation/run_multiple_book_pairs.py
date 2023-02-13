import sys
sys.path.append('../../../alignment_tool2.0/codes')

import glob
from tqdm import tqdm
import pandas as pd
from align_pipeline import align_sequences
import os
import xml.etree.ElementTree as ET
import numpy as np
import multiprocessing as mp

def process_two_books(book1, book2, sim_func, z_thresh, emb1_path, emb2_path):
  #i, j = args[0], args[1]
  book1 = str(book1)
  book2 = str(book2)
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
    z_thresh=z_thresh)
  dp = aligner.dp
  dp_save_path = '../../data/positive_pairs/dp_matrices/'+book1+"_"+book2
  np.save(dp_save_path, dp)

def process_books(id1, id2, sim_func, z_thresh, emb1_path, emb2_path):
  pool = mp.Pool(mp.cpu_count())
  jobs = []
  for k in range(len(id1)):
    i, j = id1[k], id2[k]
    print("starting job for", i, j)
    job = pool.apply_async(
      process_two_books, (i, j, sim_func, z_thresh, emb1_path, emb2_path))
    jobs.append(job)
    
  print("length of jobs", len(jobs))
  for i, job in tqdm(enumerate(jobs)): 
    job.get()
    print(i, "is done")

  pool.close()
  pool.join()


z_thresh = 3
sim_func = 'sbert'
df = pd.read_csv('../../data/positive_pairs/positive_pairs_dataset.csv')
id1 = df['Id1'].to_list()
id2 = df['Id2'].to_list()

emb1_path = '../../data/positive_pairs/embs/'
emb2_path = '../../data/positive_pairs/embs/'

if __name__ ==  '__main__':
  process_books(id1, id2, sim_func, z_thresh, emb1_path, emb2_path)


