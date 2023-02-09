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
  dp_save_path = '../../data/misc/classics_collection_'+book1+"_"+book2
  np.save(dp_save_path, dp)

z_thresh = 2
sim_func = 'sbert'
id1 = '30127'
id2 = '10748'

emb1_path = '../../data/positive_pairs/embs/'
emb2_path = '../../data/positive_pairs/embs/'

if __name__ ==  '__main__':
  process_two_books(id1, id2, sim_func, z_thresh, emb1_path, emb2_path)
