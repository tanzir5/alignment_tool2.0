import spacy
from multiset import *
import numpy as np
from scipy import stats
from scipy.special import expit, logit
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
from tqdm import tqdm
import torch
import torchtext
import re
from transformers import BertModel, BertTokenizer
import numpy as np 
import pandas as pd
from aligners.align_pipeline import align_sequences


def read_paras(path):
  df = pd.read_csv(path)
  return df['para'].tolist()

seq1 = read_paras('data/positive_pairs/text_csv/10748.csv')
seq2 = read_paras('data/positive_pairs/text_csv/30127.csv')

ret = align_sequences(
  seq1, 
  seq2, 
  unit1='paragraph', 
  unit2='paragraph', 
  sim='sbert', 
  save_emb_dirs=['misc/10748.npy', 'misc/30127.npy'],
  return_aligner=True,
)

np.save('data/experiments/correlation/10748_30127_align_score.npy', ret['aligner'].dp)