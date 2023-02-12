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

seq1 = np.load('misc/10748.npy')
seq2 = np.load('misc/30127.npy')

ret = align_sequences(
  'misc/10748.npy', 
  'misc/30127.npy', 
  unit1='embedding_path', 
  unit2='embedding_path', 
  sim='sbert', 
  z_thresh=1.5,
  return_aligner=True,
)

np.save('data/experiments/correlation/10748_30127_align_score_z_2.npy', ret['aligner'].dp)