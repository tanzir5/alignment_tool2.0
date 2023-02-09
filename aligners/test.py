
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


nlp = spacy.load("en_core_web_sm")
with open('test_input.txt') as f:
  text = f.read()

def _segment_into_paragraph(text):
    indices = [{'break_st':m.start(), 'break_end':m.end()-1} 
                for m in re.finditer('\n\n\s*', text)]
    paragraphs = []
    for i in range(len(indices)):
      if i == 0:
        indices[i]['text_st'] = 0
      else:
        indices[i]['text_st'] = indices[i-1]['break_end']+1
      indices[i]['text_end'] = indices[i]['break_st']-1
      paragraphs.append(text[indices[i]['text_st']:indices[i]['break_end']+1])
    if indices[-1]['break_end'] != len(text)-1:
      indices.append({
        'text_st':indices[-1]['break_end']+1,
        'text_end':len(text)-1,
        'break_st':len(text)-1,
        'break_end':len(text)-1
        })
      paragraphs.append(text[indices[-1]['text_st']:indices[-1]['break_end']+1])
    for i in range(len(indices)):
      indices[i] = {'st':indices[i]['text_st'], 'ed':indices[i]['text_end']}
    return paragraphs, indices

_segment_into_paragraph(text)