#TODO: If seq1 and seq2 are strings instead of lists, segment them into list of 
#      size_a and size_b levels. Use a segmenter function to do that maybe?
from multiset import *
from scipy import stats
from scipy.special import expit, logit
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

import copy
import numpy as np
import spacy
import torch
import torchtext
import re
import os
import json

INF = 1e9
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm")
all_stopwords = nlp.Defaults.stop_words
print("Running on ", DEVICE)

VALID_TOKEN_SIZES = [
  "character", "word", "sentence", "paragraph", "embedding", "embedding_path"
  ]

VALID_SIM_FUNCS = [
  "jaccard", "bert", "sbert", "glove", "hamming", 
  "tf_idf", "jaccard_glove_nn", "exact"]

GLOVE_DIM = 300
SBERT_MODEL_NAME = 'msmarco-distilbert-cos-v5'

DEFAULT_Z_THRESHOLD = 1

DEFAULT_MEAN_VALUES = {
  'jaccard': 0.004,
  'bert': 0.5,
  'sbert': 0.097,
  'glove': 0.49,
  'hamming': 0.07,
  'tf_idf': 0.03,
  'jaccard_glove_nn': 0.5,
  'exact': 0
  }

DEFAULT_STD_VALUES = {
  'jaccard': 0.01,
  'bert': 0.1,
  'sbert': 0.099,
  'glove': 0.16,
  'hamming': 0.06,
  'tf_idf': 0.03,
  'jaccard_glove_nn': 0.1,
  'exact': 1,
  }

DEFAULT_SIM_CONFIG = {
  'func': 'jaccard', 
  'threshold': DEFAULT_Z_THRESHOLD, 
  'mean': DEFAULT_MEAN_VALUES['jaccard'],
  'std': DEFAULT_STD_VALUES['jaccard'],
  }

class Preprocessor:
  '''
  if you pass embedding or embedding directory as size_a, then the 
  sim_config['func']  must be one of bert, sbert or glove. 
  '''

  def __init__(
    self, 
    seq_a, 
    seq_b=None,
    size_a='paragraph',
    size_b='paragraph',
    sim_config=None,
    clip_length=None,
    save_emb_dirs=None,
    create_embs_only=False,
    no_gap=True,
    double_break_for_paragraphs=True,
  ):
    self.sbert = None
    self.glove = None
    self.bert = None

    self.double_break_for_paragraphs = double_break_for_paragraphs
    self.indices_a = None
    self.indices_b = None
    if create_embs_only:
      self.validate_for_create_embs_only(seq_b, sim_config)
      if isinstance(seq_a, str) and size_a != 'embedding_path':
        seq_a, self.indices_a = self._segment(seq_a, size_a)
      
      self.seq_a_emb = self.create_embs_only(seq_a, size_a, sim_config['func'])
      self.save_embs(save_emb_dirs)
      return
    
    try:  
      assert(seq_b is not None)
    except:
      print("seq_b must not be none when create_embs_only is False")
      exit(0)
    if sim_config is None:
      sim_config = DEFAULT_SIM_CONFIG
    """Initializes preprocessor."""
    self.no_gap = no_gap
    self.unmodified_seq_a = copy.deepcopy(seq_a)
    self.unmodified_seq_b = copy.deepcopy(seq_b)
    self._check_validity(seq_a, seq_b, size_a, size_b, sim_config)
    sim_config = self._init_config(sim_config)
    size_compatibility = self._check_size_compatibility(size_a, size_b)
    if size_compatibility == False: 
      raise Exception(
        size_a + " and " + size_b + " are not mutually compatible")
  
    if clip_length is not None:
    	seq_a = seq_a[:clip_length]
    	seq_b = seq_b[:clip_length]

    if isinstance(seq_a, str) and size_a != 'embedding_path':
      seq_a, self.indices_a = self._segment(seq_a, size_a)
      seq_b, self.indices_b = self._segment(seq_b, size_b)  
    elif size_a == 'embedding_path':
      seq_a, self.indices_a = self.load_embedding(seq_a)
      seq_b, self.indices_b = self.load_embedding(seq_b)
      size_a = 'embedding'
      size_b = 'embedding'

    self.final_seq_a = seq_a
    self.final_seq_b = seq_b
    self.sim_config = sim_config
    
    self.raw_sim_matrix = self.get_sim_matrix(seq_a, seq_b, size_a, size_b)
    if sim_config['func'] == 'exact':
      self.sim_matrix = self.raw_sim_matrix
    else:
      self.sim_matrix = self.normalize(self.raw_sim_matrix)
    if self.no_gap:
      no_gapper_modifier = lambda x: x if x >= 0 else -INF
      self.sim_matrix = np.vectorize(no_gapper_modifier)(self.sim_matrix)
    
    self.save_embs(save_emb_dirs)

  
  #create_embs_only functions
  def create_embs_only(self, seq_a, size_a, emb_func):
    if isinstance(seq_a, str):
      seq_a = self._segment(seq_a, size_a)

    if emb_func == 'bert':
      seq_emb = self.get_bert_embedding(seq_a)
    elif emb_func == 'glove':
      seq_emb = self.get_glove_embedding_mean(seq_a)
    else:
      seq_emb = self.get_sbert_embedding(seq_a)
    return seq_emb

  #init functions
  def _init_config(self, sim_config):
    if 'threshold' not in sim_config:
      sim_config['threshold'] = DEFAULT_Z_THRESHOLD
    if 'mean' not in sim_config:
      sim_config['mean'] = DEFAULT_MEAN_VALUES[sim_config['func']]
    if 'std' not in sim_config:
      sim_config['std'] = DEFAULT_STD_VALUES[sim_config['func']]
    return sim_config

  def _init_glove(self):
    self.glove = torchtext.vocab.GloVe(name="840B", dim= GLOVE_DIM) 

  def _init_sbert(self):
    self.sbert = SentenceTransformer(SBERT_MODEL_NAME).to(DEVICE)
    
  def _init_bert(self):
    self.bert = {
      'tokenizer':BertTokenizer.from_pretrained('bert-base-uncased'),
      'model':BertModel.from_pretrained('bert-base-uncased')
    }

  #check everything's ok functions
  def validate_for_create_embs_only(self, seq_b, sim_config):
    try:
      assert(seq_b is None)
    except:
      print("seq_b should be none if create_embs_only is true")
    try:
      assert(sim_config['func'] in ['sbert', 'glove', 'bert'])
    except:
      print("sim_config['func'] must be one of",
        "['sbert', 'glove', 'bert'] when create_embs_only is True")
      exit(0)

  def _check_size_compatibility(self, size_a, size_b):
    is_compatible = True
    if size_a == 'embedding_path':
      is_compatible = (size_b == 'embedding_path')
    elif size_a == 'character':
      is_compatible = (size_b == 'character')
    elif size_a == 'word':
      is_compatible = (size_b == 'word')
    elif size_a == 'embedding':
      is_compatible = (size_b == 'embedding')
    else:
      is_compatible = (size_b != 'character' and size_b != 'word')
    
    return is_compatible
  
  def _check_validity(self, seq_a, seq_b, size_a, size_b, sim_config):
    try:
      assert(type(seq_a) == type(seq_b))
    except:
      print(
        "seq_a type(",type(seq_a), ") and seq_b type(",type(seq_b), 
        ") are not the same.")
      exit(0)

    try:
      if size_a == 'embedding':
        assert(isinstance(seq_a, str)==False)
      if size_b == 'embedding':
        assert(isinstance(seq_b, str)==False)
    except:
      print("size cannot be embedding or embedding_path when sequence is str.")
      exit(0)

    try:
      if size_a == 'embedding' or size_a == 'embedding_path':
        assert(sim_config['func'] in ['bert', 'sbert', 'glove'])
    except:
      print("embedding passed, but sim_config['func'] is not "
      "one of ['bert', 'sbert', 'glove']")    
      exit(0)

    try:
      if size_a == 'word':
        assert(sim_config['func'] in ['glove', 'exact'])
    except:
      print("unit size is word, but sim_config['func'] is not "
      "one of ['glove', 'exact']")    
      exit(0)

    try:
      assert (size_a in VALID_TOKEN_SIZES and size_b in VALID_TOKEN_SIZES)
    except:
      print(size_a, "or", size_b, "is not supported.")
      print("list of supported element sizes:", VALID_TOKEN_SIZES)
      exit(0)

    try:
      assert('func' in sim_config)
    except:
      print("sim_config does not have func as a key.")
      exit(0)
  
    try:
      if size_a == 'embedding' or size_a == 'embedding_path':
        assert(sim_config['func'] in ['bert', 'sbert', 'glove'])
    except:
      print("embedding passed, but sim_config['func'] is not"
      "one of ['bert', 'sbert', 'glove']")    
      exit(0)

    try:
      assert (sim_config['func'] in VALID_SIM_FUNCS)
    except:
      print(sim_config['func'], "is not supported.")
      print("list of supported sim functions:", VALID_SIM_FUNCS)
      exit(0)

  #segment functions
  def _segment_into_character(self, text):
    return [ch for ch in text]

  def _segment_into_word(self, text):
    doc = nlp(text)
    words = []
    indices = []
    for word in doc:
      words.append(word.text)
      indices.append({'st':word.idx, 'end':word.idx+len(word.text)-1})
    return words, indices

  def _segment_into_sentence(self, text):
    doc = nlp(text)
    sentences = []
    indices = []
    for sent in doc.sents:
      sentences.append(sent.text)
      indices.append({'st':sent.start_char, 'end':sent.end_char-1})
    return sentences, indices

  def _segment_into_paragraph(self, text, double_blank=True):
    if double_blank == True:
      pattern = '\n\n\s*'
    else:
      pattern = '\n\s*'
    indices = [{'break_st':m.start(), 'break_end':m.end()-1} 
                  for m in re.finditer(pattern, text)]
    paragraphs = []
    if len(indices) == 0:
      indices = [{
      'text_st': 0,
      'text_end': len(text)-1,
      'break_st': len(text)-1, 
      'break_end': len(text)-1,
      }]
      paragraphs.append(text)
    else:
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
      indices[i] = {'st':indices[i]['text_st'], 'end':indices[i]['text_end']}
    return paragraphs, indices

  def _segment(self, text, size):
    if size == 'character':
      return self._segment_into_character(text)
    elif size == 'word':
      return self._segment_into_word(text)
    elif size == "sentence":
      return self._segment_into_sentence(text)
    elif size == 'paragraph':
      return self._segment_into_paragraph(
        text, self.double_break_for_paragraphs
      )
    else:
      assert(False)

  #misc utility functions 
  def normalize(self, sim_matrix):
    '''
      normalization is done by doing the following: 
      1. take z-score
      2. subtract threshold.
      3. take sigmoid.
      4. make the range from -1 to +1 by linear transformation.
    '''

    #get z-score
    sim_matrix -= self.sim_config['mean']
    sim_matrix /= self.sim_config['std']
    
    sim_matrix -= self.sim_config['threshold']
    sim_matrix = expit(sim_matrix) 
    sim_matrix *= 2
    sim_matrix -= 1
    try:
      assert((-1 <= sim_matrix).all() and (sim_matrix <= 1).all()) 
    except:
      print("all members of sim_matrix are not within the range -1 to +1.")
      print("sim_matrix min and max")
      print(np.min(sim_matrix), np.max(sim_matrix))
      print("if you want to continue," 
        "type 1, otherwise type anything else to terminate")
      command = input()
      if command != '1':
        exit(0)
    
    return sim_matrix

  def get_words_multiset(self, token):
    words_multiset = Multiset()
    doc = nlp(token.lower())
    for word in doc:
      if word.text not in all_stopwords and word.text.isalnum():
        words_multiset.add(word.text)
    return words_multiset

  def get_jaccard_value(self, set_A, set_B):
    return len((set_A & set_B)) / max(1,len((set_A | set_B))) 


  #embedding utility functions
  def save_embs_in_dict(self, emb, indices, save_path):
    tmp_dict = {'embedding': emb.tolist(), 'indices':indices}
    with open(save_path, "w") as fp:
      json.dump(tmp_dict, fp)  # encode dict into JSON
  
  def save_embs(self, save_emb_dirs):
    if save_emb_dirs is None:
      return
    try:
      assert(hasattr(self, 'seq_a_emb'))
    except:
      print("save_emb_dirs is passed but the code is not saving embeddings!")
      exit(0)
    
    try:
      self.save_embs_in_dict(self.seq_a_emb, self.indices_a, save_emb_dirs[0])
      if len(save_emb_dirs) > 1:
        self.save_embs_in_dict(self.seq_b_emb, self.indices_b, save_emb_dirs[1])
    except Exception as exception:
      print(exception)
      print("error in saving embeddings.Possible reasons:")
      print("1. save_emb_dirs is not a list of strings")
      print("2. the object has not created required number of embeddings.")
      exit(0)
  
  def load_embedding(self, path):
    if os.path.exists(path):
      if path.endswith('npy'):
        embs = np.load(path)
        indices = None
      elif path.endswith('.json'):
        json_obj = json.load(open(path))
        embs = np.array(json_obj['embedding'])
        indices = json_obj['indices']
      return embs, indices
    else:
      raise Exception(path + " does not exist.") 

  def get_embedding_sim(self, seq_a, seq_b):
    sim = util.cos_sim(seq_a, seq_b)
    return sim.detach().cpu().numpy()  
  
  def get_glove_embedding(self, seq):
    if self.glove is None:
      self._init_glove()
    ret = []
    for word in seq:
      cur_emb = self.glove[word] 
      ret.append(cur_emb.unsqueeze(0))
    ret = torch.cat(ret)
    return ret

  def get_glove_embedding_mean(self, seq):
    if self.glove is None:
      self._init_glove()
    ret = []
    for element in tqdm(seq):
      doc = nlp(element.lower())
      mean_emb = torch.zeros(300)
      for word in doc:
        if word.text not in all_stopwords and word.text.isalnum():
          cur_emb = self.glove[word.text] 
          if (cur_emb == 0).all() == False:
            mean_emb += cur_emb 
            count += 1 
      if count > 0:
        mean_emb /= count    
      ret.append(mean_emb.unsqueeze(0))      
    ret = torch.cat(ret)
    return ret.detach().cpu().numpy()

  def get_sbert_embedding(self, seq):
    if self.sbert is None:
      self._init_sbert()
    ret_np = self.sbert.encode(seq)
    return ret_np 

  def get_bert_embedding(self, tokens):
    if self.bert is None:
      self._init_bert()
    tokenized_text = self.bert['tokenizer'](
      texts,
      padding='max_length',
      truncation=True,
      max_length=512,
      return_special_tokens_mask=True,
    )
    input_ids = torch.tensor(tokenized_text['input_ids'])
    attention_mask = torch.tensor(tokenized_text['attention_mask'])
    outputs = self.bert['model'](input_ids, attention_mask=attention_mask)
    ret = outputs['pooler_output']
    ret_np = ret.detach().cpu().numpy()
    return ret_np  


  #similarity functions
  def get_sim_matrix(self, seq_a, seq_b, size_a, size_b):
    if size_a == 'character':
      return self.get_sim_matrix_char_char(seq_a, seq_b)
    elif size_a == 'word':
      return self.get_sim_matrix_word_word(seq_a, seq_b)
    elif size_a == 'embedding':
      return self.get_embedding_sim(seq_a, seq_b) 
    else:
      return self.get_sim_matrix_text_text(seq_a, seq_b)
  
  def get_sim_matrix_word_word(self, seq_a, seq_b):
    func = self.sim_config['func']
    if func == 'exact':
      return self.get_exact_sim(seq_a, seq_b)
    elif func == 'glove':
      seq_a = self.get_glove_embedding(seq_a) 
      seq_b = self.get_glove_embedding(seq_b)
      return self.get_embedding_sim(seq_a, seq_b)
    else:
      assert(False)

  def get_exact_sim(self, seq_a, seq_b):
    ret = np.zeros((len(seq_a), len(seq_b)))
    for i in range(len(seq_a)):
      for j in range(len(seq_b)):
        if seq_a[i] == seq_b[j]:
          ret[i][j] = 1
        else:
          ret[i][j] = -1
    return ret
  
  def get_sim_matrix_text_text(self, seq_a, seq_b):
    func = self.sim_config['func']
    if func == 'jaccard':
      raw_sim_matrix = self.get_jaccard_sim(seq_a, seq_b)
    elif func == 'tf_idf':
      raw_sim_matrix = self.get_tf_idf_sim(seq_a, seq_b)
    elif func == 'hamming' : 
      raw_sim_matrix = self.get_hamming_sim(seq_a, seq_b)
    elif func == 'overlapping_glove_sim' : 
      raw_sim_matrix = self.get_jaccard_glove_nn(seq_a, seq_b)
    elif func in ['bert', 'glove', 'sbert']:
      if func == 'bert':
        seq_a = self.get_bert_embedding(seq_a)
        seq_b = self.get_bert_embedding(seq_b)
      elif func == 'glove':
        seq_a = self.get_glove_embedding_mean(seq_a) 
        seq_b = self.get_glove_embedding_mean(seq_b)
      else:
        seq_a = self.get_sbert_embedding(seq_a)
        seq_b = self.get_sbert_embedding(seq_b)
      self.seq_a_emb = seq_a
      self.seq_b_emb = seq_b
      raw_sim_matrix = self.get_embedding_sim(seq_a, seq_b)
      
    return raw_sim_matrix

  def get_tf_idf_sim(self, seq_a, seq_b):
    vectorizer = TfidfVectorizer(stop_words = 'english', max_features=768)
    all_tokens = list(copy.deepcopy(seq_a))
    all_tokens.extend(list(seq_b))
    vectorizer = vectorizer.fit(all_tokens)
    
    tf_idf_A = vectorizer.transform(seq_a).todense()
    tf_idf_B = vectorizer.transform(seq_b).todense()
    
    ret = util.cos_sim(tf_idf_A, tf_idf_B)
    ret_np = ret.detach().cpu().numpy()  
    return ret_np

  def get_jaccard_sim(self, seq_a, seq_b):
    sim_matrix = np.zeros((len(seq_a), len(seq_b)))
    words_multisets_A = []
    words_multisets_B = []
    for (i, element) in enumerate(seq_a):
      words_multisets_A.append(self.get_words_multiset(element))
    
    for (i, element) in enumerate(seq_b):
      words_multisets_B.append(self.get_words_multiset(element))
        
    for (i, token_A) in tqdm(enumerate(seq_a)):
      for (j, token_B) in enumerate(seq_b):
        sim_matrix[i][j] = self.get_jaccard_value(
          words_multisets_A[i], words_multisets_B[j]
        )
    return sim_matrix

  #not fixed functions  
  def _segment_into_seq(self, tokens, unit_size):
    if unit_size == 'word':
      return words.split()

  def _get_hamming_sim_seq(self, seq_a, seq_b):
    if len(seq_a) > len(seq_b):
      seq_a, seq_b = seq_b, seq_a
    window_len = len(seq_b) / len(seq_a)
    break_points = np.arange(0, len(seq_b), window_len)
    for i in range(len(break_points)):
      break_points[i] = int(round(break_points[i]))
    break_points = np.append(break_points, len(seq_b))
    break_points = break_points.astype('int')
    match_count = 0
    for i, unit_A in enumerate(seq_a):
      st = break_points[i]
      ed = break_points[i+1]
      if unit_A in set(seq_b[st:ed]):
        match_count += 1
    return match_count / len(seq_a)

  def _segment_into_seq_list(self, tokens, unit_size):
    seq_list = []
    for (i, token) in enumerate(tokens):
      seq_list.append(self._segment_into_seq(token, unit_size))
    return seq_list
  
  def hamming_sim(self, seq_a, seq_b, unit_size='word'):
    print("here")
    sim_matrix = np.zeros((len(seq_a), len(seq_b)))
    seq_a_list = self._segment_into_seq_list(seq_a, unit_size)
    seq_b_list = self._segment_into_seq_list(seq_b, unit_size)
    print("done")
    for (i, seq_a) in tqdm(enumerate(seq_a_list)):
      for (j, seq_b) in enumerate(seq_b_list):
        sim_matrix[i][j] = self._get_hamming_sim_seq(seq_a, seq_b)
    return sim_matrix

  def get_gloves_multiset(self, token):
      words_list = []
      doc = nlp(token.lower())
      for word in doc:
        if (word.text not in all_stopwords and
         word.text.isalnum() and word.text in glove.stoi
         ):
          idx = glove.stoi[word.text]
          if idx not in top_5:
              top_5[idx] = nn_search.get_nns_by_item(idx, 5)
          words_list.extend(top_5[idx])
      return Multiset(words_list)

  def overlapping_glove_sim(self, seq_a, seq_b):
      sim_matrix = np.zeros((len(seq_a), len(seq_b)))
      #print("Lengths of similarity matrix:" , len(seq_a), len(seq_b))
      top_words_multisets_A = []
      top_words_multisets_B = []
      for (i, token_A) in enumerate(seq_a):
          top_words_multisets_A.append(self.get_gloves_multiset(token_A))
      
      for (i, token_B) in enumerate(seq_b):
          top_words_multisets_B.append(self.get_gloves_multiset(token_B))
      
          
      for (i, token_A) in enumerate(seq_a):
          #print(i)
          for (j, token_B) in enumerate(seq_b):
              sim_matrix[i][j] = self.get_jaccard_similarity(
                top_words_multisets_A[i], top_words_multisets_B[j]
              )
      #sim_matrix -= 0.2
      #sim_matrix *= 2
      return sim_matrix

    