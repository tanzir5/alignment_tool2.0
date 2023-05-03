from aligners.align_pipeline import align_sequences
import glob
import json
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

def get_random_indices_similar_len(texts, length, k, illegal_name):
  mn_length = length * .7
  mx_length = length * 1.3
  candidates = []
  for i, text in enumerate(texts):
    if mn_length <= len(text[1]) <= mx_length:
      candidates.append(i)
  if len(candidates) <= k:
    return candidates
  else:
    return np.random.choice(candidates, size=k, replace=False)
  
def align_single_job(seq1, seq2, summary_name, book_name, unit_size, real, sim):
  if sim == 'jaccard':
    df_write_path = 'data/summary_exps/jaccard_score.csv'
    ret = align_sequences(
      seq1=seq1,
      seq2=seq2,
      unit1='sentence',
      unit2='paragraph',
      sim='jaccard',
      z_thresh=4)
    if len(ret['alignments']) == 0:
      score = 0
    else:
      score = ret['alignments'][0]['alignment_score']
    df = pd.DataFrame({'summary':[summary_name], 'book': [book_name], 'unit': [unit_size], 'real':[real], 'score':[score]})
    df.to_csv(df_write_path, mode='a', header=os.path.exists(df_write_path)==False)
  elif SIM == 'sbert':
    df_write_path = 'data/summary_exps/sbert_score.csv'
    ret = align_sequences(
      seq1=seq1,
      seq2=seq2,
      unit1='embedding',
      unit2='embedding',
      sim='sbert',
      z_thresh=2)
    if len(ret['alignments']) == 0:
      score = 0
    else:
      score = ret['alignments'][0]['alignment_score']
    df = pd.DataFrame({'summary':[summary_name], 'book': [book_name], 'unit': [unit_size], 'real':[real], 'score':[score]})
    df.to_csv(df_write_path, mode='a', header=os.path.exists(df_write_path)==False)


all_names = []

all_texts = {'book_sent':[], 'book_para':[], 'book_chunk':[]}

for fname in glob.glob('data/summary_exps/final_jsons/*.json'):
  name = fname.split('/')[-1][:-5]
  all_names.append(name)
  with open(fname) as f:
    data_dict = json.load(f)
    #print(data_dict.keys())
    all_texts['book_sent'].append((name, data_dict['book_sent'][0]))
    all_texts['book_para'].append((name, data_dict['book_para'][0]))
    all_texts['book_chunk'].append((name, data_dict['book_chunk']))

all_texts['book_sent'].sort(key=lambda x: len(x[1]), reverse=True)
all_texts['book_para'].sort(key=lambda x: len(x[1]), reverse=True)
all_texts['book_chunk'].sort(key=lambda x: len(x[1]), reverse=True)



SIM = 'sbert'
#SIM = 'jaccard'

if __name__ == "__main__":
  if SIM == 'jaccard':
    pool = mp.Pool(mp.cpu_count())
    
    for name in tqdm(all_names):
      jobs = []  
      fname = 'data/summary_exps/final_jsons/'+name+'.json'
      with open(fname) as f:
        data_dict = json.load(f)
      my_text = {
        'book_sent':data_dict['book_sent'][0],
        'book_para':data_dict['book_para'][0], 
        'book_chunk':data_dict['book_chunk'], 
      }
      seq1 = data_dict['summary_sent'][0]
      for unit_size in ['book_chunk', 'book_para']:
        length = len(my_text[unit_size])
        unrelated_indices = get_random_indices_similar_len(all_texts[unit_size], length, 99, name)
        seq2 = my_text[unit_size]  
        other_name = name
        job = pool.apply_async(
                align_single_job, 
                (seq1, seq2, name, other_name, unit_size, 1, SIM)
              )
        jobs.append(job)

        for idx in unrelated_indices:
          seq2 = all_texts[unit_size][idx][1]  
          other_name = all_texts[unit_size][idx][0]
          job = pool.apply_async(
                  align_single_job, 
                  (seq1, seq2, name, other_name, unit_size, 0, SIM)
                )
          jobs.append(job)
      for i, job in enumerate(jobs): 
        job.get()
  elif SIM == 'sbert':
    pool = mp.Pool(mp.cpu_count())
    
    for name in tqdm(all_names):
      jobs = []  
      fname = 'data/summary_exps/embs/summary_sent/'+name+'.json'
      seq1 = fname
      for unit_size in ['book_chunk', 'book_para']:
        fname = 'data/summary_exps/embs/'+unit_size+'/'+name+'.json'
        seq2 = fname  
        with open(seq2) as f:
          json_obj = json.load(f)
          length = len(json_obj['embedding'])
        unrelated_indices = get_random_indices_similar_len(all_texts[unit_size], length, 99, name)
        other_name = name
        job = pool.apply_async(
                align_single_job, 
                (seq1, seq2, name, other_name, unit_size, 1, SIM)
              )
        jobs.append(job)

        for idx in unrelated_indices:
          other_name = all_texts[unit_size][idx][0]
          fname = 'data/summary_exps/embs/'+unit_size+'/'+other_name+'.json'
          seq2 = fname  
          job = pool.apply_async(
                  align_single_job, 
                  (seq1, seq2, name, other_name, unit_size, 0, SIM)
                )
          jobs.append(job)
      for i, job in enumerate(jobs): 
        job.get()

