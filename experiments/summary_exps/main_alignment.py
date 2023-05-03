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
  ret_indices = []
  iterations = 0
  while len(ret_indices) < k:
    if iterations > 10000:
      break
    idx = np.random.randint(0, len(texts))
    if idx not in ret_indices and texts[idx][0] != illegal_name:
      if mn_length <= len(texts[idx][1]) <= mx_length:
        ret_indices.append(idx)
  return ret_indices

def align_single_job(seq1, seq2, summary_name, book_name, unit_size, real):
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


df_write_path = 'data/summary_exps/score.csv'

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

if __name__ == "__main__":
  pool = mp.Pool(mp.cpu_count())
  jobs = []  
  for name in all_names:
    fname = 'data/summary_exps/final_jsons/'+name+'.json'
    with open(fname) as f:
      data_dict = json.load(f)
    my_text = {
      'book_sent':data_dict['book_sent'][0],
      'book_para':data_dict['book_para'][0], 
      'book_chunk':data_dict['book_chunk'], 
    }
    seq1 = data_dict['summary_sent'][0]
    for unit_size in ['book_chunk']:
      seq2 = my_text[unit_size]  
      other_name = name
      job = pool.apply_async(
              align_single_job, 
              (seq1, seq2, name, other_name, unit_size, 1)
            )
      jobs.append(job)

      length = len(my_text[unit_size])
      unrelated_indices = get_random_indices_similar_len(all_texts[unit_size], length, 99, name)
      for idx in unrelated_indices:
        seq2 = all_texts[unit_size][idx][1]  
        other_name = all_texts[unit_size][idx][0]
        job = pool.apply_async(
                align_single_job, 
                (seq1, seq2, name, other_name, unit_size, 0)
              )
        jobs.append(job)

  print("length of jobs", len(jobs))

  for i, job in tqdm(enumerate(jobs)): 
    job.get()
      
