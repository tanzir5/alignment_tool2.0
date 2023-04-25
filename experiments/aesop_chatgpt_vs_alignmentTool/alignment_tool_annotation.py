import pandas as pd
import os
import glob
import numpy as np
from sklearn.metrics import cohen_kappa_score
from aligners.align_pipeline import align_sequences
import multiprocessing as mp

from tqdm import tqdm

def get_manual_annotations(df):
  m_alignments = df['manual_alignments'].tolist()
  fixed_alignments_tuples = []
  for x in m_alignments:
    if x is np.nan:
      continue
    x = x.strip('()')
    if len(x.split(',')) == 2:
      a, b = x.split(',')
    else:
      a, b = x.split()

    a, b = int(a), int(b)
    assert(a != -1)
    fixed_alignments_tuples.append((a, b))
  return fixed_alignments_tuples

def get_precision_recall_f1(tp, fp, fn):
  precision = tp / (tp+fp)
  recall = tp / (tp+fn)
  f1 = 2 * precision * recall / (precision+recall)
  return precision, recall, f1

def get_confusion_matrix(gold_data, predicted_data):
  tp, fp, tn, fn = (0, 0, 0, 0)
  for x in predicted_data:
    if x in gold_data:
      tp += 1
    else:
      fp += 1
  for x in gold_data:
    if x[1] != -1 and x not in predicted_data:
      fn += 1
  return tp, fp, fn

def complete_prediction_alignments(prediction_alignment, length):
  temp_dict = {}
  for a, b in prediction_alignment:
    temp_dict[a] = b
  ret_prediction = []
  for j in range(length):
    if j not in temp_dict:
      ret_prediction.append((j, -1))
    else:
      ret_prediction.append((j, temp_dict[j]))
  return ret_prediction



def get_cohens_kappa(gold_dict, predictions_dict):
  final_predictions_list = []
  final_gold_list = []
  bad_count = 0
  for name, gold_alignments in gold_dict.items():
    if name not in predictions_dict:
      continue
    final_gold_list.extend([y for x,y in gold_alignments])
    length = len(gold_alignments)
    prediction_alignment = predictions_dict[name]
    prediction_alignment = complete_prediction_alignments(
                        prediction_alignment, length)
    final_predictions_list.extend([y for x,y in prediction_alignment])
    #print(name)
    #print("*"*100)
    for i in range(len(gold_alignments)):
      if (gold_alignments[i][0] != prediction_alignment[i][0]):
        print("*"*100)
        print(name)
        bad_count += 1
        break
  assert(bad_count == 0)
  # print("total", len(final_predictions_list))
  return cohen_kappa_score(final_predictions_list, final_gold_list)


def fix_format(alignment):
  ret = []
  for i, j in enumerate(alignment):
    ret.append((i, j))
  return ret

def compute_metrics(data, sim, z, strategy, gap_start_penalty, gap_continue_penalty):
  '''
  if strategy == "ONE_TO_ONE":
    gap_start_penalty = -0.4
    gap_continue_penalty = -0.2
  elif strategy == "MANY_TO_MANY":
    gap_start_penalty = -0.2
    gap_continue_penalty = -0.1
  '''
  tp, fp, fn = 0, 0, 0
  m_alignments_dict = {}
  predicted_alignments_dict = {}
  for fname in data:
    df = pd.read_csv(fname)
    sent_seq_1 = df['sent1'].tolist()
    sent_seq_2 = df['sent2'].tolist()
    m_alignments = get_manual_annotations(df)
    csv_name = fname.split('/')[-1]
    m_alignments_dict[csv_name] = m_alignments
    for i in range(len(sent_seq_1)-1, -1, -1):
      if sent_seq_1[i] is np.nan:
        sent_seq_1.pop()
      else:
        break
    
    for i in range(len(sent_seq_2)-1, -1, -1):
      if sent_seq_2[i] is np.nan:
        sent_seq_2.pop()
      else:
        break    
    ret = align_sequences(
      sent_seq_1, 
      sent_seq_2, 
      unit1='sentence', 
      unit2='sentence', 
      sim=sim, 
      z_thresh=z,
      no_gap=False,
      matching_strategy=strategy,
      return_aligner=True,
      gap_start_penalty=gap_start_penalty,
      gap_continue_penalty=gap_continue_penalty
    ) 
    #print(ret['aligner'].s1_to_s2_align)
    predicted_alignment = fix_format(ret['aligner'].s1_to_s2_align)
    predicted_alignments_dict[csv_name] = predicted_alignment
    temp_tp, temp_fp, temp_fn = get_confusion_matrix(
                                m_alignments, predicted_alignment)
    tp += temp_tp
    fp += temp_fp
    fn += temp_fn
    #input()
  
  precision, recall, f1 = get_precision_recall_f1(tp, fp, fn)
  kappa = get_cohens_kappa(m_alignments_dict, predicted_alignments_dict)
  new_df = pd.DataFrame({
    'sim':[sim], 'strategy':[strategy], 'sim':[sim], 'z':[z], 
    'g1':[gap_start_penalty], 'g2':[gap_continue_penalty],
    'precision': [precision], 'recall':[recall], 'f1':[f1], 'kappa':[kappa]
    })
  write_path = 'data/aesop_annotations/train_set_results_mega.csv'
  new_df.to_csv(write_path, mode='a', header=not os.path.exists(write_path))
  #return precision, recall, f1, kappa


training_set, test_set, full_set = [], [], []
for fname in glob.glob('data/aesop_annotations/annotated_data/*'):
  full_set.append(fname)


if __name__ == '__main__':
  train_set_list_path = 'data/aesop_annotations/train_set_list.txt'
  if os.path.exists(train_set_list_path) is False:
    full_set = np.random.permutation(full_set)
    train_last_idx = 30
    training_set = full_set[:train_last_idx]
    test_set = full_set[train_last_idx:]
    train_set_list_file = open(train_set_list_path, 'w')
    for name in training_set:
      train_set_list_file.write(name+"\n")
    train_set_list_file.close()
  else:
    with open(train_set_list_path) as f:
      training_set = [line[:-1] for line in f.readlines()]
    test_set = [x for x in full_set if x not in training_set]

  pool = mp.Pool(mp.cpu_count())
  jobs = []  

  for sim in ['jaccard', 'sbert']:
    for strategy in ["ONE_TO_ONE", "MANY_TO_MANY"]:
      if sim == 'sbert':
        st, ed = 0, 6
      else:
        st, ed = 0, 21
      for z in range(st, ed):
        for g1 in range(0,10):
          g1_real = g1/10
          for g2 in range(0, g1+1):
            g2_real = g2/10
            compute_metrics(training_set, sim, z, strategy, g1_real, g2_real)
            job = pool.apply_async(
              compute_metrics, (training_set, sim, z, strategy, g1_real, g2_real))
            jobs.append(job)
            
  for i, job in tqdm(enumerate(jobs)): 
    job.get()
    
  pool.close()
  pool.join()
