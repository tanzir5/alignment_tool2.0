import pandas as pd
import os
import glob
import numpy as np
from sklearn.metrics import cohen_kappa_score

def check_columns_validity():
  cols = None
  for fname in glob.glob('data/aesop_annotations/annotated_data/*'):
    df = pd.read_csv(fname)
    if cols is None:
      cols = df.columns
    if not (cols==df.columns).all():
      print("YOLO")
      print(cols)
      print(df.columns)
      assert(False)



def fix_chatgpt_alignments():
  bad_count = 0
  for fname in glob.glob('data/aesop_annotations/annotated_data/*'):
    df = pd.read_csv(fname)
    c_alignments_str = df['chat_gpt_alignments'][0]
    c_alignments_tuples = []
    if isinstance(c_alignments_str, str) is False:
      print(fname)
      bad_count += 1
    else: 
      print(fname)
      for line in c_alignments_str.split('\n'):
        if (line == ""):
          continue
        line = line.strip('()')
        a,b = line.split(',')
        c_alignments_tuples.append((a,b))
      print(c_alignments_tuples)


def get_chatgpt_alignments(df):
  df = pd.read_csv(fname)
  c_alignments_str = df['chat_gpt_alignments'][0]
  if isinstance(c_alignments_str, str) is False:
    return None
  else: 
    c_alignments_tuples = []
    for line in c_alignments_str.split('\n'):
      if (line == ""):
        continue
      line = line.strip('()')
      a,b = line.split(',')
      c_alignments_tuples.append((int(a),int(b)))
    return c_alignments_tuples

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
  print("total", len(final_predictions_list))
  return cohen_kappa_score(final_predictions_list, final_gold_list)

def get_len(sents):
  for i in range(len(sents)-1, -1, -1):
    if sents[i] is np.nan:
      sents.pop()
    else:
      break
  return len(sents)    

tp, fp, fn = (0, 0, 0)
b_tp, b_fp, b_fn = (0, 0, 0)  
c_alignments_dict = {}
m_alignments_dict = {}
baseline_alignments_dict = {}

train_set = []
f = open('data/aesop_annotations/train_set_list.txt', 'r')
train_set = [x[:-1] for x in f.readlines()]

for fname in glob.glob('data/aesop_annotations/annotated_data/*'):
  if fname in train_set:
    continue
  df = pd.read_csv(fname)
  c_alignments = get_chatgpt_alignments(df)
  m_alignments = get_manual_annotations(df)
  len1 = get_len(df['sent1'].tolist())
  len2 = get_len(df['sent2'].tolist())
  b_alignments = [(i, int(len2/(len1)*i)) for i in range(len(m_alignments))]
  csv_name = fname.split('/')[-1]
  if c_alignments is None or m_alignments is None:
    continue
  c_alignments_dict[csv_name] = c_alignments
  m_alignments_dict[csv_name] = m_alignments
  baseline_alignments_dict[csv_name] = b_alignments
  temp_tp, temp_fp, temp_fn = get_confusion_matrix(m_alignments, c_alignments)
  tp += temp_tp
  fp += temp_fp
  fn += temp_fn

  temp_tp, temp_fp, temp_fn = get_confusion_matrix(m_alignments, b_alignments)
  b_tp += temp_tp
  b_fp += temp_fp
  b_fn += temp_fn


print("Chatgpt results:")
print(tp, fp, fn)
print("precision, recall, f1", get_precision_recall_f1(tp, fp, fn))
print("kappa", get_cohens_kappa(m_alignments_dict, c_alignments_dict))

print("#"*100)
print("monkey results")
print(b_tp, b_fp, b_fn)
print("precision, recall, f1", get_precision_recall_f1(b_tp, b_fp, b_fn))
print("kappa", get_cohens_kappa(m_alignments_dict, baseline_alignments_dict))
