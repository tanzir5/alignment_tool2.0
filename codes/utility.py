import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import spacy

nlp = None

def load_spacy_nlp(name='en_core_web_sm'):
  global nlp
  nlp = spacy.load(name)

def get_paragraphs(text):
  ret = re.split('\n\n\s*', text)
  ret = [x for x in ret if len(x.split()) > 0]
  return ret

def get_sentences(text, add_start_position=False):
  if nlp is None:
    load_spacy_nlp()
  doc = nlp(text)
  sentences = []
  for sent in doc.sents:
    sentences.append(sent.text)
  if not add_start_position:
    return sentences
  else:
    sentences_with_st_pos = []
    current_index = 0
    for sent in sentences:
      sentences_with_st_pos.append((sent, current_index))
      current_index += len(sent)

    return sentences_with_st_pos

def parse_text(file_path, unit):
  f = open(file_path)
  text = f.read()
  f.close()
  if unit == 'paragraph':
    paras = get_paragraphs(text)
    return paras 
  else:
    raise ValueError('only paragraphs as unit size are supported at this moment')

def print_matches(matches, parsed_text1, parsed_text2):
  for m in matches:
    print("*" * 35 + "From Book 1 " + "*" * 35)
    for p in parsed_text1[m[0][0]: m[1][0]+1]:
      print(p)
    print("")
    print("*" * 35 + "From Book 2 " + "*" * 35)
    for p in parsed_text2[m[0][1]: m[1][1]+1]:
      print(p)
    print("")
    print("-" * 35 + "END" + "-"*35 + "\n")
    print("")

def convert_4d_to_2d(dp):
  ret = np.zeros((dp.shape[0], dp.shape[1]))
  for i in range(ret.shape[0]):
      for j in range(ret.shape[1]):
          ret[i][j] = dp[i][j][0][0]
  return ret

def show_heat_map(dp, do_log = False):
    #waterman_matrix = np.clip(fix_dp(dp), a_max = 0.1, a_min = 0)
    waterman_matrix = convert_4d_to_2d(dp)
    if do_log:
        waterman_matrix = np.log10(waterman_matrix+1e-9)
    xticklabels = int(waterman_matrix.shape[0] / 20)
    yticklabels = int(waterman_matrix.shape[1] / 20)
    sns.heatmap(waterman_matrix, xticklabels = xticklabels, yticklabels = yticklabels, cmap = sns.cm.rocket_r)
    plt.show()

def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))
