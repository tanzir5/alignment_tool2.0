import json
import pandas as pd
import os
import re
import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np
from tqdm import tqdm
nlp = spacy.load("en_core_web_sm", disable=['tagger', 'ner'])
nlp.max_length = 2198623

def _segment_into_sentence(text):
    doc = nlp(text)
    sentences = []
    indices = []
    for sent in doc.sents:
      sentences.append(sent.text)
      indices.append({'st':sent.start_char, 'end':sent.end_char-1})
    return sentences, indices

def _segment_into_paragraph(text, double_blank=True):
  if double_blank == True:
    pattern = '\n\s*\n\s*'
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


def chunk_func(text_list, chunk_count):
  chunk_size = int(len(text_list) / chunk_count)
  chunks = []
  for i in range(0, len(text_list), chunk_size):
    chunks.append("\n".join(text_list[i:i+chunk_size]))
  return chunks

data_path = 'data/summary_exps/'

df = pd.read_csv(data_path + "gutenberg_and_masterplots_matches.csv")


book_to_summary_word_ratio_sum = 0
summary_word_sum = 0
book_word_sum = 0
good_count = 0
for i in tqdm(range(len(df))):
  guten_id = df.iloc[i]['guten_id']
  summary_name = df.iloc[i]['masterplots']
  with open(data_path+'summary_relevant_guten_files/'+str(guten_id)+'.json') as f:
    json_obj = json.load(f)
    if len(json_obj['text']) != 0:
      book_text = json_obj['text']
    else:
      book_text = None
  fname1 = data_path+'summaries/'+summary_name+'/The Story:.txt'
  fname2 = data_path+'summaries/'+summary_name+'/The Poem:.txt'
  assert(os.path.exists(fname1) == False or os.path.exists(fname2) == False)
  if os.path.exists(fname1) == False and os.path.exists(fname2) == False:
    summary_text = None
  else:
    if os.path.exists(fname1) == True:
      fname = fname1
    else:
      fname = fname2
      #print(guten_id, summary_name)
    with open(fname) as f:
      summary_text = f.read()
  
  if summary_text is not None and book_text is not None:
    if len(book_text) > nlp.max_length:
      continue
    write_path = data_path + 'final_jsons/' + summary_name + '.json'
    if os.path.exists(write_path):
      continue
    good_count += 1
    
    summary_sentences_data = _segment_into_sentence(summary_text)
    book_sentences_data = _segment_into_sentence(book_text)
    book_paragraphs_data = _segment_into_paragraph(book_text)
    target_chunk = None
    print(len(book_paragraphs_data[0]), len(summary_sentences_data[0]), len(book_sentences_data[0]))
    if len(book_paragraphs_data[0]) < len(summary_sentences_data[0]):
      target_chunk = book_sentences_data
      print("sentences")
    else:
      target_chunk = book_paragraphs_data
      print("paragraphs")
    if len(target_chunk[0]) < len(summary_sentences_data[0]):
      continue
    chunks = chunk_func(target_chunk[0], len(summary_sentences_data[0]))
    print(len(chunks))
    data_dict = {'summary_sent': summary_sentences_data, 
                 'book_sent': book_sentences_data, 
                 'book_para': book_paragraphs_data,
                 'book_chunk': chunks}
    
    with open(write_path, 'w') as f:
      json.dump(data_dict, f)

book_to_summary_word_ratio_avg = book_to_summary_word_ratio_sum / good_count
print(book_to_summary_word_ratio_avg)
print(book_word_sum / good_count)
print(summary_word_sum / good_count)

