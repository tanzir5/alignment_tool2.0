from aligners.preprocessor import Preprocessor
import glob
import json


for fname in glob.glob('data/summary_exps/final_jsons/*.json'):
  with open(fname) as f:
    data_dict = json.load(f)
  for key in data_dict:
    if key == 'book_sent':
      continue
    if key == 'book_chunk':
      text_seq = data_dict[key]
    else:
      text_seq = data_dict[key][0]
    name = fname.split('/')[-1]
    write_path = 'data/summary_exps/embs/'+key+"/"+name 
    pre = Preprocessor(
        text_seq, 
        sim_config={'func':'sbert'}, 
        save_emb_dirs=write_path,
        create_embs_only=True
      )
