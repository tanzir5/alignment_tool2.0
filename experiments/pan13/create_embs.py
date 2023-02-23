from aligners.preprocessor import Preprocessor
import os
import glob
from tqdm import tqdm

def create_embs_for_all(input_dir, unit, save_dir):
  for fname in tqdm(glob.glob(input_dir+"*.txt")):
    text = open(fname).read()
    filename = fname.split("/")[-1].split('.')[0]
    save_path = save_dir + filename + ".json"
    
    preprocessor = Preprocessor(
      text,
      size_a=unit,
      sim_config={'func':'sbert'},
      save_emb_dirs=[save_path],
      double_break_for_paragraphs=False,
      create_embs_only=True,
    )


input_dir = "data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/src/"
save_dir = "data/experiments/pan13/embeddings/src/sentences/"
create_embs_for_all(input_dir, unit='sentence', save_dir=save_dir)

'''
input_dir = "data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/src/"
save_dir = "data/experiments/pan13/embeddings/src/paragraphs/"
create_embs_for_all(input_dir, unit='paragraph', save_dir=save_dir)

input_dir = "data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/susp/"
save_dir = "data/experiments/pan13/embeddings/susp/sentences/"
create_embs_for_all(input_dir, unit='sentence', save_dir=save_dir)


input_dir = "data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/susp/"
save_dir = "data/experiments/pan13/embeddings/susp/paragraphs/"
create_embs_for_all(input_dir, unit='paragraph', save_dir=save_dir)
'''

#path = "data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/susp/"
#create_embs_for_all(path)