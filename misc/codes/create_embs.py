from aligners.preprocessor import Preprocessor
import pandas as pd
import os 
import glob 
from tqdm import tqdm

csv_path_root = 'data/positive_pairs/text_csv/'
emb_path_root = 'data/positive_pairs/embs/'
for fname in tqdm(glob.glob(csv_path_root+'*.csv')):
  df = pd.read_csv(fname)
  paragraphs = df['para'].tolist()
  current_book_id = fname[:-4].split('/')[-1]
  current_emb_path = emb_path_root + "/" + current_book_id + ".npy"
  temp_preprocessor = Preprocessor(
    paragraphs, 
    sim_config={'func':'sbert'}, 
    save_emb_dirs=current_emb_path,
    create_embs_only=True
  )

exit(0)