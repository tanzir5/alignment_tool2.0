import numpy as np
from aligners.align_pipeline import align_sequences

with open('data/experiments/correlation/20781_Heidi.txt') as f:
  seq1 = f.read()

with open('data/experiments/correlation/46409_Heidi.txt') as f:
  seq2 = f.read()

ret = align_sequences(
  seq1, 
  seq2, 
  unit1='paragraph', 
  unit2='paragraph', 
  sim='sbert', 
  z_thresh=2,
  no_gap=False,
  double_break_for_paragraphs=True,
  save_emb_dirs=None,
  gap_start_penalty=-0.4, 
  gap_continue_penalty=-0.1, 
  matching_strategy="ONE_TO_ONE",
  return_preprocessor=False,
  return_aligner=True,
)

np.save('data/experiments/correlation/heidi_dp.npy', ret['aligner'].dp)
