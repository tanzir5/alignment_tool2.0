from aligners.align_pipeline import align_sequences 
import numpy as np
from aligners.smith_waterman import Aligner

text1 = "I eat rice, burger, pizza. Rome is in Italy."
text2 = "Rome is in Italy. I eat rice, burger, pizza. They are in Rome."
ret = align_sequences(text1, text2)
alignments = ret['alignments'] 
for seg_pair in alignments:
  print(seg_pair)