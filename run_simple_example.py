from aligners.align_pipeline import align_sequences 

text1 = "I eat rice, burger, pizza. Rome is in Italy."
text2 = "Rome is in Italy. They are in Rome. I eat rice, burger, pizza."
ret = align_sequences(text1, text2)
alignments = ret['alignments'] 
for seg_pair in alignments:
  print(seg_pair)
