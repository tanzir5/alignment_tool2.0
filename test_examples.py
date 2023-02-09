from aligners.align_pipeline import align_sequences 
import numpy as np
from aligners.smith_waterman import Aligner
import os

def test_aligner():
  # testing alignment from similarity matrix
  r = 10
  c = 12
  sim_matrix = np.random.rand(r, c)
  sim_matrix[2][11] = 200
  sim_matrix[3][5] = 5
  sim_matrix[4][6] = 4
  aligner = Aligner(sim_matrix)
  aligned_segments, s1_to_s2_alignment, s2_to_s1_alignment = aligner.create_alignments()
  for seg_pair in aligned_segments:
    print(seg_pair)

def test_align_pipeline_embedding():
  #testing alignment from embedding and thresholds 
  seq1 = np.random.rand(10, 768) * 0.2
  seq2 = np.random.rand(12, 768) * 0.2
  ret = align_sequences(
    seq1, 
    seq2, 
    unit1='embedding', 
    unit2='embedding', 
    sim='sbert',
  )
  alignments = ret['alignments'] 
  for seg_pair in alignments:
    print(seg_pair)

def test_align_pipeline_text(unit):
  if unit == 'paragraph':
    if (os.path.exists('./misc/test_paragraphs1.txt') is False or 
        os.path.exists('./misc/test_paragraphs2.txt') is False
      ):
        print("cannot test for paragraphs since './misc/test_paragraphs1.txt'",
          "or './misc/test_paragraphs1.txt' do not exist. Run this code from",
          "the alignments2.0 directory.")
        return 
    
    with open('./misc/test_paragraphs1.txt') as f:
      text1 = f.read()
    with open('./misc/test_paragraphs2.txt') as f:
      text2 = f.read()
  else:
    text1 = "I eat rice, burger, pizza. Rome is in Italy."
    text2 = "Rome is in Italy. I eat rice, burger, pizza. They are in Rome."
  ret = align_sequences(
    text1, 
    text2, 
    unit1=unit, 
    unit2=unit, 
    sim='exact' if unit == 'word' else 'sbert',
    z_thresh=9,
  )
  alignments = ret['alignments'] 
  for seg_pair in alignments:
    print(seg_pair)

print("*"*100)
print("testing aligner with predefined similarity matrix")
test_aligner()
print("*"*100)
print("testing align pipeline with sequence of embeddings")
test_align_pipeline_embedding()
print("*"*100)
print("testing align pipeline with text and unit of alignment as word")
test_align_pipeline_text('word')
print("*"*100)
print("testing align pipeline with text and unit of alignment as sentence")
test_align_pipeline_text('sentence')
print("*"*100)
print("testing align pipeline with text and unit of alignment as paragraph")
test_align_pipeline_text('paragraph')
