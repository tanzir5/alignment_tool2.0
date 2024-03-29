import seaborn as sns
import sys
import aligners.utility 

import numpy as np
from aligners.preprocessor import Preprocessor
from aligners.smith_waterman import Aligner
import matplotlib.pyplot as plt

def get_removed_segments(alignments, indices, do_seq1):
  aligned_segments = []
  for alignment in alignments:
    if do_seq1:
      aligned_segments.append((alignment['seq1_st'], alignment['seq1_end']))
    else:
      aligned_segments.append((alignment['seq2_st'], alignment['seq2_end']))
  aligned_segments.sort()
  last_aligned_idx = -1
  removed_segments = []
  last_element_idx = len(indices)-1
  for aligned_segment in aligned_segments:
    if aligned_segment[0] != last_aligned_idx+1:
      removed_segments.append((last_aligned_idx+1, aligned_segment[0]-1))
    last_aligned_idx = aligned_segment[1]
  if last_aligned_idx != last_element_idx:
    removed_segments.append((last_aligned_idx+1, last_element_idx))
  return removed_segments  

def get_alignments_with_indices(
  alignments, indices_a, indices_b, seq_a_text, seq_b_text, from_embedding
):
  ret = []
  for alignment in alignments:
    seq1_st = indices_a[alignment['seq1_st']]['st']
    seq1_end = indices_a[alignment['seq1_end']]['end']
    seq2_st = indices_b[alignment['seq2_st']]['st']
    seq2_end = indices_b[alignment['seq2_end']]['end']
    alignment_score = alignment['alignment_score']
    cur_dict = {
      'seq1_st':seq1_st,
      'seq1_end':seq1_end,
      'seq2_st':seq2_st,
      'seq2_end':seq2_end,
      'alignment_score':alignment_score
    }
    if from_embedding == False:
      cur_dict['text_a'] = seq_a_text[seq1_st:seq1_end+1]
      cur_dict['text_b'] = seq_b_text[seq2_st:seq2_end+1]
    ret.append(cur_dict)
  return ret

def get_segments_with_indices(segments, indices):
  ret = []
  for segment in segments:
    ret.append({
      'st':indices[segment[0]]['st'],
      'end':indices[segment[1]]['end']
      })
  return ret

def prepare_all_with_indices(
  alignments, indices_a, indices_b, seq_a_text, seq_b_text, from_embedding=False
):
  removed_segments_seq1 = get_removed_segments(
    alignments, indices_a, do_seq1=True)
  removed_seq1 = get_segments_with_indices(
    removed_segments_seq1, indices_a)
  removed_segments_seq2 = get_removed_segments(
    alignments, indices_b, do_seq1=False)
  removed_seq2 = get_segments_with_indices(
    removed_segments_seq2, indices_b)
  aligned_segments = get_alignments_with_indices(
    alignments, indices_a, indices_b, seq_a_text, seq_b_text, from_embedding)
  return aligned_segments, removed_seq1, removed_seq2

def align_sequences(
  seq1, 
  seq2, 
  unit1='word', 
  unit2='word', 
  sim='exact', 
  z_thresh=2,
  clip=None,
  sim_config=None,
  ignore=None,
  no_gap=True,
  double_break_for_paragraphs=True,
  save_emb_dirs=None,
  gap_start_penalty=-0.4, 
  gap_continue_penalty=-0.1, 
  matching_strategy="ONE_TO_ONE",
  return_preprocessor=False,
  return_aligner=False,
): 
  if sim_config is None:
    sim_config = {}
  if ignore is None:
    ignore = [set(), set()]
  sim_config['func'] = sim
  sim_config['threshold'] = z_thresh
  preprocessor = Preprocessor(
    seq1, 
    seq2, 
    size_a=unit1,
    size_b=unit2,
    sim_config=sim_config,
    clip_length=clip,
    save_emb_dirs=save_emb_dirs,
    no_gap=no_gap,
    double_break_for_paragraphs=double_break_for_paragraphs
  )
  aligner = Aligner(
    preprocessor.sim_matrix, 
    ignore=ignore,
    no_gap=no_gap,
    matching_strategy=matching_strategy,
    gap_start_penalty=gap_start_penalty, 
    gap_continue_penalty=gap_continue_penalty, 
  )
  aligner.compute_smith_waterman()
  alignments, _, _ = aligner.create_alignments()
  ret = {}
  if return_preprocessor:
    ret['preprocessor'] = preprocessor
  if return_aligner:
    ret['aligner'] = aligner
  
  if preprocessor.indices_a is None or preprocessor.indices_b is None:
    ret['alignments'] = alignments
    return ret  
  
  aligned_segments, removed_seq1, removed_seq2 = prepare_all_with_indices(
    alignments, 
    preprocessor.indices_a, 
    preprocessor.indices_b, 
    preprocessor.unmodified_seq_a, 
    preprocessor.unmodified_seq_b,
    from_embedding=(unit1 in ['embedding', 'embedding_path'])
  )
  ret['alignments'] = aligned_segments
  ret['removed_seq1'] = removed_seq1
  ret['removed_seq2'] = removed_seq2
  
  return ret
