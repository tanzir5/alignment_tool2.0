#TODO: Fix the alignment score when clipped_for_overlap becomes true
import numpy as np

INF = 1e9
ONE_TO_ONE = "ONE_TO_ONE"
ONE_TO_MANY = "ONE_TO_MANY"
MANY_TO_ONE = "MANY_TO_ONE"
MANY_TO_MANY = "MANY_TO_MANY"

class Aligner:
  def __init__(self, 
    similarity_matrix, 
    ignore=None,
    no_gap=True, 
    gap_start_penalty=-0.9, 
    gap_continue_penalty=-0.5, 
    use_global_prior=False,
    matching_strategy=ONE_TO_ONE,
  ):
    if ignore is None:
      ignore = [set(), set()]
    self.similarity_matrix = similarity_matrix
    self.ignore = ignore
    self.no_gap = no_gap
    self.gap_start_penalty = gap_start_penalty
    self.gap_continue_penalty = gap_continue_penalty
    self.use_global_prior = use_global_prior
    self.sw_computed = False
    self.matching_strategy = matching_strategy

    if self.no_gap:
      self.gap_start_penalty = -INF
      self.gap_continue_penalty = -INF
  
  def global_prior(self, b, m):
    mu_b = math.ceil(b * self.N_M/self.N_B)
    exponent = (m - mu_b) * (m - mu_b) / (2*self.N_M*self.N_M)
    exponent *= -1
    return math.exp(exponent)*0.0125

  def compute_smith_waterman(self):
    self.dp = np.zeros(
      (self.similarity_matrix.shape[0]+1, 
       self.similarity_matrix.shape[1]+1, 2, 2)
    )
    if self.use_global_prior:
      self.N_M = self.dp.shape[0]
      self.N_B = self.dp.shape[1]
    self.parent = {}
    n = self.similarity_matrix.shape[0]
    m = self.similarity_matrix.shape[1]
    
    #Align n text units of first book against m text units of second book 
    #gc => gap_continue
    #similarity matrix uses 0 based indexing, dp(alone) uses 1 based indexing
    #print(n,m)
    for i in range(n+1):
      for j in range(m+1):
        for gc_1 in range(2):
          for gc_2 in range(2):
            #base case
            if i == 0 or j == 0:
              self.dp[i][j][gc_1][gc_2] = 0
              self.parent[(i, j, gc_1, gc_2)] = (-1, -1, -1, -1)
              continue
            
            #Recurrence
            best = 0
            self.parent[(i, j, gc_1, gc_2)] = (-1, -1, -1, -1)
            #handle ignore cases first
            if i in self.ignore[0]:
              parent_value = self.dp[i-1][j][gc_1][gc_2]
              self.dp[i][j][gc_1][gc_2] = parent_value
              self.parent[(i, j, gc_1, gc_2)] = (i-1, j, gc_1, gc_2)
            elif j in self.ignore[1]:
              parent_value = self.dp[i][j-1][gc_1][gc_2]
              self.dp[i][j][gc_1][gc_2] = parent_value
              self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, gc_2)
            else:
              # first try gaps
              if gc_1 == 1:
                #continue gap for first text
                current_value = (self.dp[i-1][j][gc_1][gc_2] + 
                                self.gap_continue_penalty)
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i-1, j, gc_1, gc_2)
              else:
                #start new gap for first text
                current_value = (self.dp[i-1][j][1][gc_2] + 
                                 self.gap_start_penalty)
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i-1, j, 1, gc_2)
              if gc_2 == 1:
                #continue gap for second text
                current_value = (self.dp[i][j-1][gc_1][gc_2] + 
                                 self.gap_continue_penalty)
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, gc_2)
              else:
                #start new gap for second text
                current_value = (self.dp[i][j-1][gc_1][1] + 
                                 self.gap_start_penalty)
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i, j-1, gc_1, 1)
              if self.use_global_prior:
                # 1 <= gb <= 2 always true
                gb = self.global_prior(j, i) # if positive sim, higher is better
              else:
                gb = 0
              # try matching
              current_value = (self.dp[i-1][j-1][0][0] + 
                              self.similarity_matrix[i-1][j-1] + gb) 
              if best < current_value:
                best = current_value
                self.parent[(i, j, gc_1, gc_2)] = (i-1, j-1, 0, 0)
          
              if self.matching_strategy in [ONE_TO_MANY, MANY_TO_MANY]:
                # try matching the first one with multiple units from the second
                # sequence

                current_value = (self.dp[i][j-1][1][0] + 
                                 self.similarity_matrix[i-1][j-1] + gb +
                                 self.gap_start_penalty) 
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i, j-1, 1, 0)

              if self.matching_strategy in [MANY_TO_ONE, MANY_TO_MANY]:
                # try matching the second one with multiple units from the first
                # sequence

                current_value = (self.dp[i-1][j][0][1] + 
                                 self.similarity_matrix[i-1][j-1] + gb +
                                 self.gap_start_penalty) 
                if best < current_value:
                  best = current_value
                  self.parent[(i, j, gc_1, gc_2)] = (i-1, j, 0, 1)

              self.dp[i][j][gc_1][gc_2] = best
              

  def set_global_align_variables(self):
    n = self.dp.shape[0]
    m = self.dp.shape[1]
    if n > m:
      #print("ERRROR!\nThe first one should be the smaller sequence.")
      pass

    self.dp_2d = np.zeros((n,m))
    self.is_masked_rows = np.zeros(n)
    self.is_masked_cols = np.zeros(m)
    self.s1_to_s2_align = np.full(n, -1)
    self.s2_to_s1_align = np.full(m, -1)
    self.all_cells = []
    for i in range(n):
      for j in range(m):
        self.dp_2d[i][j] = self.dp[i][j][0][0]
        self.all_cells.append({'dp_val': self.dp_2d[i][j], 'i':i, 'j':j})
    self.all_cells.sort(key=lambda x: x['dp_val'], reverse=True)

  def traverse(self, i, j, match_element_pair_threshold=0):
    gc_1 = 0
    gc_2 = 0
    seq1_end = i
    seq2_end = j
    clipped_for_overlap = False
    alignment_score = self.dp_2d[i][j]
    while i > 0 and j > 0:
      if self.dp[i][j][gc_1][gc_2] == 0:
        break
      if self.is_masked_rows[i] == 1 or self.is_masked_cols[j] == 1:
        clipped_for_overlap = True
        break
      current_parent = self.parent[(i, j, gc_1, gc_2)]
      if current_parent[0] != i:  
        self.is_masked_rows[i] = 1
      if current_parent[1] != j:
        self.is_masked_cols[j] = 1

      if (
          (current_parent[0] == i-1 and current_parent[1] == j-1) and
          (self.similarity_matrix[i-1][j-1] > match_element_pair_threshold)
         ):
        #print(self.similarity_matrix[i-1][j-1])
        self.s1_to_s2_align[i] = j
        self.s2_to_s1_align[j] = i
      
      if ((self.matching_strategy in [ONE_TO_MANY, MANY_TO_MANY]) and 
          (current_parent == (i, j-1, 1, 0)) and
          (self.similarity_matrix[i-1][j-1] > match_element_pair_threshold)
        ):
        self.s2_to_s1_align[j] = i
        if self.s1_to_s2_align[i] == -1:
          self.s1_to_s2_align[i] = j
      if ((self.matching_strategy in [MANY_TO_ONE, MANY_TO_MANY]) and 
          (current_parent == (i-1, j, 0, 1)) and
          (self.similarity_matrix[i-1][j-1] > match_element_pair_threshold)
        ):
        self.s1_to_s2_align[i] = j 
        if self.s2_to_s1_align[j] == -1:
          self.s2_to_s1_align[j] = i
      i, j, gc_1, gc_2 = current_parent
    
    seq1_st = i+1
    seq2_st = j+1
    if (seq1_st > seq1_end) or (seq2_st > seq2_end):
      return None 
    else: 
      alignment_score = self.dp_2d[seq1_end][seq2_end] - self.dp_2d[seq1_st-1][seq2_st-1]
      if alignment_score <= 0:
        return None
      seq1_st -= 1
      seq2_st -= 1
      seq1_end -= 1
      seq2_end -= 1 
      return {
        'seq1_st':seq1_st,
        'seq1_end':seq1_end,
        'seq2_st':seq2_st,
        'seq2_end':seq2_end,
        'clipped_for_overlap':clipped_for_overlap,
        'alignment_score':alignment_score,
      } 
    
  def create_alignments(self, match_element_pair_threshold=0, top_k=-1):
    if self.sw_computed is False:
      self.compute_smith_waterman()
    self.set_global_align_variables()
    aligned_segments = []
    for count, cell in enumerate(self.all_cells):
      if count == top_k or cell['dp_val'] == 0:
        break
      current_aligned_segment = self.traverse(cell['i'], cell['j'])
      if current_aligned_segment is not None:
        aligned_segments.append(current_aligned_segment)
    for i in range(1, len(self.s1_to_s2_align)):
        if self.s1_to_s2_align[i] >= 0:
          self.s1_to_s2_align[i-1] = self.s1_to_s2_align[i] - 1
        else:
          self.s1_to_s2_align[i-1] = self.s1_to_s2_align[i]
    for i in range(1, len(self.s2_to_s1_align)):
        if self.s2_to_s1_align[i] >= 0:
          self.s2_to_s1_align[i-1] = self.s2_to_s1_align[i] - 1
        else:
          self.s2_to_s1_align[i-1] = self.s2_to_s1_align[i]
    self.s1_to_s2_align = self.s1_to_s2_align[:-1]
    self.s2_to_s1_align = self.s2_to_s1_align[:-1]

    aligned_segments.sort(reverse = True, key=lambda x: x['alignment_score'])
    return aligned_segments, self.s1_to_s2_align, self.s2_to_s1_align
