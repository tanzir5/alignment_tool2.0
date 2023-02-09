import argparse
import seaborn as sns
import sys
import utility 

import numpy as np
from aligners.preprocessor import Preprocessor
from aligner import Aligner
from utility import get_paragraphs, parse_text, print_matches, show_heat_map
import matplotlib.pyplot as plt

# Define the parser
arg_parser = argparse.ArgumentParser(description='Text Aligner Tool')
arg_parser.add_argument("file1", help="source of text 1")
arg_parser.add_argument("file2", help="source of text 2")
arg_parser.add_argument('--parsed', action="store", dest="parsed", default='0', 
  help='denotes if the text has been broken down to list of smallest unit')
arg_parser.add_argument(
  '--unit1', action="store", dest='unit1', default='paragraph',
  help=('unit size of text 1\n' 
        'valid options are:\n'
        '1.character\n'
        '2.word\n'
        '3.paragraph\n'
        '4.chapter\n'
        )
  )
arg_parser.add_argument(
  '--unit2', action="store", dest='unit2', default='paragraph', help='same as unit1')
arg_parser.add_argument(
  '--sim', action="store", dest='sim', default='jaccard',
  help=('valid options are:\n'
        '1.jaccard\n'
        '2.sbert\n'
        '3.tf-idf\n'
        '4.glove\n'
        )
  )
arg_parser.add_argument(
  '--norm', action="store", dest='norm', default='sigmoid', 
  help=('valid options are:\n'
        '1.sigmoid\n'
        '2.log\n'
        '3.tanh\n'
        )
  )
arg_parser.add_argument('--z_thresh', action='store', dest='z_thresh', default='0', 
  help='min similiarty z-score needed to be non-negative')
arg_parser.add_argument('--min_thresh', action='store', dest='min_thresh', default='0',
  help='min final similarity score needed to be aligned with each other')
arg_parser.add_argument('-k', action='store', dest='k', default='3',
  help='k top matches will be shown.')
arg_parser.add_argument('--clip', action='store', dest='clip', default='-1',
  help='Clip max length of both lists of text')
args = arg_parser.parse_args()

if args.sim == 'jaccard':
  args.sim = 'overlapping_token_similarity_matrix'

if args.parsed == '0':
  parsed_text1 = parse_text(args.file1, args.unit1)
  parsed_text2 = parse_text(args.file2, args.unit2)

similarity_config = {}
similarity_config['sim'] = args.sim
if args.norm == 'sigmoid':
  similarity_config['normalization'] = 'z_normalize_with_sigmoid'

print("similarity computation started")
preprocessor = Preprocessor(parsed_text1, parsed_text2, 
                            token_size_of_A = args.unit1,
                            token_size_of_B = args.unit2,
                            threshold = float(args.z_thresh),
                            similarity_config = similarity_config,
                            clip_length = int(args.clip))

print(preprocessor.similarity_matrix.shape)
print("similarity computation ended\n")
'''t = 5
max_sim_indices = utility.k_largest_index_argsort(preprocessor.similarity_matrix, t)
print(max_sim_indices)
for cell in max_sim_indices:
  i = cell[0]
  j = cell[1]
  print(preprocessor.similarity_matrix[i][j])
  print(parsed_text1[i])
  print()
  print(parsed_text2[j])
exit(0)'''
#sns.heatmap(preprocessor.similarity_matrix, xticklabels = 20, yticklabels = 20, cmap = sns.cm.rocket_r)
#plt.show()    
print("DP computation started")
aligner = Aligner(preprocessor.similarity_matrix)
aligner.smith_waterman()
print("DP computation ended\n")

'''print("Getting best matches")
matches = aligner.get_best_matches(float(args.min_thresh), int(args.k))
print("Getting best matches ended\n")
print_matches(matches, parsed_text1, parsed_text2)
print(matches)
'''
show_heat_map(aligner.dp)

#'''
