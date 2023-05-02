from aligners import align_pipeline
import glob

all_names = []

for fname in glob.glob('data/summary_exps/final_jsons/*.json'):
  name = fname.split('/')[-1][:-4]
  all_names.append(name)
  
for name in all_names:
  while True:
    unrelateds = np.random.choice(all_names, size=9, replace=False)
    if name not in unrelateds:
      break
  fname = 'data/summary_exps/final_jsons/'+name+'.json'
  with open(fname) as f:
    data_dict = json.load(fname)
  seq1 = data_dict['summary_sent'][0]
  summaries, books, units, real, scores = [], [], [], [], []
  
  for other_name in unrelateds:
    fname = 'data/summary_exps/final_jsons/'+name+'.json'
    with open(fname) as f:
      other_dict = json.load(fname)
    for unit_size in ['book_para', 'book_chunk']
      seq2 = other_dict[unit_size]
      summaries.append(name)
      books.append(other_name)
      units.append(unit_size)
      real.append(0)
      ret = align_pipeline(
        seq1=seq1,
        seq2=seq2,
        sim='jaccard',
        z_thresh=5)
      score = ret['alignments'][0]['alignment_score']
      scores.append(score)
  df = pd.DataFrame({'summary':summaries, 'book': books, 'unit': units, 'real':real, 'score':scores})

