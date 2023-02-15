First, git clone the repository. 

Then it is highly recommended (but not required) that you create a virtual environment first.
```
python -m venv align_env 
source align_env/bin/activate
```

Then run the run.sh script first. 
```
./run.sh 
```

Then try out the run_simple_example.py to see how to use the alignment tool. 

```
from aligners.align_pipeline import align_sequences 

text1 = "I eat rice, burger, pizza. Rome is in Italy."
text2 = "Rome is in Italy. They are in Rome. I eat rice, burger, pizza."
ret = align_sequences(text1, text2)
alignments = ret['alignments'] 
for seg_pair in alignments:
  print(seg_pair)
```

Your output should look like the following:
```
Running on  cpu/gpu
{'seq1_st': 0, 'seq1_end': 25, 'seq2_st': 18, 'seq2_end': 43, 'text_a': 'I eat rice, burger, pizza.', 'text_b': 'I eat rice, burger, pizza.'}
{'seq1_st': 27, 'seq1_end': 43, 'seq2_st': 0, 'seq2_end': 16, 'text_a': 'Rome is in Italy.', 'text_b': 'Rome is in Italy.'}
```
The list of aligned segment pairs are returned and printed. 
*seq1_st* denotes the starting index of the segment from the first string 
*seq1_ed* denotes the ending index of the segment from the first string. 
*text_a* denotes the text of the aligned segment from the first string. 

Similar information for sequence 2 is returned and printed too.


Let us look at how we can do semantic matching and control the level of semantic similarity. 
```
text1 = "This is a nice restaurant"
text2 = "This is a beautiful restaurant"
ret = align_sequences(text1, text2, sim='glove', z_thresh=1)
print(ret['alignments'])
```

We pass *glove* as similarity metric since we want to match semantically similar words according to glove embeddings. We pass *z_thresh=1* to denote the level of semantically similar two words have to be to get aligned. If you try with *z_thresh=2*, then "nice" and "beautiful" will not be matched and we will get two disjoint aligned segment pairs. More details on z_thresh coming soon.   

Try out the test_examples.py to see different ways you can use the *align_sequences()* function.
