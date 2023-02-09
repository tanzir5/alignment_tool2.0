You need to run the run.sh script first. 
```
./run.sh 
```

Try out the run_simple_example.py to see how to use the alignment tool. 

```
from aligners.align_pipeline import align_sequences 
import numpy as np
from aligners.smith_waterman import Aligner

text1 = "I eat rice, burger, pizza. Rome is in Italy."
text2 = "Rome is in Italy. I eat rice, burger, pizza. They are in Rome."
ret = align_sequences(text1, text2)
alignments = ret['alignments'] 
for seg_pair in alignments:
  print(seg_pair)
```

Try out the test_examples.py to see different ways you can use the *align_sequences()* function.