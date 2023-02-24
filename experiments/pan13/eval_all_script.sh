#!/bin/bash
for i in {-2..30}
do
  python experiments/pan13/eval2.py -p data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/small_02-no-obfuscation/  -d data/experiments/pan13/outputs/jaccard_02_sent/$i
done  