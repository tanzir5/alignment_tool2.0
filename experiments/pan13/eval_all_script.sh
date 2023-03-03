#!/bin/bash
for i in {-2..11}
do
  python experiments/pan13/eval2.py -p data/experiments/pan13/pan13-text-alignment-training-corpus-2013-01-21/small_04-translation-obfuscation/  -d data/experiments/pan13/outputs/sbert_04_sent/$i
done  