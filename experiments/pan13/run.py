#!/usr/bin/env python
""" Plagiarism detection for near-duplicate plagiarism.

    This program provides the baseline for the PAN 2013 Plagiarism Detection
    task and can be used as outline for a plagiarism detection program.
"""
__author__ = 'Arnd Oberlaender'
__email__ = 'arnd.oberlaender at uni-weimar dot de'
__version__ = '1.1'

import os
import string
import sys
import xml.dom.minidom
import codecs
from aligners.align_pipeline import align_sequences
from tqdm import tqdm
import multiprocessing as mp

# Const
# =====

DELETECHARS = ''.join([string.punctuation, string.whitespace])
LENGTH = 50


# Helper functions
# ================

""" The following functions are some simple helper functions you can utilize
and modify to fit your own program.
"""

def tokenize(text, length):
    """ Tokeniz a given text and return a dict containing all start and end
    positions for each token.
    Characters defined in the global string DELETECHARS will be ignored.

    Keyword arguments:
    text   -- the text to tokenize
    length -- the length of each token
    """
    tokens = {}
    token = []

    for i in range(0, len(text)):
        if text[i] not in DELETECHARS:
            token.append((i, text[i]))
        if len(token) == length:
            ngram = ''.join([x[1].lower() for x in token])
            if ngram not in tokens:
                tokens[ngram] = []
            tokens[ngram].append((token[0][0], token[-1][0]))
            token = token[1:]

    return tokens

def serialize_features(susp, src, features, outdir):
    """ Serialze a feature list into a xml file.
    The xml is structured as described in the readme file of the 
    PAN plagiarism corpus 2012. The filename will follow the naming scheme
    {susp}-{src}.xml and is located in the current directory.
    Existing files will be overwritten.

    Keyword arguments:
    susp     -- the filename of the suspicious document
    src      -- the filename of the source document
    features -- a list containing feature-tuples of the form
                ((start_pos_susp, end_pos_susp),
                 (start_pos_src, end_pos_src))
    """
    impl = xml.dom.minidom.getDOMImplementation()
    doc = impl.createDocument(None, 'document', None)
    root = doc.documentElement
    root.setAttribute('reference', susp)
    doc.createElement('feature')

    for f in features:
        feature = doc.createElement('feature')
        feature.setAttribute('name', 'detected-plagiarism')
        feature.setAttribute('this_offset', str(f[1][0]))
        feature.setAttribute('this_length', str(f[1][1] - f[1][0]))
        feature.setAttribute('source_reference', src)
        feature.setAttribute('source_offset', str(f[0][0]))
        feature.setAttribute('source_length', str(f[0][1] - f[0][0]))
        root.appendChild(feature)

    doc.writexml(open(outdir + susp.split('.')[0] + '-'
                      + src.split('.')[0] + '.xml', 'w'),
                 encoding='utf-8')


# Plagiarism pipeline
# ===================

""" The following class implement a very basic baseline comparison, which
aims at near duplicate plagiarism. It is only intended to show a simple
pipeline your plagiarism detector can follow.
Replace the single steps with your implementation to get started.
"""

class PlagDetector:
    def __init__(
      self, 
      susp, 
      src, 
      outdir,
      unit1='sentence',
      unit2='sentence',
      sim='sbert', 
      z_thresh=4,
      ):
        self.susp = susp
        self.src = src
        self.susp_file = os.path.split(susp)[1]
        self.src_file = os.path.split(src)[1]
        self.susp_id = os.path.splitext(susp)[0]
        self.src_id = os.path.splitext(src)[0]
        self.output = self.susp_id + '-' + self.src_id + '.xml'
        self.detections = None
        self.outdir=outdir
        self.unit1 = unit1
        self.unit2 = unit2
        self.sim = sim
        self.z_thresh = z_thresh

    def read_files(self):
        """ Preprocess the suspicious and source document. """
        # TODO: Implement your preprocessing steps here.
        susp_fp = codecs.open(self.susp, 'r', 'utf-8')
        self.susp_text = susp_fp.read()
        susp_fp.close()
        #self.tokens = tokenize(self.susp_text, LENGTH)
        src_fp = codecs.open(self.src, 'r', 'utf-8')
        self.src_text = src_fp.read()
        src_fp.close()

    def detect(self):
      #print("lens", len(self.src_text.split()), len(self.susp_text.split()))
      ret = align_sequences(
          self.src_text, 
          self.susp_text,
          unit1=self.unit1, 
          unit2=self.unit2,
          sim=self.sim,
          z_thresh=self.z_thresh, 
          return_aligner=True,
      )
      print("lens")
      print(ret['aligner'].dp.shape)
      detections = []
      for alignment in ret['alignments']:
        detections.append((
          (alignment['seq1_st'], alignment['seq1_end']),
          (alignment['seq2_st'], alignment['seq2_end'])
        ))
      print(len(detections))
      return detections

    def process(self):
        self.read_files()
        self.detections = self.detect()
        self.postprocess()

    def compare(self):
        """ Test a suspicious document for near-duplicate plagiarism with regards to
        a source document and return a feature list.
        """

        #TODO: Implement your comparison here and replace the following
        #      algorithm with your own.

        detections = []
        skipto = -1
        token = []
        for i in range(0, len(self.src_text)):
            if i > skipto:
                if self.src_text[i] not in DELETECHARS:
                    token.append((i, self.src_text[i]))
                if len(token) == LENGTH:
                    ngram = ''.join([x[1].lower() for x in token])
                    if ngram in self.tokens:
                        d = ((token[0][0],token[-1][0]),
                             (self.tokens[ngram][0][0],
                              self.tokens[ngram][0][1]))
                        for t in self.tokens[ngram]:
                            start_src = token[0][0]
                            start_susp = t[0]
                            while (start_susp < len(self.susp_text) and
                                   start_src < len(self.src_text) and
                                   self.src_text[start_src] == self.susp_text[start_susp]):
                                start_susp = start_susp + 1
                                start_src = start_src + 1
                                while (start_susp < len(self.susp_text) and
                                       self.susp_text[start_susp] in DELETECHARS):
                                    start_susp = start_susp + 1
                                while (start_src < len(self.src_text) and
                                       self.src_text[start_src] in DELETECHARS):
                                    start_src = start_src + 1
                            if (start_src - 1) - token[0][0] > d[0][1] - d[0][0]:
                                d = ((token[0][0], start_src), (t[0], start_susp))
                        detections.append(d)
                        skipto = d[0][1]
                        if skipto < len(self.src_text):
                            token = [(skipto, self.src_text[skipto])]
                        else:
                            break
                    else:
                        token = token[1:]

        print(len(detections))

        return detections

    def postprocess(self):
        """ Postprocess the results. """
        # TODO: Implement your postprocessing steps here.
        serialize_features(self.susp_file, self.src_file, self.detections, self.outdir)

# Main
# ====

def single_process(
  susp, 
  src, 
  outdir,
  z_thresh, 
  unit1='sentence',  
  unit2='sentence',
  sim='sbert', 
):
  plag_detector = PlagDetector(susp, src, outdir, z_thresh=z_thresh)
  plag_detector.process()

def parallel_process(
  lines,
  outdir, 
  unit1='sentence',  
  unit2='sentence',
  sim='sbert', 
  z_thresh=4,
):
  pool = mp.Pool(mp.cpu_count())
  jobs = []
  for line in lines:
    susp, src = line.split()
    job = pool.apply_async(
      single_process, 
      (os.path.join(suspdir, susp), os.path.join(srcdir, src), outdir, z_thresh)
    )
    jobs.append(job)
    
  print("length of jobs", len(jobs))
  for i, job in tqdm(enumerate(jobs)): 
    job.get()
    print(i, "is done")

  pool.close()
  pool.join()
  
if __name__ == "__main__":
    """ Process the commandline arguments. We expect three arguments: The path
    pointing to the pairs file and the paths pointing to the directories where
    the actual source and suspicious documents are located.
    """
    if len(sys.argv) == 6:
        srcdir = sys.argv[2]
        suspdir = sys.argv[3]
        outdir = sys.argv[4]
        if outdir[-1] != "/":
            outdir+="/"
        lines = open(sys.argv[1], 'r').readlines()
        for z in range(0, 12, 1):
          if sys.argv[5] == 'single':
            pass
            '''for line in tqdm(lines):
              susp, src = line.split()
              single_process(os.path.join(suspdir, susp),
                             os.path.join(srcdir, src), outdir)'''
          else:
            final_outdir = os.path.join(outdir, str(z)) + "/"
            if os.path.exists(final_outdir) is False:
              os.mkdir(final_outdir)
            parallel_process(lines, final_outdir, z_thresh=z)

    else:
        print('\n'.join(["Unexpected number of commandline arguments.",
                         "Usage: ./pan12-plagiarism-text-alignment-example.py {pairs} {src-dir} {susp-dir} {out-dir} {paralllel/single}"]))