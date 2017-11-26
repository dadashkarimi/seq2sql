"""Some code to deal with geo880 data."""
import collections
import glob
from nltk.stem.porter import PorterStemmer
import os
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/sempre-examples')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed')
STEM_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed-stems')
LESS_COPY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/geo880/processed-lesscopy')

def read_examples(filename):
  examples = []
  utterance = None
  logical_form = None
  with open(filename) as f:
    for line in f:
      line = line.strip()
      if line.startswith('(utterance'):
        utterance = re.match('\(utterance "(.*)"\)', line).group(1)
      elif line.startswith('(targetFormula'):
        logical_form = re.match(
            r'\(targetFormula \(string "(.*)"\)\)', line).group(1)
        examples.append((utterance, logical_form))
  return examples

def split_logical_form(lf):
  replacements = [
      ('(', ' ( '),
      (')', ' ) '),
      (',', ' , '),
      ("'", " ' "),
      ("\\+", " \\+ "),
  ]
  for a, b in replacements:
    lf = lf.replace(a, b)
  return ' '.join(lf.split())

def reduce_copying(lf):
  # List all predicates (whitelist)
  PREDS = [
      'cityid', 'countryid', 'placeid', 'riverid', 'stateid',
      'capital', 'city', 'lake', 'major', 'mountain', 'place', 'river',
      'state', 'area', 'const', 'density', 'elevation', 'high_point',
      'higher', 'loc', 'longer', 'low_point', 'lower', 'len', 'next_to',
      'population', 'size', 'traverse',
      'answer', 'largest', 'smallest', 'highest', 'lowest', 'longest',
      'shortest', 'count', 'most', 'fewest', 'sum']
  toks = ['_' + w if w in PREDS else w for w in lf.split()]
  return ' '.join(toks)

def write(out_basename, out_data, stemmer, less_copy):
  if stemmer:
    out_path = os.path.join(STEM_DIR, out_basename)
  elif less_copy:
    out_path = os.path.join(LESS_COPY_DIR, out_basename)
  else:
    out_path = os.path.join(OUT_DIR, out_basename)
  with open(out_path, 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)

def process(filename, stemmer=None, less_copy=False):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  domain = basename.split('_')[0]
  stage = basename.split('_')[1].split('.')[0]
  
  in_data = read_examples(filename)
  out_data = []
  for (utterance, logical_form) in in_data:
    if stemmer:
      utterance = ' '.join(stemmer.stem(w) for w in utterance.split())
    y = split_logical_form(logical_form)
    if less_copy:
      y = reduce_copying(y)
    out_data.append((utterance, y))

  out_basename = '%s_%s.tsv' % (domain, stage)
  write(out_basename, out_data, stemmer, less_copy)
  if stage == 'train500':
    for n in (100, 200, 300, 400):
      cur_data = out_data[:n]
      cur_basename = '%s_train%d.tsv' % (domain, n)
      write(cur_basename, cur_data, stemmer, less_copy)

def main():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  if not os.path.exists(STEM_DIR):
    os.makedirs(STEM_DIR)
  if not os.path.exists(LESS_COPY_DIR):
    os.makedirs(LESS_COPY_DIR)
  stemmer = PorterStemmer()
  for filename in sorted(glob.glob(os.path.join(IN_DIR, '*.examples'))):
    process(filename)
    process(filename, less_copy=True)
    process(filename, stemmer=stemmer)

if __name__ == '__main__':
  main()
