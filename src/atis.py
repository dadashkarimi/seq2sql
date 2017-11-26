"""Some code to deal with atis data."""
import collections
import glob
import os
import re
import sys

IN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/raw-oneshot')
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/processed')

def read_examples(filename):
  examples = []
  utterance = None
  logical_form = None
  with open(filename) as f:
    for i, line in enumerate(f):
      if i % 3 == 0:
        utterance = line.strip()
      elif i % 3 == 1:
        logical_form = line.strip()
        examples.append((utterance, logical_form))
  return examples

def split_logical_form(lf):
  words = lf.split()

  # Insert spaces and obscure predicates
  replacements = [
      ('(', ' ( _'),
      (')', ' ) '),
      (':', ':_'),
  ]
  for r in replacements:
    lf = lf.replace(r[0], r[1])
  return ' '.join(lf.split())

def process(filename, stemmer=None, less_copy=False):
  print >> sys.stderr, 'Processing %s' % filename
  basename = os.path.basename(filename)
  domain = basename.split('.')[0]
  stage = basename.split('.')[1]
  
  in_data = read_examples(filename)
  out_data = []
  for (utterance, logical_form) in in_data:
    print utterance,y
    input('im here ..')
    y = split_logical_form(logical_form)
    out_data.append((utterance, y))

  out_basename = '%s_%s.tsv' % (domain, stage)
  out_path = os.path.join(OUT_DIR, out_basename)
  with open(out_path, 'w') as f:
    for x, y in out_data:
      print >> f, '%s\t%s' % (x, y)

def main():
  if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)
  for filename in sorted(glob.glob(os.path.join(IN_DIR, 'atis.*'))):
    process(filename)

if __name__ == '__main__':
  main()
