"""Artificial data."""
import collections
import itertools
import os
import random
import re
import sys

from augmentation import Augmenter
import domains

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/artificial-final')

ENTITIES = ['ent:%02d' % x for x in range(20)]
RELATIONS = ['rel:%02d' % x for x in range(50)]

def gen_nested(depth=2):
  data = []
  for e in ENTITIES:
    for rels in itertools.product(RELATIONS, repeat=depth):
      rels = list(rels)
      x = ' of '.join(rels + [e])
      y = '( ' + ' ( '.join(['_%s' % r for r in rels]) + ' _%s' % e + ' )' * depth
      data.append((x, y))
  random.shuffle(data)
  return data

def sample_nested(depth=2, num=0):
  data = set()
  while len(data) < num:
    rels = [random.sample(RELATIONS, 1)[0] for i in range(depth)]
    e = random.sample(ENTITIES, 1)[0]
    x = ' of '.join(rels + [e])
    y = '( ' + ' ( '.join(['_%s' % r for r in rels]) + ' _%s' % e + ' )' * depth
    data.add((x, y))
  return list(data)

# Augmentation Routines
# Don't use any external information besides what's in the datsaet
# and entity alignments
# Assume that there's a single entity in each example.
def get_templates(dataset):
  def create_template(ex):
    x, y = ex
    x_new = re.sub('ent:[0-9]{2}', '%s', x)
    y_new = re.sub('_ent:[0-9]{2}', '%s', y)
    return (x_new, y_new)
  templates = [create_template(ex) for ex in dataset]
  return templates

def augment_nesting(dataset):
  # Augment by ensting one thing within another
  def combine(template, ex):
    x_t, y_t = template
    x_e, y_e = ex
    x_new = x_t % x_e
    y_new = y_t % y_e
    return (x_new, y_new)
  templates = get_templates(dataset)
  new_examples = [combine(t, ex) for t in templates for ex in dataset]
  return new_examples

def augment_entities(dataset):
  # Augment by swapping in new entity names
  def replace(template, ent):
    x_t, y_t = template
    x_new = x_t % ent
    y_new = y_t % ('_' + ent)
    return (x_new, y_new)
  entities = sorted(list(set(re.search('ent:[0-9]{2}', x).group(0)
                             for x, y in dataset)))
  templates = get_templates(dataset)
  new_examples = [replace(t, e) for t in templates for e in entities]
  return new_examples

def write_data(basename, data):
  print >> sys.stderr, 'Writing %s' % basename
  out_path = os.path.join(OUT_DIR, basename)
  with open(out_path, 'w') as f:
    for x, y in data:
      print >>f, '%s\t%s' % (x, y)

def main():
  random.seed(0)
  base_data = gen_nested()
  base_train, base_test = base_data[:100], base_data[-500:]
  write_data('train_base100.tsv', base_train)
  write_data('test_base500.tsv', base_test)

  domain = domains.new('artificial')
  augmenter_entity = Augmenter(domain, base_train, ['entity'])
  augmenter_nesting = Augmenter(domain, base_train, ['nesting', 'entity'])
  deeper = sample_nested(depth=4, num=500)
  entity_data = augmenter_entity.sample(500)
  nesting_data = augmenter_nesting.sample(500)
  aug_nums = (25, 50, 75, 100, 150, 200, 250, 300, 400, 500)
  for n in aug_nums:
    write_data('train_base%d.tsv' % (100 + n), base_data[:(100+n)])
    write_data('train_base100_entity%d.tsv' % n, base_train + entity_data[:n])
    write_data('train_base100_nesting%d.tsv' % n, base_train + nesting_data[:n])
    write_data('train_base100_deeper%d.tsv' % n, base_train + deeper[:n])

if __name__ == '__main__':
  main()
