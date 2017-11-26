"""A Synchronous Context-Free Grammar."""
import collections
import random
import re

class Grammar(object):
  ROOT = '$ROOT'

  def __init__(self, dataset=None):
    self.rules = collections.defaultdict(list)
    self.rule_list = []
    self.rule_sets = collections.defaultdict(set)
    if dataset:
      for x, y in dataset:
        self.add_rule(self.ROOT, x, y)

  def add_rule(self, cat, x, y):
    if (x, y) not in self.rule_sets[cat]:
      self.rule_sets[cat].add((x, y))
      self.rules[cat].append((x, y))
      self.rule_list.append((cat, x, y))

  def sample(self, cat=ROOT):
    """Generate a sample from this grammar."""
    x, y = random.sample(self.rules[cat], 1)[0]
    matches = re.finditer('(\\$[^ ]*)_([0-9]+)', y)
    #print cat, x, y
    for m in matches:
      inner_cat = m.group(1)
      num = int(m.group(2))
      nonterm = '%s_%d' % (inner_cat, num)
      x_rep, y_rep = self.sample(cat=inner_cat)
      x = x.replace(nonterm, x_rep)
      y = y.replace(nonterm, y_rep)
    return x, y

  def print_self(self):
    for k in self.rules:
      x, y = self.rules[k][0]
      print '%s: %d rules, example = (%s, %s)' % (k, len(self.rules[k]), x, y)
