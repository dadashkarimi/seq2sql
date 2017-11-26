"""Module that handles all augmentation."""
import collections
import random
import re
import sys

import domains
from grammar import Grammar
from vocabulary import Vocabulary 

class Augmenter(object):
  def __init__(self, domain, dataset, aug_types):
    self.domain = domain
    self.dataset = dataset
    self.dataset_set = set(dataset)
    self.setup_grammar(aug_types)

  def setup_grammar(self, aug_types):
    grammar = Grammar(self.dataset)
    for aug_type in aug_types:
      if aug_type == 'entity':
        grammar = self.induce_entity_grammar(grammar)
      elif aug_type == 'nesting':
        grammar = self.induce_nesting_grammar(grammar)
      elif aug_type.startswith('concat'):
        concat_num = int(aug_type[6:])
        grammar = self.induce_concat_grammar(grammar, concat_num)
    self.grammar = grammar

  def splice(self, s, swaps):
    # Process s from right to left
    swaps.sort(key=lambda x: x[0], reverse=True)

    cur_left = len(s)
    new_s = s
    for span, rep in swaps:
      # Ensure disjoint
      if span[1] > cur_left:
        print >> sys.stderr, s
        print >> sys.stderr, swaps
        raise ValueError('Non-disjoint spans detected')
      new_s = new_s[:span[0]] + rep + new_s[span[1]:]
      cur_left = span[0]
    return new_s

  def induce_entity_grammar(self, start_grammar):
    """Induce an entity-swapping grammar.

    Get the entities from the original dataset.
    Get the places to put holes from start_grammar.
    """
    new_grammar = Grammar()

    # Entity rules
    for x, y in self.dataset:
      alignments = self.domain.get_entity_alignments(x, y)
      for cat, x_span, y_span in alignments:
        x_str = x[x_span[0]:x_span[1]]
        y_str = y[y_span[0]:y_span[1]]
        new_grammar.add_rule(cat, x_str, y_str)

    # Root/template rules
    for cat, x_str, y_str in start_grammar.rule_list:
      # Anchor on single mention in x--allow one-to-many x-to-y mapping
      alignments = self.domain.get_entity_alignments(x_str, y_str)
      x_swaps = list(set(
          [(x_span, '%s_%d' % (inner_cat, x_span[0]))
           for i, (inner_cat, x_span, y_span) in enumerate(alignments)]))
      x_new = self.splice(x_str, x_swaps)
      y_swaps = [(y_span, '%s_%d' % (inner_cat, x_span[0]))
                 for i, (inner_cat, x_span, y_span) in enumerate(alignments)]
      y_new = self.splice(y_str, y_swaps)
      new_grammar.add_rule(cat, x_new, y_new)

    #new_grammar.print_self()
    return new_grammar

  def induce_nesting_grammar(self, start_grammar):
    """Induce an entity-swapping grammar.

    Get everything from the start_grammar.
    """
    new_grammar = Grammar()
    for cat, x_str, y_str in start_grammar.rule_list:
      alignments, productions = self.domain.get_nesting_alignments(x_str, y_str)

      # Replacements
      for cat_p, x_p, y_p in productions:
        new_grammar.add_rule(cat_p, x_p, y_p)

      # Templates
      x_swaps = list(set(
          [(x_span, '%s_%d' % (inner_cat, x_span[0]))
           for i, (inner_cat, x_span, y_span) in enumerate(alignments)]))
      x_new = self.splice(x_str, x_swaps)
      y_swaps = [(y_span, '%s_%d' % (inner_cat, x_span[0]))
                 for i, (inner_cat, x_span, y_span) in enumerate(alignments)]
      y_new = self.splice(y_str, y_swaps)
      new_grammar.add_rule(cat, x_new, y_new)
    new_grammar.print_self()
    return new_grammar

  def induce_concat_grammar(self, start_grammar, concat_num):
    new_grammar = Grammar()

    for cat, x_str, y_str in start_grammar.rule_list:
      if cat == start_grammar.ROOT:
        new_grammar.add_rule('$sentence', x_str, y_str)
      else:
        new_grammar.add_rule(cat, x_str, y_str)
    root_str = (' %s ' % Vocabulary.END_OF_SENTENCE).join(
        '$sentence_%d' % i for i in range(concat_num))
    new_grammar.add_rule(new_grammar.ROOT, root_str, root_str)
    #new_grammar.print_self()
    return new_grammar

  def sample(self, num):
    aug_data = []
    while len(aug_data) < num:
      x, y = self.grammar.sample()
      if (x, y) in self.dataset_set: continue
      aug_data.append((x, y))
    return aug_data

def main():
  """Print augmented data to stdout."""
  if len(sys.argv) < 5:
    print >> sys.stderr, 'Usage: %s [file] [domain] [aug-type] [num]' % sys.argv[0]
    sys.exit(1)
  fname, domain_name, aug_type_str, num = sys.argv[1:5]
  num = int(num)
  aug_types = aug_type_str.split('+')
  data = []
  domain = domains.new(domain_name)
  with open(fname) as f:
    for line in f:
      x, y = line.strip().split('\t')
      y = domain.preprocess_lf(y)
      data.append((x, y))
  augmenter = Augmenter(domain, data, aug_types)
  aug_data = augmenter.sample(num)
  for ex in aug_data:
    print '\t'.join(ex)

if __name__ == '__main__':
  main()
