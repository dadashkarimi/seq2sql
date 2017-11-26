"""A lexicon maps input substrings to an output token."""
import collections
import itertools
import re
import sys

def strip_unk(w):
  # Strip unk:%06d identifiers
  m = re.match('^unk:[0-9]{6,}:(.*)$', w)
  if m:
    return m.group(1)
  else:
    return w

class Lexicon:
  """A Lexicon class.

  The lexicon stores two types of rules:
    1. Entries are pairs (name, entity), 
       where name could be a single word or multi-word phrase.
    2. Handlers are pairs (regex, func(match) -> entity).
       regex is checked to see if it matches any span of the input.
       If so, the function is applied to the match object to yield an entity.

  We additionally keep track of:
    3. Unique words.  If a word |w| appears in exactly one entry (|n|, |e|),
       then a lower-precision rule maps |w| directly to |e|, even if the
       entire name |n| is not present.

  Rules take precedence in the order given: 1, then 2, then 3.
  Within each block, rules that match larger spans take precedence
  over ones that match shorter spans.
  """
  def __init__(self):
    self.entries = collections.OrderedDict()
    self.handlers = []
    self.unique_word_map = collections.OrderedDict()
    self.seen_words = set()

  def add_entries(self, entries):
    for name, entity in entries:
      # Update self.entries
      if name in self.entries:
        if self.entries[name] != entity:
          print >> sys.stderr, 'Collision detected: %s -> %s, %s' % (
              name, self.entries[name], entity)
          continue
      self.entries[name] = entity

      # Update self.unique_word_map
      for w in name.split(' '):
        if w in self.seen_words:
          # This word is not unique!
          if w in self.unique_word_map:
            del self.unique_word_map[w]
        else:
          self.unique_word_map[w] = entity
          self.seen_words.add(w)

  def add_handler(self, regex, func):
    self.handlers.append((regex, func))

  def test_handlers(self, s):
    """Apply all handlers to a word; for debugging."""
    entities = []
    for regex, func in self.handlers:
      m = re.match(regex, s)
      if m:
        entities.append(func(m))
    print '  %s -> %s' % (s, entities)

  def test_map(self, s):
    words = s.split(' ')
    entities = self.map_over_sentence(words)
    print '  %s -> %s' % (s, entities)

  def map_over_sentence(self, words, return_entries=False):
    """Apply unambiguous lexicon rules to an entire sentence.
    
    Args:
      words: A list of words
      return_entries: if True, return a list (span_inds, entity) pairs instead.
    Returns: 
      A list of length len(words), where words[i] maps to retval[i]
    """
    entities = ['' for i in range(len(words))]
    ind_pairs = sorted(list(itertools.combinations(range(len(words) + 1), 2)),
                       key=lambda x: x[0] - x[1])
    ret_entries = []
    words = [strip_unk(w) for w in words]  # Strip unk:%06d stuff

    # Entries
    for i, j in ind_pairs:
      if any(x for x in entities[i:j]): 
        # Something in this span has already been assinged
        continue
      span = ' '.join(words[i:j])
      if span in self.entries:
        entity = self.entries[span]
        for k in range(i, j):
          entities[k] = entity
        ret_entries.append(((i, j), entity))

    # Handlers
    for i, j in ind_pairs:
      if any(x for x in entities[i:j]): continue
      span = ' '.join(words[i:j])
      for regex, func in self.handlers:
        m = re.match(regex, span)
        if m:
          entity = func(m)
          for k in range(i, j):
            entities[k] = entity
          ret_entries.append(((i, j), entity))

    # Unique words
    for i in range(len(words)):
      if entities[i]: continue
      word = words[i]
      if word in self.unique_word_map:
        entity = self.unique_word_map[word]
        entities[i] = entity
        ret_entries.append(((i, i+1), entity))

    if return_entries:
      return ret_entries
    return entities
