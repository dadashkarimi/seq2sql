"""Reads a lexicon for ATIS.

A lexicon simply maps natural language phrases to identifiers in the ATIS database.
"""
import collections
import os
import re
import sys

from lexicon import Lexicon

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/atis/db')
if not os.path.isdir(DB_DIR):
  # We're on codalab
  DB_DIR = os.path.join(
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'atis-db')

def clean_id(s, id_suffix, strip=None):
  true_id = s.replace(' ', '_')
  if strip:
    for v in strip:
      true_id = true_id.replace(v, '').strip()
  return '%s:%s' % (true_id, id_suffix)

def clean_name(s, strip=None, split=None, prefix=None):
  if split:
    for v in split:
      s = s.replace(v, ' ')
  if strip:
    for v in strip:
      s = s.replace(v, '')
  if prefix:
    s = prefix + s
  return s

def read_db(basename, id_col, name_col, id_suffix,
            strip_id=None, strip_name=None, split_name=None, prefix_name=None):
  filename = os.path.join(DB_DIR, basename)
  data = []  # Pairs of (name, id)
  with open(filename) as f:
    for line in f:
      row = [s[1:-1] for s in re.findall('"[^"]*"|[0-9]+', line.strip())]
      cur_name = clean_name(row[name_col].lower(), strip=strip_name,
                            split=split_name, prefix=prefix_name)
      cur_id = clean_id(row[id_col].lower(), id_suffix, strip=strip_id)
      data.append((cur_name, cur_id))
  return data

def handle_times(lex):
  # Mod 12 deals with 12am/12pm special cases...
  lex.add_handler('([0-9]{1,2})am$',
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12))
  lex.add_handler('([0-9]{1,2})pm$',
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12 + 12))
  lex.add_handler('([0-9]{1,2})([0-9]{2})am$', 
                  lambda m: '%d%02d:_ti' % (int(m.group(1)) % 12, int(m.group(2))))
  lex.add_handler('([0-9]{1,2})([0-9]{2})pm$', 
                  lambda m: '%d%02d:_ti' % (int(m.group(1)) % 12 + 12, int(m.group(2))))
  lex.add_handler("([0-9]{1,2}) o'clock am$",
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12))
  lex.add_handler("([0-9]{1,2}) o'clock pm$",
                  lambda m: '%d00:_ti' % (int(m.group(1)) % 12 + 12))

def handle_flight_numbers(lex):
  lex.add_handler('[0-9]{2,}$', lambda m: '%d:_fn' % int(m.group(0)))

def handle_dollars(lex):
  lex.add_handler('([0-9]+) dollars$', lambda m: '%d:_do' % int(m.group(1)))

def get_manual_lexicon():
  DAYS_OF_WEEK = [
      (s, '%s:_da' % s) 
      for s in ('monday', 'tuesday', 'wednesday', 'thursday', 
                'friday', 'saturday', 'sunday')
  ]
  # For dates
  WORD_NUMBERS = [('one', '1:_dn'), ('two', '2:_dn'), ('three', '3:_dn'), ('four', '4:_dn'), ('five', '5:_dn'), ('six', '6:_dn'), ('seven', '7:_dn'), ('eight', '8:_dn'), ('nine', '9:_dn'), ('ten', '10:_dn'), ('eleven', '11:_dn'), ('twelve', '12:_dn'), ('thirteen', '13:_dn'), ('fourteen', '14:_dn'), ('fifteen', '15:_dn'), ('sixteen', '16:_dn'), ('seventeen', '17:_dn'), ('eighteen', '18:_dn'), ('nineteen', '19:_dn'), ('twenty', '20:_dn'), ('twenty one', '21:_dn'), ('twenty two', '22:_dn'), ('twenty three', '23:_dn'), ('twenty four', '24:_dn'), ('twenty five', '25:_dn'), ('twenty six', '26:_dn'), ('twenty seven', '27:_dn'), ('twenty eight', '28:_dn'), ('twenty nine', '29:_dn'), ('thirty', '30:_dn'), ('thirty one', '31:_dn')]
  ORDINAL_NUMBERS = [('second', '2:_dn'), ('third', '3:_dn'), ('fourth', '4:_dn'), ('fifth', '5:_dn'), ('sixth', '6:_dn'), ('seventh', '7:_dn'), ('eighth', '8:_dn'), ('ninth', '9:_dn'), ('tenth', '10:_dn'), ('eleventh', '11:_dn'), ('twelfth', '12:_dn'), ('thirteenth', '13:_dn'), ('fourteenth', '14:_dn'), ('fifteenth', '15:_dn'), ('sixteenth', '16:_dn'), ('seventeenth', '17:_dn'), ('eighteenth', '18:_dn'), ('nineteenth', '19:_dn'), ('twentieth', '20:_dn'), ('twenty first', '21:_dn'), ('twenty second', '22:_dn'), ('twenty third', '23:_dn'), ('twenty fourth', '24:_dn'), ('twenty fifth', '25:_dn'), ('twenty sixth', '26:_dn'), ('twenty seventh', '27:_dn'), ('twenty eighth', '28:_dn'), ('twenty ninth', '29:_dn'), ('thirtieth', '30:_dn'), ('thirty first', '31:_dn')]  # Prefer first class to "first = 1"
  MEALS = [(m, '%s:_me' % m) for m in ('breakfast', 'lunch', 'dinner', 'snack')]

  lex = Lexicon()
  lex.add_entries(read_db('CITY.TAB', 1, 1, '_ci', strip_id=['.']))
  lex.add_entries(DAYS_OF_WEEK)
  lex.add_entries([(x + 's', y) for x, y in DAYS_OF_WEEK])  # Handle "on tuesdays"
  lex.add_entries(read_db('AIRLINE.TAB', 0, 1, '_al',
                          strip_name=[', inc.', ', ltd.']))
  handle_times(lex)
  lex.add_entries(read_db('INTERVAL.TAB', 0, 0, '_pd'))
  lex.add_entries(WORD_NUMBERS)
  lex.add_entries(ORDINAL_NUMBERS)
  lex.add_entries(read_db('MONTH.TAB', 1, 1, '_mn'))
  lex.add_entries(read_db('AIRPORT.TAB', 0, 1, '_ap',
                          strip_name=[], split_name=['/']))
  lex.add_entries(read_db('COMP_CLS.TAB', 1, 1, '_cl'))
  lex.add_entries(read_db('CLS_SVC.TAB', 0, 0, '_fb', prefix_name='code '))
  handle_flight_numbers(lex)
  lex.add_entries(MEALS)
  handle_dollars(lex)
  return lex

#   8207 _ci = cities
#    888 _da = days of the week
#    735 _al = airlines
#    607 _ti = times
#    594 _pd = time of day
#    404 _dn = date number
#    389 _mn = month
#    203 _ap = airport
#    193 _cl = class
#     72 _fb = fare code
#     65 _fn = flight number
#     52 _me = meal
#     50 _do = dollars
# ----------------------
#     28 _rc
#     23 _ac = aircraft
#     22 _yr
#      4 _mf
#      2 _dc
#      2 _st
#      1 _hr

def print_aligned(a, b, indent=0):
  a_toks = []
  b_toks = []
  for x, y in zip(a, b):
    cur_len = max(len(x), len(y))
    a_toks.append(x.ljust(cur_len))
    b_toks.append(y.ljust(cur_len))

  prefix = ' ' * indent
  print '%s%s' % (prefix, ' '.join(a_toks))
  print '%s%s' % (prefix, ' '.join(b_toks))

def parse_entry(line):
  """Parse an entry from the CCG lexicon."""
  return tuple(line.strip().split(' :- NP : '))

def get_ccg_lexicon():
  lexicon = Lexicon()
  filename = os.path.join(DB_DIR, 'lexicon.txt')
  entries = []
  with open(filename) as f:
    for line in f:
      x, y = line.strip().split(' :- NP : ')
      y = y.replace(':', ':_').strip()
      entries.append((x, y))
  lexicon.add_entries(entries)
  return lexicon

def get_lexicon():
  return get_ccg_lexicon()
  #return get_manual_lexicon()

if __name__ == '__main__':
  # Print out the lexicon
  lex = get_lexicon()
  print 'Lexicon entries:'
  for name, entity in lex.entries.iteritems():
    print '  %s -> %s' % (name, entity)
  print 'Unique word map:'
  for word, entity in lex.unique_word_map.iteritems():
    print '  %s -> %s'  % (word, entity)

  print 'Test cases:'
  lex.test_handlers('8am')
  lex.test_handlers('8pm')
  lex.test_handlers('12am')
  lex.test_handlers('12pm')
  lex.test_handlers('832am')
  lex.test_handlers('904am')
  lex.test_handlers('1204am')
  lex.test_handlers('832pm')
  lex.test_handlers('904pm')
  lex.test_handlers('1204pm')
  lex.test_handlers("8 o'clock am")
  lex.test_handlers("8 o'clock pm")
  lex.test_handlers('21')
  lex.test_handlers('4341')
  lex.test_handlers('4341 dollars')
  lex.test_map('unk:1234:chicago')
  lex.test_map('unk:23456:11pm')

  with open('data/atis/processed/atis_train.tsv') as f:
    for line in f:
      words = line.split('\t')[0].split(' ')
      entities = lex.map_over_sentence(words) 
      print '-' * 80
      print_aligned(words, entities, indent=2)
