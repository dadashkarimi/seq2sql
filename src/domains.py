"""A domain defines domain-specific processing."""
import os
import re
import subprocess
import sys
import tempfile

import atislexicon

class Domain(object):
  """A semantic parsing domain. 

  This parent class defines methods that can be overridden by subclasses to
  induce domain-specific behavior.
  By default, the methods here do the most generic thing.
  """
  DEFAULT_TRAIN_FILE = None
  def __init__(self):
    """Run some preprocessing of the dataset."""
    self.lex = None

  def preprocess_lf(self, lf):
    """Perform preprocessing on the given logical form."""
    return lf

  def postprocess_lf(self, lf):
    """Perform postprocessing (e.g. for printing or executing)."""
    return lf
  
  def get_entity_alignments(self, x, y):
    """Find entity-swapping alignments for an (x, y) pair.

    Args:
      x: An utterance, or utterance half of production rule
      y: A logical form, or logical form half of production rule
    Returns:
      List of pairs (category, x_span, y_span)
      where x_span and y_span are 2-tuples defining half-open intervals.
    """
    return []

  def get_nesting_alignments(self, x, y):
    """Find nesting alignments for an (x, y) pair.

    Args:
      x: An utterance, or utterance half of production rule
      y: A logical form, or logical form half of production rule
    Returns: (alignments, entities)
      alignments: List of pairs (category, x_span, y_span)
        where x_span and y_span are 2-tuples defining half-open intervals.
        Used to generate root-level rules.
      productions: a list (category, x_str, y_str)
        Used to generate type-level rules (e.g. $state)
    """
    return []

  def compare_answers(self, true_answers, all_derivs):
    pass

class MTDomain(Domain):
  #DEFAULT_TRAIN_FILE = os.path.join( 
  #    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  #    'data/geo880/processed-lesscopy/geo880_train600.tsv')

  def preprocess_lf(self, lf):
    """Standardize variable names with De Brujin indices."""
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w.isalpha() and len(w) == 1:
        if w in cur_vars:
          ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
          new_toks.append('V%d' % ind_from_end)
        else:
          cur_vars.append(w)
          new_toks.append('NV')
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def postprocess_lf(self, lf):
    """Undo the variable name standardization."""
    cur_var = chr(ord('A') - 1)
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w == 'NV':
        cur_var = chr(ord(cur_var) + 1)
        new_toks.append(cur_var)
      elif w.startswith('V'):
        ind = int(w[1:])
        new_toks.append(chr(ord(cur_var) - ind))
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def clean_name(self, name):
    return name.split(',')[0].replace("'", '').strip()

  def get_entity_alignments(self, x, y):
    alignments = []
    for m in re.finditer('_([a-z]*id) \\( ([^)]*) \\)', y):
      ent_type = m.group(1)
      ent_name = self.clean_name(m.group(2))
      y_span = (m.start(2), m.end(2))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'cityid' and '_' not in m.group(2): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      alignments.append(('$' + ent_type, x_span, y_span))
    return alignments

  def extract_type(self, y):
    y = self.postprocess_lf(y)  # undo de Brujin indexing
    if '_city ( A )' in y or '_capital ( A )' in y:
      return 'city'
    elif '_river ( A )' in y:
      return 'river'
    elif '_state ( A )' in y:
      return 'state'
    return None

  def prune_utterance(self, x):
    patterns = [
        '\\?$',
        '^in',
        '^what is',
        '^what are',
        '^what',
        '^which',
        '^name',
        '^give me',
        '^can you tell me',
        '^show',
    ]
    old_x = x
    for pattern in patterns:
      x = re.sub(pattern, '', x).strip()
    if x == old_x or x == old_x[:-2]:
      # print >> sys.stderr, 'Could not prune: "%s"' % old_x
      return None
    return x

  def prune_lf(self, y):
    m = re.match('_answer \\( NV , (.*) \\)', y)
    if not m:
      print >> sys.stderr, 'Bad logical form "%s"' % y
      return None
    lf = m.group(1)
    m2 = re.match('\\( (.*) \\)', lf)
    if m2:
      lf = m2.group(1)
    return lf

  def get_nesting_alignments(self, x, y):
    alignments = []
    productions = []
    # alignments (root level with holes)
    for m in re.finditer('_const \\( (V0|NV) , _([a-z]*)id \\( ([^)]*) \\) \\)', y):
      varname = m.group(1)
      ent_type = m.group(2)
      ent_name = self.clean_name(m.group(3))
      y_span = (m.start(0), m.end(0))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'place': continue  # No expressions evaluating to place
      if '(' in y[y_span[1]:] or ',' in y[y_span[1]:]: continue  # Want _const at end of lf
      if ent_type == 'city' and '_' not in m.group(3): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      cat = '$%s-%s' % (ent_type, varname)
      alignments.append((cat, x_span, y_span))

    # productions (whole expressions)
    prod_type = self.extract_type(y)
    if prod_type:
      x_prod = self.prune_utterance(x)
      if x_prod:
        y_prod = self.prune_lf(y)
        if y_prod:
          productions.append(('$%s-%s' % (prod_type, 'V0'), x_prod, y_prod))
          y_prod_nv = y_prod.replace('V0', 'NV', 1)
          productions.append(('$%s-%s' % (prod_type, 'NV'), x_prod, y_prod_nv))
    return alignments, productions

  def format_lf(self, lf):
    # Strip underscores, collapse spaces when not inside quotation marks
    lf = self.postprocess_lf(lf)
    toks = []
    in_quotes = False
    quoted_toks = []
    for t in lf.split():
      if in_quotes:
        if t == "'":
          in_quotes = False
          toks.append('"%s"' % ' '.join(quoted_toks))
          quoted_toks = []
        else:
          quoted_toks.append(t)
      else:
        if t == "'":
          in_quotes = True
        else:
          if len(t) > 1 and t.startswith('_'):
            toks.append(t[1:])
          else:
            toks.append(t)
    lf = ''.join(toks)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      lf = lf + ')' * diff
    return lf

  def get_denotation(self, line):
    m = re.search('\{[^}]*\}', line)
    if m: 
      return m.group(0)
    else:
      return line.strip()

  def print_failures(self, dens, name):
    num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
    num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
    num_join_error = sum('Join failed syntactically' in d for d in dens)
    print '%s: %d syntax errors, %d executor errors' % (
        name, num_syntax_error, num_exec_error)

  def is_error(self, d):
    return 'FAILED' in d or 'Join failed syntactically' in d

  def compare_answers(self, true_answers, all_derivs):
    all_lfs = ([self.format_lf(s) for s in true_answers] +
               [self.format_lf(' '.join(d.y_toks)) 
                for x in all_derivs for d in x])
    tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
    tf = tempfile.NamedTemporaryFile(suffix='.dlog')
    for line in tf_lines:
      print >>tf, line
      print line
    tf.flush()
    msg = subprocess.check_output(['evaluator/geoquery', tf.name])
    tf.close()
    denotations = [self.get_denotation(line)
                   for line in msg.split('\n')
                   if line.startswith('        Example')]
    true_dens = denotations[:len(true_answers)]
    all_pred_dens = denotations[len(true_answers):]

    # Find the top-scoring derivation that executed without error
    derivs, pred_dens = pick_derivations(true_dens, all_pred_dens, all_derivs,
                                         self.is_error)
    self.print_failures(true_dens, 'gold')
    self.print_failures(pred_dens, 'predicted')
    for t, p in zip(true_dens, pred_dens):
      print '%s: %s == %s' % (t == p, t, p)
    return derivs, [t == p for t, p in zip(true_dens, pred_dens)]


class ToyDomain(Domain):
  #DEFAULT_TRAIN_FILE = os.path.join( 
  #    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  #    'data/geo880/processed-lesscopy/geo880_train600.tsv')

  def preprocess_lf(self, lf):
    """Standardize variable names with De Brujin indices."""
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w.isalpha() and len(w) == 1:
        if w in cur_vars:
          ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
          new_toks.append('V%d' % ind_from_end)
        else:
          cur_vars.append(w)
          new_toks.append('NV')
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def postprocess_lf(self, lf):
    """Undo the variable name standardization."""
    cur_var = chr(ord('A') - 1)
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w == 'NV':
        cur_var = chr(ord(cur_var) + 1)
        new_toks.append(cur_var)
      elif w.startswith('V'):
        ind = int(w[1:])
        new_toks.append(chr(ord(cur_var) - ind))
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def clean_name(self, name):
    return name.split(',')[0].replace("'", '').strip()

  def get_entity_alignments(self, x, y):
    alignments = []
    for m in re.finditer('_([a-z]*id) \\( ([^)]*) \\)', y):
      ent_type = m.group(1)
      ent_name = self.clean_name(m.group(2))
      y_span = (m.start(2), m.end(2))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'cityid' and '_' not in m.group(2): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      alignments.append(('$' + ent_type, x_span, y_span))
    return alignments

  def extract_type(self, y):
    y = self.postprocess_lf(y)  # undo de Brujin indexing
    if '_city ( A )' in y or '_capital ( A )' in y:
      return 'city'
    elif '_river ( A )' in y:
      return 'river'
    elif '_state ( A )' in y:
      return 'state'
    return None

  def prune_utterance(self, x):
    patterns = [
        '\\?$',
        '^in',
        '^what is',
        '^what are',
        '^what',
        '^which',
        '^name',
        '^give me',
        '^can you tell me',
        '^show',
    ]
    old_x = x
    for pattern in patterns:
      x = re.sub(pattern, '', x).strip()
    if x == old_x or x == old_x[:-2]:
      # print >> sys.stderr, 'Could not prune: "%s"' % old_x
      return None
    return x

  def prune_lf(self, y):
    m = re.match('_answer \\( NV , (.*) \\)', y)
    if not m:
      print >> sys.stderr, 'Bad logical form "%s"' % y
      return None
    lf = m.group(1)
    m2 = re.match('\\( (.*) \\)', lf)
    if m2:
      lf = m2.group(1)
    return lf

  def get_nesting_alignments(self, x, y):
    alignments = []
    productions = []
    # alignments (root level with holes)
    for m in re.finditer('_const \\( (V0|NV) , _([a-z]*)id \\( ([^)]*) \\) \\)', y):
      varname = m.group(1)
      ent_type = m.group(2)
      ent_name = self.clean_name(m.group(3))
      y_span = (m.start(0), m.end(0))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'place': continue  # No expressions evaluating to place
      if '(' in y[y_span[1]:] or ',' in y[y_span[1]:]: continue  # Want _const at end of lf
      if ent_type == 'city' and '_' not in m.group(3): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      cat = '$%s-%s' % (ent_type, varname)
      alignments.append((cat, x_span, y_span))

    # productions (whole expressions)
    prod_type = self.extract_type(y)
    if prod_type:
      x_prod = self.prune_utterance(x)
      if x_prod:
        y_prod = self.prune_lf(y)
        if y_prod:
          productions.append(('$%s-%s' % (prod_type, 'V0'), x_prod, y_prod))
          y_prod_nv = y_prod.replace('V0', 'NV', 1)
          productions.append(('$%s-%s' % (prod_type, 'NV'), x_prod, y_prod_nv))
    return alignments, productions

  def format_lf(self, lf):
    # Strip underscores, collapse spaces when not inside quotation marks
    lf = self.postprocess_lf(lf)
    toks = []
    in_quotes = False
    quoted_toks = []
    for t in lf.split():
      if in_quotes:
        if t == "'":
          in_quotes = False
          toks.append('"%s"' % ' '.join(quoted_toks))
          quoted_toks = []
        else:
          quoted_toks.append(t)
      else:
        if t == "'":
          in_quotes = True
        else:
          if len(t) > 1 and t.startswith('_'):
            toks.append(t[1:])
          else:
            toks.append(t)
    lf = ''.join(toks)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      lf = lf + ')' * diff
    return lf

  def get_denotation(self, line):
    m = re.search('\{[^}]*\}', line)
    if m: 
      return m.group(0)
    else:
      return line.strip()

  def print_failures(self, dens, name):
    num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
    num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
    num_join_error = sum('Join failed syntactically' in d for d in dens)
    print '%s: %d syntax errors, %d executor errors' % (
        name, num_syntax_error, num_exec_error)

  def is_error(self, d):
    return 'FAILED' in d or 'Join failed syntactically' in d

  def compare_answers(self, true_answers, all_derivs):
    all_lfs = ([self.format_lf(s) for s in true_answers] +
               [self.format_lf(' '.join(d.y_toks)) 
                for x in all_derivs for d in x])
    tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
    tf = tempfile.NamedTemporaryFile(suffix='.dlog')
    for line in tf_lines:
      print >>tf, line
      print line
    tf.flush()
    msg = subprocess.check_output(['evaluator/geoquery', tf.name])
    tf.close()
    denotations = [self.get_denotation(line)
                   for line in msg.split('\n')
                   if line.startswith('        Example')]
    true_dens = denotations[:len(true_answers)]
    all_pred_dens = denotations[len(true_answers):]

    # Find the top-scoring derivation that executed without error
    derivs, pred_dens = pick_derivations(true_dens, all_pred_dens, all_derivs,
                                         self.is_error)
    self.print_failures(true_dens, 'gold')
    self.print_failures(pred_dens, 'predicted')
    for t, p in zip(true_dens, pred_dens):
      print '%s: %s == %s' % (t == p, t, p)
    return derivs, [t == p for t, p in zip(true_dens, pred_dens)]


class Seq2SQLDomain(Domain):
  #DEFAULT_TRAIN_FILE = os.path.join( 
  #    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
  #    'data/geo880/processed-lesscopy/geo880_train600.tsv')

  def preprocess_lf(self, lf):
    """Standardize variable names with De Brujin indices."""
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w.isalpha() and len(w) == 1:
        new_toks.append(w)
    return ' '.join(new_toks)

  def postprocess_lf(self, lf):
    """Undo the variable name standardization."""
    cur_var = chr(ord('A') - 1)
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
        new_toks.append(w)
    return ' '.join(new_toks)

  def clean_name(self, name):
    return name.split(',')[0].replace("'", '').strip()

  def get_entity_alignments(self, x, y):
    alignments = []
    return alignments

  def extract_type(self, y):
    y = self.postprocess_lf(y)  # undo de Brujin indexing
    return 'word'

  def prune_utterance(self, x):
    return x

  def prune_lf(self, y):
    return y

  def get_nesting_alignments(self, x, y):
    alignments = []
    productions = []
    # alignments (root level with holes)
    for m in re.finditer('[a-z]*', y):
      varname = m.group(1)
      ent_type = m.group(2)
      ent_name = self.clean_name(m.group(3))
      y_span = (m.start(0), m.end(0))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'place': continue  # No expressions evaluating to place
      if '(' in y[y_span[1]:] or ',' in y[y_span[1]:]: continue  # Want _const at end of lf
      if ent_type == 'city' and '_' not in m.group(3): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      cat = '$%s-%s' % (ent_type, varname)
      alignments.append((cat, x_span, y_span))

    # productions (whole expressions)
    prod_type = self.extract_type(y)
    if prod_type:
      x_prod = self.prune_utterance(x)
      if x_prod:
        y_prod = self.prune_lf(y)
        if y_prod:
          productions.append(('$%s-%s' % (prod_type, 'V0'), x_prod, y_prod))
          y_prod_nv = y_prod.replace('V0', 'NV', 1)
          productions.append(('$%s-%s' % (prod_type, 'NV'), x_prod, y_prod_nv))
    return alignments, productions

  def format_lf(self, lf):
    # Strip underscores, collapse spaces when not inside quotation marks
    lf = self.postprocess_lf(lf)
    return lf

  def get_denotation(self, line):
    return line.strip()

  def print_failures(self, dens, name):
    num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
    num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
    num_join_error = sum('Join failed syntactically' in d for d in dens)
    print '%s: %d syntax errors, %d executor errors' % (
        name, num_syntax_error, num_exec_error)

  def is_error(self, d):
    return 'FAILED' in d or 'Join failed syntactically' in d

  def compare_answers(self, true_answers, all_derivs):
    all_lfs = ([self.format_lf(s) for s in true_answers] +
               [self.format_lf(' '.join(d.y_toks)) 
                for x in all_derivs for d in x])
    tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
    tf = tempfile.NamedTemporaryFile(suffix='.dlog')
    for line in tf_lines:
      print >>tf, line
      print line
    tf.flush()
    msg = subprocess.check_output(['evaluator/geoquery', tf.name])
    tf.close()
    denotations = [self.get_denotation(line)
                   for line in msg.split('\n')
                   if line.startswith('        Example')]
    true_dens = denotations[:len(true_answers)]
    all_pred_dens = denotations[len(true_answers):]

    # Find the top-scoring derivation that executed without error
    derivs, pred_dens = pick_derivations(true_dens, all_pred_dens, all_derivs,
                                         self.is_error)
    self.print_failures(true_dens, 'gold')
    self.print_failures(pred_dens, 'predicted')
    for t, p in zip(true_dens, pred_dens):
      print '%s: %s == %s' % (t == p, t, p)
    return derivs, [t == p for t, p in zip(true_dens, pred_dens)]



class GeoqueryDomain(Domain):
  DEFAULT_TRAIN_FILE = os.path.join( 
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'data/geo880/processed-lesscopy/geo880_train600.tsv')

  def preprocess_lf(self, lf):
    """Standardize variable names with De Brujin indices."""
    cur_vars = []
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w.isalpha() and len(w) == 1:
        if w in cur_vars:
          ind_from_end = len(cur_vars) - cur_vars.index(w) - 1
          new_toks.append('V%d' % ind_from_end)
        else:
          cur_vars.append(w)
          new_toks.append('NV')
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def postprocess_lf(self, lf):
    """Undo the variable name standardization."""
    cur_var = chr(ord('A') - 1)
    toks = lf.split(' ')
    new_toks = []
    for w in toks:
      if w == 'NV':
        cur_var = chr(ord(cur_var) + 1)
        new_toks.append(cur_var)
      elif w.startswith('V'):
        ind = int(w[1:])
        new_toks.append(chr(ord(cur_var) - ind))
      else:
        new_toks.append(w)
    return ' '.join(new_toks)

  def clean_name(self, name):
    return name.split(',')[0].replace("'", '').strip()

  def get_entity_alignments(self, x, y):
    alignments = []
    for m in re.finditer('_([a-z]*id) \\( ([^)]*) \\)', y):
      ent_type = m.group(1)
      ent_name = self.clean_name(m.group(2))
      y_span = (m.start(2), m.end(2))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'cityid' and '_' not in m.group(2): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      alignments.append(('$' + ent_type, x_span, y_span))
    return alignments

  def extract_type(self, y):
    y = self.postprocess_lf(y)  # undo de Brujin indexing
    if '_city ( A )' in y or '_capital ( A )' in y:
      return 'city'
    elif '_river ( A )' in y:
      return 'river'
    elif '_state ( A )' in y:
      return 'state'
    return None

  def prune_utterance(self, x):
    patterns = [
        '\\?$',
        '^in',
        '^what is',
        '^what are',
        '^what',
        '^which',
        '^name',
        '^give me',
        '^can you tell me',
        '^show',
    ]
    old_x = x
    for pattern in patterns:
      x = re.sub(pattern, '', x).strip()
    if x == old_x or x == old_x[:-2]:
      # print >> sys.stderr, 'Could not prune: "%s"' % old_x
      return None
    return x

  def prune_lf(self, y):
    m = re.match('_answer \\( NV , (.*) \\)', y)
    if not m:
      print >> sys.stderr, 'Bad logical form "%s"' % y
      return None
    lf = m.group(1)
    m2 = re.match('\\( (.*) \\)', lf)
    if m2:
      lf = m2.group(1)
    return lf

  def get_nesting_alignments(self, x, y):
    alignments = []
    productions = []
    # alignments (root level with holes)
    for m in re.finditer('_const \\( (V0|NV) , _([a-z]*)id \\( ([^)]*) \\) \\)', y):
      varname = m.group(1)
      ent_type = m.group(2)
      ent_name = self.clean_name(m.group(3))
      y_span = (m.start(0), m.end(0))
      if ent_type == 'country': continue  # Only 1 country in geoquery...
      if ent_type == 'place': continue  # No expressions evaluating to place
      if '(' in y[y_span[1]:] or ',' in y[y_span[1]:]: continue  # Want _const at end of lf
      if ent_type == 'city' and '_' not in m.group(3): continue
      if ent_name + ' river' in x:
        ent_name = ent_name + ' river'
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      cat = '$%s-%s' % (ent_type, varname)
      alignments.append((cat, x_span, y_span))

    # productions (whole expressions)
    prod_type = self.extract_type(y)
    if prod_type:
      x_prod = self.prune_utterance(x)
      if x_prod:
        y_prod = self.prune_lf(y)
        if y_prod:
          productions.append(('$%s-%s' % (prod_type, 'V0'), x_prod, y_prod))
          y_prod_nv = y_prod.replace('V0', 'NV', 1)
          productions.append(('$%s-%s' % (prod_type, 'NV'), x_prod, y_prod_nv))
    return alignments, productions

  def format_lf(self, lf):
    # Strip underscores, collapse spaces when not inside quotation marks
    lf = self.postprocess_lf(lf)
    toks = []
    in_quotes = False
    quoted_toks = []
    for t in lf.split():
      if in_quotes:
        if t == "'":
          in_quotes = False
          toks.append('"%s"' % ' '.join(quoted_toks))
          quoted_toks = []
        else:
          quoted_toks.append(t)
      else:
        if t == "'":
          in_quotes = True
        else:
          if len(t) > 1 and t.startswith('_'):
            toks.append(t[1:])
          else:
            toks.append(t)
    lf = ''.join(toks)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      lf = lf + ')' * diff
    return lf

  def get_denotation(self, line):
    m = re.search('\{[^}]*\}', line)
    if m: 
      return m.group(0)
    else:
      return line.strip()

  def print_failures(self, dens, name):
    num_syntax_error = sum(d == 'Example FAILED TO PARSE' for d in dens)
    num_exec_error = sum(d == 'Example FAILED TO EXECUTE' for d in dens)
    num_join_error = sum('Join failed syntactically' in d for d in dens)
    print '%s: %d syntax errors, %d executor errors' % (
        name, num_syntax_error, num_exec_error)

  def is_error(self, d):
    return 'FAILED' in d or 'Join failed syntactically' in d

  def compare_answers(self, true_answers, all_derivs):
    all_lfs = ([self.format_lf(s) for s in true_answers] +
               [self.format_lf(' '.join(d.y_toks)) 
                for x in all_derivs for d in x])
    tf_lines = ['_parse([query], %s).' % lf for lf in all_lfs]
    tf = tempfile.NamedTemporaryFile(suffix='.dlog')
    for line in tf_lines:
      print >>tf, line
      print line
    tf.flush()
    msg = subprocess.check_output(['evaluator/geoquery', tf.name])
    tf.close()
    denotations = [self.get_denotation(line)
                   for line in msg.split('\n')
                   if line.startswith('        Example')]
    true_dens = denotations[:len(true_answers)]
    all_pred_dens = denotations[len(true_answers):]

    # Find the top-scoring derivation that executed without error
    derivs, pred_dens = pick_derivations(true_dens, all_pred_dens, all_derivs,
                                         self.is_error)
    self.print_failures(true_dens, 'gold')
    self.print_failures(pred_dens, 'predicted')
    for t, p in zip(true_dens, pred_dens):
      print '%s: %s == %s' % (t == p, t, p)
    return derivs, [t == p for t, p in zip(true_dens, pred_dens)]

def pick_derivations(true_dens, all_pred_dens, all_derivs, is_error_fn):
  # Find the top-scoring derivation that executed without error
  derivs = []
  pred_dens = []
  cur_start = 0
  for deriv_set in all_derivs:
    for i in range(len(deriv_set)):
      cur_denotation = all_pred_dens[cur_start + i]
      if not is_error_fn(cur_denotation):
        derivs.append(deriv_set[i])
        pred_dens.append(cur_denotation)
        break
    else:
      derivs.append(deriv_set[0])  # Default to first derivation
      pred_dens.append(all_pred_dens[cur_start])
    cur_start += len(deriv_set)
  return (derivs, pred_dens)

class AtisDomain(Domain):
  DEFAULT_TRAIN_FILE = os.path.join( 
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'data/atis/processed/atis_train.tsv')

  def standardize_varnames(self, ans):
    toks = ans.split(' ')
    varnames = {}
    new_toks = []
    for t in toks:
      if t == 'x' or t.startswith('$'):
        # t is a variable name
        if t in varnames:
          new_toks.append(varnames[t])
        else:
          new_varname = '$v%d' % len(varnames)
          varnames[t] = new_varname
          new_toks.append(new_varname)
      else:
        new_toks.append(t)
    lf = ' '.join(new_toks)
    return lf

  def sort_args(self, ans):
    lisp_tree = to_lisp_tree(ans)
    if lisp_tree is None: return ans

    # Post-order traversal, sorting as we go
    def recurse(node):
      if isinstance(node, str): return
      for child in node:
        recurse(child)
      if node[0] in ('_and', '_or'):
        node[1:] = sorted(node[1:], key=lambda x: str(x))
    recurse(lisp_tree)

    def tree_to_str(node):
      if isinstance(node, str):
        return node
      else:
        return '( %s )' % ' '.join(tree_to_str(child) for child in node)
    return tree_to_str(lisp_tree)

  def normalize(self, ans):
    """Perform some normalization of ATIS logical forms

    1. Standardize variable names, since atis_test uses different convention.
    2. Sort arguments to commutative operators such as _and() and _or().
    3. Balance parentheses.
    """
    lf = self.standardize_varnames(ans)
    # Balance parentheses
    # In particular, atis_test gold lf's have unbalanced parens...
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      lf = lf + ' )' * diff
    lf = self.sort_args(lf)
    return lf

  def compare_answers(self, true_answers, all_derivs):
    derivs = [x[0] for x in all_derivs]
    pred_answers = [' '.join(d.y_toks) for d in derivs]
    is_correct_list = []
    for true_ans, pred_ans in zip(true_answers, pred_answers):
      is_correct_list.append(
          self.normalize(true_ans) == self.normalize(pred_ans))
    return derivs, is_correct_list

  def get_entity_alignments(self, x, y):
    # Cache lexicon
    if not self.lex:
      self.lex = atislexicon.get_lexicon()
    alignments = []
    x_toks = x.split(' ')
    running_lens = [0] * (len(x_toks) + 1) # index to use if start at token i
    for i in range(1, len(x_toks) + 1):
      running_lens[i] = running_lens[i-1] + 1 + len(x_toks[i-1])
    y_toks = y.split(' ')
    lex_items = self.lex.map_over_sentence(x_toks, return_entries=True)
    lex_ents = [a[1] for a in lex_items]
    for (i, j), ent in lex_items:
      # Make sure this entity occurs exactly once in lexicon entries
      # and in the logical form
      x_span = (running_lens[i], running_lens[j] - 1)
      ent_type = ent.split(':')[1][1:]
      cat = '$' + ent_type
      if lex_ents.count(ent) != 1: continue
      if y_toks.count(ent) != 1: continue
      y_ind = y.index(ent)
      y_span = (y_ind, y_ind + len(ent))
      alignments.append((cat, x_span, y_span))
    return alignments

class OvernightDomain(Domain):
  DEFAULT_TRAIN_FILE = os.path.join( 
      os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
      'data/overnight/processed-lf/calendar_train.tsv')

  def __init__(self, subdomain):
    self.subdomain = subdomain

  def format_lf(self, lf):
    lf = self.postprocess_lf(lf)
    replacements = [
        ('! ', '!'),
        ('SW', 'edu.stanford.nlp.sempre.overnight.SimpleWorld'),
    ]
    for a, b in replacements:
      lf = lf.replace(a, b)
    # Balance parentheses
    num_left_paren = sum(1 for c in lf if c == '(')
    num_right_paren = sum(1 for c in lf if c == ')')
    diff = num_left_paren - num_right_paren
    if diff > 0:
      while len(lf) > 0 and lf[-1] == '(' and diff > 0:
        lf = lf[:-1]
        diff -= 1
      if len(lf) == 0: return ''
      lf = lf + ' )' * diff
    return lf

  def is_error(self, d):
    return 'BADJAVA' in d or 'ERROR' in d or d == 'null'

  def compare_answers(self, true_answers, all_derivs):
    all_lfs = ([self.format_lf(s) for s in true_answers] +
               [self.format_lf(' '.join(d.y_toks))
                for x in all_derivs for d in x])
    tf_lines = all_lfs
    tf = tempfile.NamedTemporaryFile(suffix='.examples')
    for line in tf_lines:
      print >> tf, line
      print line
    tf.flush()
    msg = subprocess.check_output(['evaluator/overnight', self.subdomain, tf.name])
    tf.close()
    denotations = [line.split('\t')[1] for line in msg.split('\n')
                   if line.startswith('targetValue\t')]
    true_dens = denotations[:len(true_answers)]
    all_pred_dens = denotations[len(true_answers):]
    derivs, pred_dens = pick_derivations(true_dens, all_pred_dens, all_derivs,
                                         self.is_error)
    for t, p in zip(true_dens, pred_dens):
      print '%s: %s == %s' % (t == p, t, p)
    return derivs, [t == p for t, p in zip(true_dens, pred_dens)]

  def clean_name(self, name):
    new_chars = []
    prev_c = None
    for c in name:
      if prev_c and ((prev_c.isalpha() and c.isdigit()) or
                     prev_c.isdigit() and c.isalpha()):
        new_chars.append(' ')
      if c == '_':
        new_chars.append(' ')
      else:
        new_chars.append(c)
      prev_c = c
    return ''.join(new_chars)

  def get_entity_alignments(self, x, y):
    alignments = []
    for m in re.finditer('en\\.([^. ]+)\\.([^ ]+)', y):
      ent_type = m.group(1)
      ent_name = self.clean_name(m.group(2))
      y_span = (m.start(0), m.end(0))
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      alignments.append(('$' + ent_type + '-id', x_span, y_span))
    return alignments

  def extract_type(self, y):
    y_toks = y.split(' ')
    i = 3
    while True:
      cur_call = ' '.join(y_toks[i:i+3])
      if cur_call == '( call SW.filter':
        i += 3
      elif cur_call == '( call SW.getProperty':
        if (' '.join(y_toks[i+3:i+6]) == '( call SW.singleton' and
            ' '.join(y_toks[i+7:i+14]) == ') ( string ! type ) )'):
          return y_toks[i+6][3:]
        else:
          return None
      else:
        return None

  def prune_utterance(self, x):
    patterns = [
        'show me ',
        'what is ',
        'what are ',
        'what ',
        'which ',
        'name ',
        'give me ',
        'can you tell me ',
        'show ',
        'find me ',
        'find ',
        'select ',
        'choose a',
        'choose ',
        'list all ',
    ]
    old_x = x
    for pattern in patterns:
      x = re.sub('^' + pattern, '', x).strip()
    if x == old_x or x == old_x[:-2]:
      print >> sys.stderr, 'Could not prune: "%s"' % old_x
      #return None
    return x

  def prune_lf(self, y):
    y_toks = y.split(' ')
    if ' '.join(y_toks[:3]) != '( call SW.listValue':
      raise ValueError('Unexpected logical form ' + y)
    return ' '.join(y_toks[3:-1])

  def get_nesting_alignments(self, x, y):
    alignments = []
    productions = []

    # alignments (root level with holes)
    for m in re.finditer('en\\.([^. ]+)\\.([^ ]+)', y):
      ent_type = m.group(1)
      ent_name = self.clean_name(m.group(2))
      y_span = (m.start(0), m.end(0))
      ind = x.find(ent_name)
      if ind == -1: continue
      x_span = (ind, ind + len(ent_name))
      alignments.append(('$' + ent_type, x_span, y_span))
      # Throw in these as productions too
      # Since some types don't have any large expression proudctions
      productions.append(('$' + ent_type, ent_name, m.group(0)))

    # productions (whole expressions)
    prod_type = self.extract_type(y)
    if prod_type:
      x_prod = self.prune_utterance(x)
      if x_prod:
        y_prod = self.prune_lf(y)
        if y_prod:
          productions.append(('$' + prod_type, x_prod, y_prod))

    return alignments, productions

class ArtificialDomain(Domain):
  def get_entity_alignments(self, x, y):
    print (x, y)
    x_ind = x.find('ent')
    y_ind = y.find('_ent')
    if x_ind == -1 or y_ind == -1: return []
    return [('$ent', (x_ind, x_ind + 6), (y_ind, y_ind + 7))]

  def get_nesting_alignments(self, x, y):
    x_ind = x.find('ent')
    y_ind = y.find('_ent')
    alignments = [('$phrase', (x_ind, x_ind + 6), (y_ind, y_ind + 7))]
    productions = [('$phrase', x, y)]
    return alignments, productions

def to_lisp_tree(expr):
  toks = expr.split(' ')
  def recurse(i):
    if toks[i] == '(':
      subtrees = []
      j = i+1
      while True:
        subtree, j = recurse(j)
        subtrees.append(subtree)
        if toks[j] == ')':
          return subtrees, j + 1
    else:
      return toks[i], i+1

  try:
    lisp_tree, final_ind = recurse(0)
    return lisp_tree
  except Exception as e:
    print >> sys.stderr, 'Failed to convert "%s" to lisp tree' % expr
    print e
    return None

def new(domain_name):
  if domain_name == 'geoquery':
    return GeoqueryDomain()
  elif domain_name == 'atis':
    return AtisDomain()
  elif domain_name.startswith('overnight'):
    subdomain = domain_name.split('-')[1]
    return OvernightDomain(subdomain)
  elif domain_name == 'artificial':
    return ArtificialDomain()
  elif domain_name =='seq2sql':
      return Seq2SQLDomain()
  elif domain_name =='toy':
      return ToyDomain()
  elif domain_name =='mt':
      return MTDomain()
  else:
    raise ValueError('Unrecognized domain "%s"' % domain_name)

def test_process_lf(domain):
  num_diffs = 0
  with open(domain.DEFAULT_TRAIN_FILE) as f:
    dataset = [tuple(line.rstrip('\n').split('\t')) for line in f]
  for i, (x, y) in enumerate(dataset):
    y_pre = domain.preprocess_lf(y)
    y_post = domain.postprocess_lf(y_pre)
    print 'Example %d' % i
    print '  x      = "%s"' % x
    print '  y      = "%s"' % y
    print '  y_pre  = "%s"' % y_pre
    print '  y_post = "%s"' % y_post
    print '  y == y_post: %s' % (y == y_post)
    print
    if y != y_post:
      num_diffs += 1
  print 'Found %d cases where y != y_post' % num_diffs

def test_get_entity_alignments(domain):
  with open(domain.DEFAULT_TRAIN_FILE) as f:
    dataset = [tuple(line.rstrip('\n').split('\t')) for line in f]
  for i, (x, y) in enumerate(dataset):
    y = domain.preprocess_lf(y)
    print 'Example %d' % i
    print '  x      = "%s"' % x
    print '  y      = "%s"' % y
    for cat, x_span, y_span in domain.get_entity_alignments(x, y):
      print '  %s -> ("%s", "%s")' % (cat, x[x_span[0]:x_span[1]], y[y_span[0]:y_span[1]])

def test_get_nesting_alignments(domain):
  with open(domain.DEFAULT_TRAIN_FILE) as f:
    dataset = [tuple(line.rstrip('\n').split('\t')) for line in f]
  for i, (x, y) in enumerate(dataset):
    y = domain.preprocess_lf(y)
    print 'Example %d' % i
    print '  x      = "%s"' % x
    print '  y      = "%s"' % y
    alignments, productions = domain.get_nesting_alignments(x, y)
    for cat, x_span, y_span in alignments:
      print '  align: %s -> ("%s", "%s")' % (cat, x[x_span[0]:x_span[1]], y[y_span[0]:y_span[1]])
    for cat, x_str, y_str in productions:
      print '  prod : %s -> ("%s", "%s")' % (cat, x_str, y_str)


def main():
  if len(sys.argv) < 3:
    print >> sys.stderr, 'Usage: %s domain [process_lf]' % sys.argv[0]
    sys.exit(1)
  domain_name = sys.argv[1]
  method = sys.argv[2]
  domain = new(domain_name)
  if method == 'process_lf':
    test_process_lf(domain)
  elif method == 'get_entity_alignments':
    test_get_entity_alignments(domain)
  elif method == 'get_nesting_alignments':
    test_get_nesting_alignments(domain)
  else:
    print >> sys.stderr,'Unrecognized method "%s"' % method

if __name__ == '__main__':
  main()
