"""A single example in a dataset."""
import lexicon

class Example(object):
  """A single example in a dataset.

  Basically a struct after it's initialized, with the following fields:
    - self.x_str, self.y_str: input/output as single space-separated strings
    - self.x_toks, self.y_toks: input/output as list of strings
    - self.input_vocab, self.output_vocab: Vocabulary objects
    - self.x_inds, self.y_inds: input/output as indices in corresponding vocab
    - self.copy_toks: list of length len(x_toks), having tokens that should
        be generated if copying is performed.
    - self.y_in_x_inds: ji-th entry is whether copy_toks[i] == y_toks[j].

  Treat these objects as read-only.
  """
  def __init__(self, x_str, y_str, input_vocab, output_vocab, lex,
               reverse_input=False):
    """Create an Example object.
    
    Args:
      x_str: Input sequence as a space-separated string
      y_str: Output sequence as a space-separated string
      input_vocab: Vocabulary object for input
      input_vocab: Vocabulary object for output
      reverse_input: If True, reverse the input.
    """
    self.x_str = x_str  # Don't reverse this, used just for printing out
    self.y_str = y_str
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.lex = lex
    self.reverse_input = reverse_input
    self.x_toks = x_str.split(' ')
    if reverse_input:
      self.x_toks = self.x_toks[::-1]
    self.y_toks = y_str.split(' ')
    self.input_vocab = input_vocab
    self.output_vocab = output_vocab
    self.x_inds = input_vocab.sentence_to_indices(x_str)
    if reverse_input:
      self.x_inds = self.x_inds[::-1]
    self.y_inds = output_vocab.sentence_to_indices(y_str)

    if lex:
      entities = lex.map_over_sentence(self.x_str.split(' '))
      self.copy_toks = [x if x else '<COPY>' for x in entities]
      if reverse_input:
        self.copy_toks = self.copy_toks[::-1]
    else:
      self.copy_toks = [lexicon.strip_unk(w) for w in self.x_toks]

    self.y_in_x_inds = ([
        [int(x_tok == y_tok) for x_tok in self.copy_toks] + [0]
        for y_tok in self.y_toks
        ] + [[0] * (len(self.x_toks) + 1)])

    self.y_in_src_inds = ([
        [int(x_tok == y_tok) for x_tok in self.input_vocab.word_list]
        for y_tok in self.y_toks
        ] + [[0] * (input_vocab.size())])
