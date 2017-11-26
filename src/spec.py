"""Specifies a particular instance of a model."""
import numpy
import dill as pickle

from gru import GRULayer
from atnh import ATNHLayer
from lstm import LSTMLayer
from vanillarnn import VanillaRNNLayer

RNN_TYPES=['vanillarnn', 'gru', 'lstm','atnh']

class Spec(object):
  """Abstract class for a specification of a sequence-to-sequence RNN model.

  Concrete sublcasses must implement the following methods:
  - self.create_vars(): called by __init__, should initialize parameters.
  - self.get_local_params(): Get all local parameters (excludes vocabulary).
  """
  def __init__(self, in_vocabulary, out_vocabulary, lexicon, hidden_size,
               rnn_type='lstm', step_rule='simple', **kwargs):
    """Initialize.

    Args:
      in_vocabulary: list of words in the vocabulary of the input
      out_vocabulary: list of words in the vocabulary of the output
      embedding_dim: dimension of word vectors
      hidden_size: dimension of hidden layer
    """
    self.in_vocabulary = in_vocabulary
    self.out_vocabulary = out_vocabulary
    self.lexicon = lexicon
    self.hidden_size = hidden_size
    self.rnn_type = rnn_type
    self.step_rule = step_rule
    self.create_vars()
    self._process_init_kwargs(**kwargs)

  def _process_init_kwargs(self, **kwargs):
    """Optionally override this to process special kwargs at __init__()."""
    pass

  def set_in_vocabulary(self, in_vocabulary):
    self.in_vocabulary = in_vocabulary

  def set_out_vocabulary(self, out_vocabulary):
    self.out_vocabulary = out_vocabulary

  def create_vars(self):
    raise NotImplementedError

  def get_local_params(self):
    raise NotImplementedError

  def f_read_embedding(self, i):
    return self.in_vocabulary.get_theano_embedding(i)

  def f_write_embedding(self, i):
    return self.out_vocabulary.get_theano_embedding(i)

  def get_params(self):
    """Get all parameters (things we optimize with respect to)."""
    params = (self.get_local_params()
              + self.in_vocabulary.get_theano_params()
              + self.out_vocabulary.get_theano_params())
    return params

  def get_all_shared(self):
    """Get all shared theano varaibles.

    There are shared variables that we do not necessarily optimize respect to,
    but may be held fixed (e.g. GloVe vectors, if desired).
    We don't backpropagate through these, but we do need to feed them to scan.
    """
    params = (self.get_local_params() 
              + self.in_vocabulary.get_theano_all()
              + self.out_vocabulary.get_theano_all())
    return params

  def create_rnn_layer(self, hidden_dim, input_dim, vocab_size, is_encoder):
    if self.rnn_type == 'vanillarnn':
      return VanillaRNNLayer(hidden_dim, input_dim, vocab_size,
                             create_init_state=is_encoder)
    elif self.rnn_type == 'gru':
      return GRULayer(hidden_dim, input_dim, vocab_size,
                      create_init_state=is_encoder)
    elif self.rnn_type == 'lstm':
      return LSTMLayer(hidden_dim, input_dim, vocab_size,
              create_init_state=is_encoder)
    elif self.rnn_type == 'atnh':
      return LSTMLayer(hidden_dim, input_dim, vocab_size,
                       create_init_state=is_encoder)

    raise Exception('Unrecognized rnn_type %s' % self.rnn_type)

  def save(self, filename):
    """Save the parameters to a filename."""
    with open(filename, 'w') as f:
      pickle.dump(self, f)

def load(filename):
  with open(filename) as f:
    return pickle.load(f)
