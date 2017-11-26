"""An output layer."""
import numpy
import theano
from theano.ifelse import ifelse
import theano.tensor as T

class OutputLayer(object):
  """Class that sepcifies parameters of an output layer.
  
  Conventions used by this class (shared with spec.py):
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """ 
  def __init__(self, vocab, hidden_size):
    self.vocab = vocab
    self.de = vocab.emb_size
    self.nh = hidden_size
    self.nw = vocab.size()
    self.create_vars()

  def create_vars(self):
    self.w_out = theano.shared(
        name='w_out', 
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nw, self.nh)).astype(theano.config.floatX))
        # Each row is one word
    self.params = [self.w_out]

  def write(self, h_t, attn_scores=None):
    """Get a distribution over words to write.
    
    Entries in [0, nw) are probablity of emitting i-th output word,
    and entries in [nw, nw + len(attn_scores))
    are probability of copying the (i - nw)-th word.

    Args:
      h_t: theano vector representing hidden state
      attn_scores: unnormalized scores from the attention module, if doing 
          attention-based copying.
    """
    if attn_scores:
      scores = T.dot(h_t, self.w_out.T)
      return T.nnet.softmax(T.concatenate([scores, attn_scores]))[0] #p(aj=write[w]|x,y) = exp(Uw[sj,cj]) 
    else:
      return T.nnet.softmax(T.dot(h_t, self.w_out.T))[0] # p(aj=copy[i]) = exp(eji)
