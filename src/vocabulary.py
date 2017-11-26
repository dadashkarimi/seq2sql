"""A vocabulary for a neural model."""
import collections
import numpy
import os
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

GLOVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data/glove')

class Vocabulary:
  """A vocabulary of words, and their embeddings.
  
  By convention, the end-of-sentence token '</s>' is 0, and
  the unknown word token 'UNK' is 1.
  """
  END_OF_SENTENCE = '</s>'
  END_OF_SENTENCE_INDEX = 0
  UNKNOWN = 'UNK'
  UNKNOWN_INDEX = 1
  NUM_SPECIAL_SYMBOLS = 2

  def __init__(self, word_list, emb_size, use_glove=False, float_type=numpy.float64,
               unk_cutoff=0):
    """Create the vocabulary. 

    Args:
      word_list: List of words that occurred in the training data.
      emb_size: dimension of word embeddings
      use_glove: Whether to initialize with GloVe vectors.
      float_type: numpy float type for theano
    """
    self.word_list = [self.END_OF_SENTENCE, self.UNKNOWN] + word_list
    self.word_to_index = dict((x[1], x[0]) for x in enumerate(self.word_list))
    self.emb_size = emb_size
    self.float_type = float_type

    # Embedding matrix
    init_val = 0.1 * numpy.random.uniform(-1.0, 1.0, (self.size(), emb_size)).astype(theano.config.floatX)

    # Initialize with GloVe
    if use_glove:
      glove_file = os.path.join(GLOVE_DIR, 'glove.6B.%dd.txt' % emb_size)
      with open(glove_file) as f:
        for line in f:
          toks = line.split(' ')
          word = toks[0]
          if word not in self.word_to_index: continue
          ind = self.word_to_index[word]
          vec = numpy.array([float(x) for x in toks[1:]])
          init_val[ind] = vec
          print 'Found GloVe vector for "%s": %s' % (word, str(vec))

    self.emb_mat = theano.shared(
        name='vocab_emb_mat',
        value=init_val)

  def get_theano_embedding(self, i):
    """Get theano embedding for given word index."""
    return self.emb_mat[i]

  def get_theano_params(self):
    """Get theano parameters to back-propagate through."""
    return [self.emb_mat]

  def get_theano_all(self):
    """By default, same as self.get_theano_params()."""
    return self.get_theano_params()

  def get_index(self, word):
    if word in self.word_to_index:
      return self.word_to_index[word]
    return self.word_to_index[self.UNKNOWN]

  def get_word(self, i):
    return self.word_list[i]

  def sentence_to_indices(self, sentence, add_eos=True):
    words = sentence.split(' ')
    if add_eos:
      words.append(self.END_OF_SENTENCE)
    indices = [self.get_index(w) for w in words]
    return indices

  def indices_to_sentence(self, indices, strip_eos=False):
    return ' '.join(self.word_list[i] for i in indices)

  def size(self):
    return len(self.word_list)

  @classmethod
  def from_sentences(cls, sentences, emb_size, unk_cutoff=0, **kwargs):
    """Get list of all words used in a list of sentences.
    
      Args:
        sentences: list of sentences
        emb_size: size of embedding
        unk_cutoff: Treat words with <= this many occurrences as UNK.
    """
    counts = collections.Counter()
    for s in sentences:
      counts.update(s.split(' '))
    word_list = [w for w in counts if counts[w] > unk_cutoff]
    print 'Extracted vocabulary of size %d' % len(word_list)
    return cls(word_list, emb_size, **kwargs)
