"""A generic continuous neural sequence-to-sequence model."""
import collections
import itertools
import numpy as np
import os
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T
import time

from example import Example
from vocabulary import Vocabulary

CLIP_THRESH = 3.0  # Clip gradient if norm is larger than this
NESTEROV_MU = 0.95  # mu for Nesterov momentum

class NeuralModel(object):
  """A generic continuous neural sequence-to-sequence model.

  Implementing classes must implement the following functions:
    - self.setup(): set up the model.
    - self.get_objective_and_gradients(x, y): Get objective and gradients.
    - self.decode_greedy(ex, max_len=100): Do a greedy decoding of x, predict y.
    - self.decode_greedy(ex, beam_size=1, max_len=100): Beam search to predict y

  They should also override
    - cls.get_spec_class(): The associated Spec subclass for this NeuralModel.

  Convention used by this class:
    nh: dimension of hidden layer
    nw: number of words in the vocabulary
    de: dimension of word embeddings
  """
  def __init__(self, spec, distract_num=0, float_type=np.float64):
    """Initialize.

    Args:
      spec: Spec object.
      float_type: Floating point type (default 64-bit/double precision)
    """
    self.spec = spec
    self.in_vocabulary = spec.in_vocabulary
    self.out_vocabulary = spec.out_vocabulary
    self.lexicon = spec.lexicon
    self.distract_num=distract_num
    self.float_type = float_type
    self.params = spec.get_params()
    if spec.step_rule in ('adagrad', 'rmsprop', 'nesterov'):
      # Initialize the cache for grad norms (adagrad, rmsprop) 
      # or velocity (Nesterov momentum)
      self.grad_cache = [
          theano.shared(
              name='%s_grad_cache' % p.name,
              value=np.zeros_like(p.get_value()))
          for p in self.params]
    self.all_shared = spec.get_all_shared()
    self.setup()
    print >> sys.stderr, 'Setup complete.'

  @classmethod
  def get_spec_class(cls):
    raise NotImplementedError

  def setup(self):
    """Do all necessary setup (e.g. compile theano functions)."""
    raise NotImplementedError

  def sgd_step(self, ex, eta, l2_reg, distractors=None):
    """Perform an SGD step.

    This is a default implementation, which assumes
    a function called self._backprop() which updates parameters.

    Override if you need a more complicated way of computing the 
    objective or updating parameters.

    Returns: the current objective value
    """
    print 'x: %s' % ex.x_str
    print 'y: %s' % ex.y_str
    if distractors: 
      print("TSET")
      #for ex_d in distractors:
      #  print 'd: %s' % ex_d.x_str
      #x_inds_d_all = [ex_d.x_inds for ex_d in distractors]
      #info = self._backprop_distract(
      #    ex.x_inds, ex.y_inds, eta, ex.y_in_x_inds, l2_reg, *x_inds_d_all)
    else:
      #info = self._backprop(ex.x_inds, ex.y_inds, eta, ex.y_in_x_inds, l2_reg)
      #dec_init_state, annotations = self._get_dec_annot(ex.x_inds)
      h_for = self.h_for(ex.x_inds)
      c_for = self.c_for(ex.x_inds)
      scores = self.get_scores(ex.x_inds)
      #w_for = self.get_write(ex.x_inds)
      #print("shape of write: {}\n".format(w_for.shape))
      #print("scores size: {}".format(len(scores)))
      #print("scores size: {}".format(len(scores[0])))
      print("h_for_write size: {}\n".format(np.array(h_for).shape))
      print("c_t size: {}\n".format(np.array(c_for).shape))
      print("scores size: {}\n".format(np.array(scores).shape))
      #print(dec_init_state)
      last_state = self._get_fwd_states(ex.x_inds)
      #print(last_state)
      #print("dec_size: {}, annot size: {} \n".format(dec_init_state.shape, annotations.shape))
      #print("cur_y_in_x:{}".format(len(self._get_y_in_x_shape(ex.x_inds))))
      #print("cur_y_in_x:{}".format(len(self._get_y_in_x_shape(ex.x_inds)[0])))
      #print("copying_p_y_t:{}".format(self._get_copying_p_y_t_shape(ex.y_in_x_inds).shape))

    p_y_seq = info[0]
    objective = info[1]
    print 'P(y_i): %s' % p_y_seq
    return objective

  def decode_greedy(self, ex, max_len=100):
    """Decode input greedily.
    
    Returns list of (prob, y_tok_seq) pairs."""
    raise NotImplementedError

  def decode_beam(self, ex, beam_size=1, max_len=100):
    """Decode input with beam search.
    
    Returns list of (prob, y_tok_seq) pairs."""
    raise NotImplementedError

  def on_train_epoch(self, t):
    """Optional method to do things every epoch."""
    for p in self.params:
      print '%s: %s' % (p.name, p.get_value())

  def train(self, dataset, eta=0.1, T=[], verbose=False, dev_data=None,
            l2_reg=0.0, distract_num = 0, distract_prob=0.0,
            concat_num=1, concat_prob=0.0, augmenter=None, aug_frac=0.0):
    # train with SGD (batch size = 1)
    # 
    cur_lr = eta
    max_iters = sum(T)
    lr_changes = set([sum(T[:i]) for i in range(1, len(T))])
    for it in range(max_iters):
      t0 = time.time()
      if it in lr_changes:
        # Halve the learning rate
        cur_lr = 0.5 * cur_lr
      total_nll = 0.0
      random.shuffle(dataset)
      cur_dataset = dataset
      if augmenter:
        # Do data augmentation on the fly
        aug_num = int(round(aug_frac * len(dataset)))
        aug_exs = [Example(
            x, y, dataset[0].input_vocab, dataset[0].output_vocab, 
            dataset[0].lex, reverse_input=dataset[0].reverse_input)
            for x, y in augmenter.sample(aug_num)]
        cur_dataset = cur_dataset + aug_exs
        random.shuffle(cur_dataset)
      if concat_num > 1:
        # Generate new concatenated examples on the fly
        num_concat_exs = int(round(len(cur_dataset) * concat_prob / concat_num)) * concat_num
        normal_exs = cur_dataset[num_concat_exs:]
        concat_exs = []
        for i in range(num_concat_exs / concat_num):
          cur_exs = cur_dataset[i*concat_num:(i+1)*concat_num]
          new_x_str = (' ' + Vocabulary.END_OF_SENTENCE + ' ').join(
              ex.x_str for ex in cur_exs)
          new_y_str = (' ' + Vocabulary.END_OF_SENTENCE + ' ').join(
              ex.y_str for ex in cur_exs)
          new_ex = Example(new_x_str, new_y_str, dataset[0].input_vocab,
                           dataset[0].output_vocab, dataset[0].lex,
                           reverse_input=dataset[0].reverse_input)
          concat_exs.append(new_ex)
        cur_dataset = concat_exs + normal_exs
        random.shuffle(cur_dataset)

      for ex in cur_dataset:
        do_distract = distract_num > 0 and random.random() < distract_prob
        if do_distract:
          distractors = random.sample(dataset, distract_num)
          nll = self.sgd_step(ex, cur_lr, l2_reg, distractors=distractors)
        else:
          nll = self.sgd_step(ex, cur_lr, l2_reg)
          total_nll += nll
      dev_nll = 0.0
      if dev_data:
        for ex in dev_data:
          dev_nll += self._get_nll(ex.x_inds, ex.y_inds, ex.y_in_x_inds)
      self.on_train_epoch(it)
      t1 = time.time()
      print 'NeuralModel.train(): iter %d (lr = %g): train obj = %g, dev nll = %g (%g seconds)' % (
          it, cur_lr, total_nll, dev_nll, t1 - t0)
