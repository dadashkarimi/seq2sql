"""An LSTM layer."""
import numpy
import random
import sys
import theano
from theano.ifelse import ifelse
from theano import tensor as T

from rnnlayer import RNNLayer

class ATNHLayer(RNNLayer):
  """An ATNH layer.

  Parameter names follow convention in Richard Socher's CS224D slides.
  """
  def create_vars(self, create_init_state):
    # Initial state
    # The hidden state must store both c_t, the memory cell, 
    # and h_t, what we normally call the hidden state
    if create_init_state:
      self.h0 = theano.shared(
          name='h0', 
          value=0.1 * numpy.random.uniform(-1.0, 1.0, 2 * self.nh).astype(theano.config.floatX))
      init_state_params = [self.h0]
    else:
      init_state_params = []

    # Recurrent layer 
    self.wi = theano.shared(
        name='wi',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.ui = theano.shared(
        name='ui',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wf = theano.shared(
        name='wf',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.uf = theano.shared(
        name='uf',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wo = theano.shared(
        name='wo',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.uo = theano.shared(
        name='uo',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    self.wc = theano.shared(
        name='wc',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.de, self.nh)).astype(theano.config.floatX))
    self.uc = theano.shared(
        name='uc',
        value=0.1 * numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)).astype(theano.config.floatX))
    recurrence_params = [
        self.wi, self.ui, self.wf, self.uf,
        self.wo, self.uo, self.wc, self.uc,
    ]

    # Params
    self.params = init_state_params + recurrence_params

  def unpack(self, hidden_state):
    c_t = hidden_state[0:self.nh]
    h_t = hidden_state[self.nh:]
    return (c_t, h_t)

  def pack(self, c_t, h_t):
    return T.concatenate([c_t, h_t])

  def get_init_state(self):
    return self.h0

  def step(self, input_t, c_h_prev):
    c_prev, h_prev = self.unpack(c_h_prev)
    i_t = T.nnet.sigmoid(T.dot(input_t, self.wi) + T.dot(h_prev, self.ui))
    f_t = T.nnet.sigmoid(T.dot(input_t, self.wf) + T.dot(h_prev, self.uf))
    o_t = T.nnet.sigmoid(T.dot(input_t, self.wo) + T.dot(h_prev, self.uo))
    c_tilde_t = T.tanh(T.dot(input_t, self.wc) + T.dot(h_prev, self.uc))
    c_t = f_t * c_prev + i_t * c_tilde_t
    h_t = o_t * T.tanh(c_t)
    return self.pack(c_t, h_t)

  def get_h_for_write(self, c_h_t):
    c_t, h_t = self.unpack(c_h_t)
    return h_t

