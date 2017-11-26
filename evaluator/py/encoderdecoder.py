"""A basic encoder-decoder model."""
import itertools
import numpy
import theano
from theano import tensor as T
from theano.ifelse import ifelse
import sys

from encdecspec import EncoderDecoderSpec
from derivation import Derivation
from neural import NeuralModel, CLIP_THRESH
from vocabulary import Vocabulary

class EncoderDecoderModel(NeuralModel):
  """An encoder-decoder RNN model."""
  def setup(self):
    self.setup_encoder()
    self.setup_decoder_step()
    self.setup_decoder_write()
    self.setup_backprop()

  @classmethod
  def get_spec_class(cls):
    return EncoderDecoderSpec

  def setup_encoder(self):
    """Run the encoder.  Used at test time."""
    x = T.lvector('x_for_enc')
    def recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)
    results, _ = theano.scan(recurrence, sequences=[x],
                             outputs_info=[self.spec.get_init_state()],
                             non_sequences=self.spec.get_all_shared())
    h_last = results[-1]
    self._encode = theano.function(inputs=[x], outputs=h_last)

  def setup_decoder_step(self):
    """Advance the decoder by one step.  Used at test time."""
    y_t = T.lscalar('y_t_for_dec')
    h_prev = T.vector('h_prev_for_dec')
    h_t = self.spec.f_dec(y_t, h_prev)
    self._decoder_step = theano.function(inputs=[y_t, h_prev], outputs=h_t)

  def setup_decoder_write(self):
    """Get the write distribution of the decoder.  Used at test time."""
    h_prev = T.vector('h_prev_for_write')
    h_for_write = self.spec.decoder.get_h_for_write(h_prev)
    write_dist = self.spec.f_write(h_for_write)
    self._decoder_write = theano.function(inputs=[h_prev], outputs=write_dist)

  def setup_backprop(self):
    eta = T.scalar('eta_for_backprop')
    x = T.lvector('x_for_backprop')
    y = T.lvector('y_for_backprop')
    y_in_x_inds = T.lmatrix('y_in_x_inds_for_backprop')
    l2_reg = T.scalar('l2_reg_for_backprop')
    def enc_recurrence(x_t, h_prev, *params):
      return self.spec.f_enc(x_t, h_prev)
    enc_results, _ = theano.scan(fn=enc_recurrence,
                                 sequences=[x],
                                 outputs_info=[self.spec.get_init_state()],
                                 non_sequences=self.spec.get_all_shared())
    h_last = enc_results[-1]
    
    def decoder_recurrence(y_t, h_prev, *params):
      h_for_write = self.spec.decoder.get_h_for_write(h_prev)
      write_dist = self.spec.f_write(h_for_write)
      p_y_t = write_dist[y_t]
      h_t = self.spec.f_dec(y_t, h_prev)
      return (h_t, p_y_t)
    dec_results, _ = theano.scan(
        fn=decoder_recurrence, sequences=[y],
        outputs_info=[h_last, None],
        non_sequences=self.spec.get_all_shared())
    p_y_seq = dec_results[1]
    log_p_y = T.sum(T.log(p_y_seq))
    nll = -log_p_y
    # Add L2 regularization
    objective = nll 
    for p in self.params:
      objective += l2_reg / 2 * T.sum(p**2)
    gradients = T.grad(objective, self.params)
    self._get_nll = theano.function(
        inputs=[x, y, y_in_x_inds], outputs=nll, on_unused_input='warn')

    # Do the updates here
    updates = []
    for p, g in zip(self.params, gradients):
      grad_norm = g.norm(2)
      clipped_grad = ifelse(grad_norm >= CLIP_THRESH, 
                            g * CLIP_THRESH / grad_norm, g)
      updates.append((p, p - eta * clipped_grad))

    self._backprop = theano.function(
        inputs=[x, y, eta, y_in_x_inds, l2_reg],
        outputs=[p_y_seq, objective],
        updates=updates, on_unused_input='warn')

  def decode_greedy(self, ex, max_len=100):
    h_t = self._encode(ex.x_inds)
    y_tok_seq = []
    p_y_seq = []  # Should be handy for error analysis
    p = 1
    for i in range(max_len):
      write_dist = self._decoder_write(h_t)
      y_t = numpy.argmax(write_dist)
      p_y_t = write_dist[y_t]
      p_y_seq.append(p_y_t)
      p *= p_y_t
      if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
        break
      y_tok = self.out_vocabulary.get_word(y_t)
      y_tok_seq.append(y_tok)
      h_t = self._decoder_step(y_t, h_t)
    return [Derivation(ex, p, y_tok_seq)]

  def decode_beam(self, ex, beam_size=1, max_len=100):
    h_t = self._encode(ex.x_inds)
    beam = [[Derivation(ex, 1, [], hidden_state=h_t,
                        attention_list=[], copy_list=[])]]
    finished = []
    for i in range(1, max_len):
      new_beam = []
      for deriv in beam[i-1]:
        cur_p = deriv.p
        h_t = deriv.hidden_state
        y_tok_seq = deriv.y_toks
        write_dist = self._decoder_write(h_t)
        sorted_dist = sorted([(p_y_t, y_t) for y_t, p_y_t in enumerate(write_dist)],
                             reverse=True)
        for j in range(beam_size):
          p_y_t, y_t = sorted_dist[j]
          new_p = cur_p * p_y_t
          if y_t == Vocabulary.END_OF_SENTENCE_INDEX:
            finished.append(Derivation(ex, new_p, y_tok_seq))
            continue
          y_tok = self.out_vocabulary.get_word(y_t)
          new_h_t = self._decoder_step(y_t, h_t)
          new_entry = Derivation(ex, new_p, y_tok_seq + [y_tok],
                                 hidden_state=new_h_t)
          new_beam.append(new_entry)
      new_beam.sort(key=lambda x: x.p, reverse=True)
      beam.append(new_beam[:beam_size])
      finished.sort(key=lambda x: x.p, reverse=True)
    return sorted(finished, key=lambda x: x.p, reverse=True)
