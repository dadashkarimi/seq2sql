"""Specifies a particular instance of an encoder-decoder model."""
from outputlayer import OutputLayer
from spec import Spec

class EncoderDecoderSpec(Spec):
  """Abstract class for a specification of an encoder-decoder model.
  
  Concrete subclasses must implement the following method:
  - self.create_rnn_layer(vocab, hidden_size): Create an RNN layer.
  """
  def create_vars(self):
    self.encoder = self.create_rnn_layer(
        self.hidden_size, self.in_vocabulary.emb_size,
        self.in_vocabulary.size(), True)
    self.decoder = self.create_rnn_layer(
        self.hidden_size, self.out_vocabulary.emb_size,
        self.out_vocabulary.size(), False)
    self.writer = self.create_output_layer(self.out_vocabulary, self.hidden_size)

  def get_local_params(self):
    return self.encoder.params + self.decoder.params + self.writer.params

  def create_output_layer(self, vocab, hidden_size):
    return OutputLayer(vocab, hidden_size)

  def get_init_state(self):
    return self.encoder.get_init_state()

  def f_enc(self, x_t, h_prev):
    """Returns the next hidden state for encoder."""
    input_t = self.in_vocabulary.get_theano_embedding(x_t)
    return self.encoder.step(input_t, h_prev)

  def f_dec(self, y_t, h_prev):
    """Returns the next hidden state for decoder."""
    input_t = self.out_vocabulary.get_theano_embedding(y_t)
    return self.decoder.step(input_t, h_prev)

  def f_write(self, h_t):
    """Gives the softmax output distribution."""
    return self.writer.write(h_t)
