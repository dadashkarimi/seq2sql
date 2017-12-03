import theano
import theano.tensor as T
import numpy as np
x_inner = T.tensor3('x2')
x_outer = T.dvector('x')
 

dot = T.dot(x_inner, x_outer)
logistic = theano.function([x_inner, x_outer], dot)

h_for_write = np.array([0, 1, 3])
scores_inner = np.array([[0, 1, 3], [1,2,3], [1,2,3], [1,2,3], [1,2,3]])
scores_inner = np.expand_dims(scores_inner, axis=0)
print(scores_inner.shape)
print(h_for_write.shape)
out = logistic(scores_inner, h_for_write)
print(out.shape)
