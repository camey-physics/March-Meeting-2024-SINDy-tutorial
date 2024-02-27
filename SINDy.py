import numpy as np
import h5py
import matplotlib.pyplot as plt
# import sklearn
from SINDy_tools import calc_gradients, sequentially_thresholded_LSQ
from SINDy_tools import construct_vector_test_functions

tensorTerm = 0
numSamples = 100_000

f = h5py.File('/home/chris/Work/tutorial/test.h5', 'r')
dx = f['dx'][()]
dt = f['dt'][()]
Q = np.moveaxis(f['qtensor'][::4], 0, -1)
Q = np.array([[Q[0],Q[1]],[Q[1],-Q[0]]])
u = np.moveaxis(f['velocity'][::4], 0, -1)
P = np.moveaxis(f['pressure'][::4], 0, -1)
f.close()

'''All data is arranged as [...,x,y,t], where "..." corresponds to vector terms'''
lx, ly, lt = Q.shape[-3:]

numDataPoints = lx *ly *lt
sample = np.arange(numDataPoints)
np.random.shuffle(sample)
sample = sample[:numSamples]

lap_u = calc_gradients(u, lap_order=[1], dh=(dx,dx))
grad_Q = calc_gradients(Q, grad_order=[1], dh=(dx,dx))
grad_P = calc_gradients(P, grad_order=[1], dh=(dx,dx))
dt_u = np.gradient(u, dt, axis=-1)

term_names = []
# term_names.append('∇^2 u')
term_names.append('d_t u')
term_names.append('u')
term_names.append('∇.Q')
term_names.append('Q.u')
term_names.append('(u.u)u')
term_names.append('∇(Q:Q)')
term_names.append('∇P')
# term_names.append('(∇.u)u')

term_names = np.array(term_names)

# LHS
lhs = np.copy(lap_u)[tensorTerm].flatten()[sample]

rhs = []
'''term_names.append('d_t u')'''
tmp = dt_u[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('u')'''
tmp = u[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('∇.Q')'''
tmp = np.einsum('aa...', grad_Q)[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('Q.u')'''
tmp = np.einsum('ia...,a...->i...', Q, u)[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('(u.u)u')'''
tmp = np.einsum('a...,a...->...', u, u)[None,...] *u
tmp = tmp[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('∇(Q:Q)')'''
tmp = 2 *np.einsum('iab...,ba...->i...', grad_Q, Q)
tmp = tmp[tensorTerm].flatten()[sample]
rhs.append(tmp)
'''term_names.append('∇P')'''
tmp = grad_P[tensorTerm].flatten()[sample]
rhs.append(tmp)

rhs = np.array(rhs).T

coeff, R2 = sequentially_thresholded_LSQ(rhs, lhs)

I = coeff != 0

print(coeff[2][I[2]])
print(term_names[I[2]])

