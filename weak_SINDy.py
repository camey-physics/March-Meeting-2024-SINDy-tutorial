import numpy as np
import h5py
import matplotlib.pyplot as plt
# import sklearn
from SINDy_tools import calc_gradients, sequentially_thresholded_LSQ
from SINDy_tools import get_vector_test_functions, integrate_3d

tensorTerm = 0
numSamples = 1_000
wx = 20
wy = 20
wt = 15

f = h5py.File('/home/chris/Work/tutorial/test.h5', 'r')
dx = f['dx'][()]
dy = dx
dt = f['dt'][()]
Q = np.moveaxis(f['qtensor'][::4], 0, -1)
Q = np.array([[Q[0],Q[1]],[Q[1],-Q[0]]])
u = np.moveaxis(f['velocity'][::4], 0, -1)
# P = np.moveaxis(f['pressure'][::4], 0, -1)
f.close()

vec_w, lap_w, grad_w, dt_w = get_vector_test_functions(wx, wy, wt, dx, dy, dt)

'''All data is arranged as [...,x,y,t], where "..." corresponds to vector terms'''
lx, ly, lt = Q.shape[-3:]

numDataPoints = lx *ly *lt
sample = np.arange(numDataPoints)
np.random.shuffle(sample)
sample = sample[:numSamples]

lap_u = calc_gradients(u, lap_order=[1], dh=(dx,dx))
grad_Q = calc_gradients(Q, grad_order=[1], dh=(dx,dx))
# grad_P = calc_gradients(P, grad_order=[1], dh=(dx,dx))
dt_u = np.gradient(u, dt, axis=-1)

term_names = []
# term_names.append('∇^2 u')
term_names.append('d_t u')
term_names.append('u')
term_names.append('∇.Q')
term_names.append('Q.u')
term_names.append('(u.u)u')
term_names.append('∇(Q:Q)')
# term_names.append('∇P')
# term_names.append('(∇.u)u')

term_names = np.array(term_names)

numTerms = term_names.shape[0]

lhs = np.zeros(numSamples, dtype=float)
rhs = np.zeros((numTerms, numSamples), dtype=float)

for sample in range(numSamples):
    xi = np.random.randint(wx, lx-wx)
    yi = np.random.randint(wy, ly-wy)
    ti = np.random.randint(wt, lt-wt)
    s = (..., slice(xi-wx, xi+wx+1), slice(yi-wy, yi+wy+1), slice(ti-wt, ti+wt+1))
    
    # LHS
    tmp = np.einsum('a...,a...', lap_w, u[s])
    tmp = integrate_3d(tmp, dx, dy, dt)
    lhs[sample] = tmp
    
    # RHS
    term = 0
    '''term_names.append('d_t u')'''
    tmp = np.einsum('a...,a...', dt_w, u[s])
    tmp = integrate_3d(tmp, dx, dy, dt)
    rhs[term,sample] = tmp#[0]+tmp[1]
    term += 1
    '''term_names.append('u')'''
    tmp = np.einsum('a...,a...', vec_w, u[s])
    tmp = integrate_3d(tmp, dx, dy, dt)
    rhs[term,sample] = tmp
    term += 1
    '''term_names.append('∇.Q')'''
    tmp = np.einsum('ab...,ba...->...',grad_w, Q[s])
    tmp = integrate_3d(tmp, dx, dy, dt)
    rhs[term,sample] = tmp
    term += 1
    '''term_names.append('Q.u')'''
    tmp = np.einsum('ba...,a...,b...', Q[s], u[s], vec_w)
    tmp = integrate_3d(tmp)
    rhs[term,sample] = tmp
    term += 1
    '''term_names.append('(u.u)u')'''
    tmp = u[s]
    tmp = np.einsum('a...,a...,b...,b...', tmp, tmp, tmp, vec_w)
    tmp = integrate_3d(tmp)
    rhs[term,sample] = tmp
    term += 1
    '''term_names.append('∇(Q:Q)')'''
    tmp = 2 *np.einsum('cab...,ba...,c...', grad_Q[s], Q[s], vec_w)
    tmp = integrate_3d(tmp)
    rhs[term,sample] = tmp
    term += 1

rhs = np.array(rhs).T

coeff, R2 = sequentially_thresholded_LSQ(rhs, lhs)

I = coeff != 0

print(coeff[1][I[1]])
print(term_names[I[1]])

