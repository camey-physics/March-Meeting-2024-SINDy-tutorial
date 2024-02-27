#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:53:42 2024

@author: chris
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
import sklearn

def grad_2d(M, dh=(1,1), axes=(-2,-1), mode='wrap'):
    '''Data is assumed to have dimensions of [...,x,y]'''
    from scipy import ndimage
    kernel = -np.array([-1/2,0,1/2])
    dx = ndimage.convolve1d(M, kernel, axis=axes[0], mode=mode) /dh[0]
    dy = ndimage.convolve1d(M, kernel, axis=axes[1], mode=mode) /dh[1]
    return np.array([dx, dy])

def calc_gradients(A, grad_order=[], lap_order=[], axes=(-3,-2), dh=(1,1)):
    '''Data is assumed to have dimensions of [...,x,y,t].'''
    grad_order = np.sort(np.unique(grad_order))
    lap_order = np.sort(np.unique(lap_order))
    xax, yax = axes
    xdim, ydim = np.array(A.shape)[[xax, yax]]
    dx, dy = dh
    lap_l = []
    if lap_order.shape[0] != 0:
        lap_A = np.copy(A)
        for i in range(lap_order[-1]):
            lap_A = grad_2d(lap_A, dh=dh, axes=axes)
            lap_A = grad_2d(lap_A, dh=dh, axes=axes)
            lap_A = np.einsum('ii...', lap_A)
            if (i+1 in lap_order):
                lap_l.append(lap_A)
        del lap_A
    grad_l = []
    if grad_order.shape[0] != 0:
        grad_A = A
        for i in range(grad_order[-1]):
            grad_A = grad_2d(grad_A, dh=dh, axes=axes)
            if (i+1 in grad_order):
                grad_l.append(grad_A)
    if (len(lap_l+grad_l) > 1):
        return grad_l + lap_l
    elif (len(lap_l+grad_l) == 1):
        return (grad_l + lap_l)[0]

def sequentially_thresholded_LSQ(rhs, lhs, test=False):
    from sklearn import linear_model
    numSamples, numTerms = rhs.shape
    
    ind = np.arange(numSamples)
    np.random.shuffle(ind)
    if test:
        trainSize = np.floor(4 *numSamples /5).astype(int)
        trainInd = ind[:trainSize]
        testInd = ind[trainSize:]
    else:
        trainInd = ind
        testInd = ind
    
    coeff = np.zeros((numTerms, numTerms))
    currentTerms = np.zeros(numTerms, dtype=bool) + True
    R2 = np.zeros(numTerms) + 1
    for i in range(numTerms):
        reg = linear_model.LinearRegression().fit(rhs[trainInd][:,currentTerms], lhs[trainInd])
        coeff[-1-i][currentTerms] = reg.coef_
        R2[-1-i] = reg.score(rhs[testInd][:,currentTerms], lhs[testInd])
        currentTermIndices = np.argwhere(currentTerms)[:,0]
        R2_l = []
        if (i < numTerms-1):
            for ind in currentTermIndices:
                currentTerms[ind] = False
                reg = linear_model.LinearRegression().fit(rhs[trainInd][:,currentTerms], lhs[trainInd])
                R2_l.append(reg.score(rhs[trainInd][:,currentTerms], lhs[trainInd]))
                currentTerms[ind] = True
            maxInd = currentTermIndices[np.argmax(R2_l)]
            currentTerms[maxInd] = False
        
    return coeff, R2

def integrate_3d(A, dx=1, dy=1, dt=1):
    from scipy.integrate import trapezoid as trap
    return trap(trap(trap(A, dx=dx, axis=-3), dx=dy, axis=-2), dx=dt, axis=-1)

def get_vector_test_functions(lx, ly, lt, dx, dy, dt, p=4, q=4):
    import numpy as np
    import sympy as sym
    from sympy import Matrix
    X = np.arange(0, 2*lx+1) - lx
    Y = np.arange(0, 2*ly+1) - ly
    T = np.arange(0, 2*lt+1) - lt
    X, Y, T = np.meshgrid(X,Y,T,indexing='ij')
    
    x = sym.Symbol('x');
    y = sym.Symbol('y');
    t = sym.Symbol('t');
    sym_w = (x**2/lx**2 - 1)**p *(y**2 /ly**2 - 1)**p *(t**2 /lt**2 - 1)**q
    
    '''Make a dummy placeholder vec_w'''
    sym_vec_w = Matrix([0,0]).T
    sym_vec_w[0] = sym.diff(sym_w, y, 1) /dy
    sym_vec_w[1] = -sym.diff(sym_w, x, 1) /dx
    
    vec_w = sym.lambdify([x,y,t], sym_vec_w)(X,Y,T)[0,...]
    
    l = [x,y]
    dl = [dx, dy]
    sym_lap_w = Matrix([0,0]).T
    for i in range(2):
        for j in range(2):
            sym_lap_w[i] += sym.diff(sym_vec_w[i],l[j],2) /dl[j]**2
    lap_w = sym.lambdify([x,y,t], sym_lap_w)(X,Y,T)[0,...]
    
    sym_grad_w = Matrix([[0,0],[0,0]])
    
    for i in range(2):
        for j in range(2):
            sym_grad_w[i,j] = sym.diff(sym_vec_w[j], l[i], 1) /dl[j]
    grad_w = sym.lambdify([x,y,t], sym_grad_w)(X,Y,T)
    
    sym_dt_w = sym.diff(sym_vec_w,t,1) /dt
    dt_w = sym.lambdify([x,y,t], sym_dt_w)(X,Y,T)[0,...]

    return vec_w, lap_w, grad_w, dt_w

# def get_vector_test_functions(wx, wy, wt, dx, dy, dt, p=4, q=4):
#     import numpy as np
#     import sympy as sym
#     from sympy import Matrix
#     X = np.linspace(-1,1,2*wx+1)
#     Y = np.linspace(-1,1,2*wy+1)
#     T = np.linspace(-1,1,2*wt+1)
#     X, Y, T = np.meshgrid(X,Y,T,indexing='ij')
    
#     x = sym.Symbol('x');
#     y = sym.Symbol('y');
#     t = sym.Symbol('t');
#     sym_w = (x**2 - 1)**p *(y**2 - 1)**p *(t**2 - 1)**q
    
#     '''Make a dummy placeholder vec_w'''
#     sym_vec_w = Matrix([0,0]).T
#     sym_vec_w[0] = sym.diff(sym_w, y, 1)
#     sym_vec_w[1] = -sym.diff(sym_w, x, 1)
    
#     vec_w = sym.lambdify([x,y,t], sym_vec_w)(X,Y,T)[0,...]
    
#     l = [x,y]
#     sym_lap_w = Matrix([0,0]).T
#     for i in range(2):
#         for j in range(2):
#             sym_lap_w[i] += sym.diff(sym_vec_w[i],l[j],2)
#     lap_w = sym.lambdify([x,y,t], sym_lap_w)(X,Y,T)[0,...]
    
#     sym_grad_w = Matrix([[0,0],[0,0]])
    
#     for i in range(2):
#         for j in range(2):
#             sym_grad_w[i,j] = sym.diff(sym_vec_w[j], l[i], 1)
#     grad_w = sym.lambdify([x,y,t], sym_grad_w)(X,Y,T)
    
#     sym_dt_w = sym.diff(sym_vec_w,t,1)
#     dt_w = sym.lambdify([x,y,t], sym_dt_w)(X,Y,T)[0,...]

#     return vec_w, lap_w, grad_w, dt_w

def get_scalar_test_functions(wx, wy, wt, dx, dy, dt, p = 4, q = 4):
    import numpy as np
    import sympy as sym
    from sympy import Matrix
    X =  np.linspace(-1,1,2*wx+1)
    Y =  np.linspace(-1,1,2*wy+1)
    T =  np.linspace(-1,1,2*wt+1)
    X, Y, T = np.meshgrid(X,Y,T,indexing='ij')
    
    x = sym.Symbol('x');
    y = sym.Symbol('y');
    t = sym.Symbol('t');
    sym_w = (x**2 - 1)**p *(y**2 - 1)**p *(t**2 - 1)**q
    w = sym.lambdify([x,y,t], sym_w)(X,Y,T)
    
    l = [x,y]
    dl = [dx,dy]
    
    sym_grad_w = Matrix([0,0])
    for i in range(2):
        sym_grad_w[i] = sym.diff(sym_w,l[i],1) /dl[i]
    
    grad_w = sym.lambdify([x,y,t], sym_grad_w)(X,Y,T)[:,0]
    
    sym_grad_grad_w = Matrix([[0,0],[0,0]])
    for i in range(2):
        for j in range(2):
            sym_grad_grad_w[i,j] = sym.diff(sym.diff(sym_w,l[i],1),l[j],1) /dl[i] /dl[j]
    grad_grad_w = sym.lambdify([x,y,t], sym_grad_grad_w)(X,Y,T)
    
    sym_dt_w = sym.diff(sym_w,t,1) /dt
    dt_w = sym.lambdify([x,y,t], sym_dt_w)(X,Y,T)

    lap_w = np.einsum('aa...', grad_grad_w)

    return w, grad_w, grad_grad_w, dt_w
