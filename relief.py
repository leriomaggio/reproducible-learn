## This code is written by Davide Albanese, <albanese@fbk.eu>.

## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

# from __future__ import division
import numpy as np


def diff(a, b, nu):
    d = np.abs(a - b) / nu
    d[np.isnan(d)] = 0.0
    return d


def k_nearest(x, y, i, cl, k, nu):
    dist, idx = [], []
    for j in range(x.shape[0]):
        if (y[j] == cl) and (i != j):
            dist.append(np.sum(diff(x[i], x[j], nu)))
            idx.append(j)
    return np.array(idx)[np.argsort(dist)][:k]


class ReliefF(object):
    def __init__(self, k, m=None, equal_priors=False, seed=0):
        self._m = m
        self._k = k
        self._equal_priors = equal_priors
        self._seed = seed

        self._w = None
        self._classes = None
        self._model = False

    def learn(self, x, y):
         
        np.random.seed(self._seed)
        
        xa = np.asarray(x)
        ya = np.asarray(y, dtype=np.int)
        classes = np.unique(ya)
        k = classes.shape[0]

        if xa.ndim != 2:
            raise ValueError('x must be an 2d array_like object')
        
        if ya.ndim != 1:
            raise ValueError('y must be an 2d array_like object')

        if xa.shape[0] != ya.shape[0]:
            raise ValueError('x, y: shape mismatch')
        
        if k < 2:
            raise ValueError('number of classes must be >= 2')

        n, p = xa.shape
                
        # m randomly selected instances
        instances = np.arange(n)
        if self._m is None:
            instances = np.arange(n)
        else:
            np.random.shuffle(instances)
            instances = instances[:self._m]
            
        m = instances.shape[0]

        # prior probabilities
        priors = {}
        if self._equal_priors:
            for c in classes:            
                priors[c] = (1 / k)
        else:
            for c in classes:
                priors[c] = np.sum(ya == c) / n

        # precompute (max(A) - min(A))
        nu = np.max(xa, axis=0) - np.min(xa, axis=0)

        w = np.zeros(p, dtype=np.float)
        for i in instances:
            c_m = [c for c in classes if c!=y[i]]

            idx_nh = k_nearest(xa, ya, i, y[i], self._k, nu)
            H = 0.0
            for j in idx_nh:
                H += diff(xa[i], xa[j], nu)

            M = 0.0
            for j, c in enumerate(c_m):
                a = priors[c] / (1 - priors[y[i]])
                idx_nm = k_nearest(xa, ya, i, c, self._k, nu)
                b = 0.0
                for z in idx_nm:
                    b += diff(xa[i], xa[z], nu)
                M += a * b

            w = w - (H / (m * self._k)) + \
                (M / (m * self._k))
    
        self._w = w
        self._classes = classes
        self._model = False
                  
    def w(self):
        return self._w
