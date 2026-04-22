import numpy as np
import scipy
from scipy.special import ndtr, log_ndtr
from scipy.stats import norm


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        raise NotImplementedError('Grad oracle is not implemented.')

    def hess(self, x):
        raise NotImplementedError('Hessian oracle is not implemented.')

    def func_directional(self, x, d, alpha):
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^T A x - b^T x.
    """
    def __init__(self, A, b):
        if not scipy.sparse.issparse(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A

    def hess_vec(self, x, v):
        return self.A.dot(v)


class NonConvexOracle(BaseSmoothOracle):
    """
    Beale function.
    """
    def __init__(self):
        self.c = np.array([1.5, 2.25, 2.625])
        self.p = np.array([1, 2, 3])

    def func(self, x):
        x1, x2 = x[0], x[1]
        g = self.c - x1 + x1 * x2 ** self.p
        return np.sum(g ** 2)

    def grad(self, x):
        x1, x2 = x[0], x[1]
        g = self.c - x1 + x1 * x2 ** self.p
        dg_dx = x2 ** self.p - 1
        dg_dy = x1 * self.p * x2 ** (self.p - 1)
        return np.array([2 * np.sum(g * dg_dx), 2 * np.sum(g * dg_dy)])

    def hess(self, x):
        x1, x2 = x[0], x[1]
        g = self.c - x1 + x1 * x2 ** self.p
        d2_dx2 = 2 * np.sum((x2 ** self.p - 1) ** 2)
        d2_dxdy = np.sum(2 * self.p * x2 ** (self.p - 1) * (x1 * (x2 ** self.p - 1) + g))

        contrib = np.zeros_like(self.p, dtype=float)
        mask = self.p > 1
        contrib[mask] = (2 * self.p[mask] * x1 * x2 ** (self.p[mask] - 2) *
                         (self.p[mask] * x1 * x2 ** self.p[mask] + g[mask] * (self.p[mask] - 1)))
        contrib[~mask] = 2 * x1 ** 2
        d2_dy2 = np.sum(contrib)
        return np.array([[d2_dx2, d2_dxdy], [d2_dxdy, d2_dy2]])


class RidgeL2Oracle(BaseSmoothOracle):
    """
    Oracle for ridge regression with l2 regularization.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        z = self.matvec_Ax(x)
        return 0.5 * np.mean((z - self.b) ** 2) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = self.matvec_Ax(x)
        return self.matvec_ATx((z - self.b) / self.m) + self.regcoef * x

    def hess(self, x):
        H = self.matmat_ATsA(np.ones(self.m)) / self.m
        n = x.shape[0]
        if scipy.sparse.issparse(H):
            return H + self.regcoef * scipy.sparse.eye(n, format='csr')
        return H + self.regcoef * np.eye(n)

    def hess_vec(self, x, v):
        return self.matvec_ATx(self.matvec_Ax(v)) / self.m + self.regcoef * v


class ProbitL2Oracle(BaseSmoothOracle):
    """
    Oracle for probit classification with l2 regularization.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef, matvec_ATx_sqr=None):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.matvec_ATx_sqr = matvec_ATx_sqr
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        z = self.matvec_Ax(x)
        return np.mean(-log_ndtr(self.b * z)) + 0.5 * self.regcoef * np.dot(x, x)

    def grad(self, x):
        z = self.matvec_Ax(x)
        return self.matvec_ATx(self._grad_loss(z, self.b)) / self.m + self.regcoef * x

    def hess(self, x):
        z = self.matvec_Ax(x)
        H = self.matmat_ATsA(self._hess_loss(z, self.b)) / self.m
        n = x.shape[0]
        if scipy.sparse.issparse(H):
            return H + self.regcoef * scipy.sparse.eye(n, format='csr')
        return H + self.regcoef * np.eye(n)

    def hess_vec(self, x, v):
        z = self.matvec_Ax(x)
        s = self._hess_loss(z, self.b)
        return self.matvec_ATx(s * self.matvec_Ax(v)) / self.m + self.regcoef * v

    def hess_diagonal(self, x):
        z = self.matvec_Ax(x)
        s = self._hess_loss(z, self.b)
        if self.matvec_ATx_sqr is None:
            H = self.hess(x)
            return np.asarray(H.diagonal()).ravel() if scipy.sparse.issparse(H) else np.diag(H)
        return np.asarray(self.matvec_ATx_sqr(s)).ravel() / self.m + self.regcoef

    def _grad_loss(self, z, b):
        t = b * z
        ratio = np.exp(norm.logpdf(t) - log_ndtr(t))
        return -b * ratio

    def _hess_loss(self, z, b):
        t = b * z
        ratio = np.exp(norm.logpdf(t) - log_ndtr(t))
        return ratio * (t + ratio)


REG_MODEL_NAMEL2Oracle = RidgeL2Oracle
CLASS_MODEL_NAMEL2Oracle = ProbitL2Oracle


def grad_finite_diff(func, x, eps=1e-8):
    n = len(x)
    grad = np.zeros(n)
    fx = func(x)
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        grad[i] = (func(x + e) - fx) / eps
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    fx = func(x)
    f_i = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        f_i.append(func(x + e))
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = eps
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = eps
            hess[i, j] = (func(x + e_i + e_j) - f_i[i] - f_i[j] + fx) / eps ** 2
    return hess


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    n = len(x)
    hv = np.zeros(n)
    fx = func(x)
    f_v = func(x + eps * v)
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        hv[i] = (func(x + eps * v + e) - f_v - func(x + e) + fx) / eps ** 2
    return hv
