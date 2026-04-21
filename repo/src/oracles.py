import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v
        """
        return self.hess(x).dot(v)


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class NonConvexOracle(BaseSmoothOracle):
    """
    Oracle for test function from your assignment.
    """

    def __init__(self):
	pass

    def func(self, x):
        # TODO

    def grad(self, x):
        # TODO

    def hess(self, x):
        # TODO

    def hess_vec(self, x, v):
 	# TODO


class REG_MODEL_NAMEL2Oracle(BaseSmoothOracle):
    """
    Oracle for regression loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        return None

    def grad(self, x):
        # TODO: Implement
        return None

    def hess(self, x):
        # TODO: Implement
        return None

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v.
        """
        # TODO: Implement matrix-vector product f''(x) v WITHOUT explicitly building
        # the full Hessian matrix (to be fast and efficient for Truncated Newton).
        return super().hess_vec(x, v)


class CLASS_MODEL_NAMEL2Oracle(BaseSmoothOracle):
    """
    Oracle for classification loss  function with l2 regularization:
         check your individual assignment

    Let A and b be parameters of the model (feature matrix
    and labels vector respectively).   

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef

    def func(self, x):
        # TODO: Implement
        return None

    def grad(self, x):
        # TODO: Implement
        return None

    def hess(self, x):
        # TODO: Implement
        return None

    def hess_vec(self, x, v):
        """
        Computes matrix-vector product with Hessian matrix f''(x) v.
        """
        # TODO: Implement matrix-vector product f''(x) v WITHOUT explicitly building
        # the full Hessian matrix (to be fast and efficient for Truncated Newton).
        return super().hess_vec(x, v)


def hess_vec_finite_diff(func, x, v, eps=1e-5):
    """
    Returns approximation of the matrix product 'Hessian times vector'
    using finite differences.
    """
    # TODO: Implement numerical estimation of the Hessian times vector
    # using the formula from Section 1.4 of the PDF:
    # [\nabla^2 f(x)v]_i \approx \frac{f(x + eps*v + eps*e_i) - f(x + eps*v) - f(x + eps*e_i) + f(x)}{eps^2}
    return None
