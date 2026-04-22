import numpy as np
from scipy.optimize._linesearch import scalar_search_wolfe2


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'Best':
            pass
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        alpha_0 = previous_alpha if previous_alpha is not None else getattr(self, 'alpha_0', 1.0)

        phi = lambda a: oracle.func_directional(x_k, d_k, a)
        dphi = lambda a: oracle.grad_directional(x_k, d_k, a)
        dphi0 = dphi(0)

        if dphi0 >= 0:
            return None

        if self._method == 'Constant':
            return self.c

        if self._method == 'Best':
            h_d = oracle.hess_vec(x_k, d_k) if hasattr(oracle, 'hess_vec') else oracle.hess(x_k).dot(d_k)
            denom = np.dot(d_k, h_d)
            if denom <= 0:
                return None
            return -dphi0 / denom

        if self._method == 'Wolfe':
            alpha, _, _, _ = scalar_search_wolfe2(
                phi,
                dphi,
                phi0=phi(0),
                derphi0=dphi0,
                c1=self.c1,
                c2=self.c2,
            )
            if alpha is not None:
                return alpha

        if self._method in ['Wolfe', 'Armijo']:
            alpha = alpha_0
            phi0 = phi(0)
            while phi(alpha) > phi0 + self.c1 * alpha * dphi0:
                alpha *= 0.5
                if alpha < 1e-16:
                    return None
            return alpha

        raise ValueError('Unknown method {}'.format(self._method))


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if hasattr(line_search_options, 'line_search'):
            return line_search_options
        return LineSearchTool.from_dict(line_search_options)
    return LineSearchTool()
