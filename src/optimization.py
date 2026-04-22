import time
import numpy as np
import scipy
from collections import defaultdict, deque
from numpy.linalg import LinAlgError
try:
    from utils import get_line_search_tool
except ImportError:
    from .utils import get_line_search_tool


def _update_history(history, oracle, x, grad, start_time):
    history['time'].append(time.perf_counter() - start_time)
    history['func'].append(oracle.func(x))
    history['grad_norm'].append(np.linalg.norm(grad))
    if x.size <= 2:
        history['x'].append(np.copy(x))


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad0 = oracle.grad(x_k)
    grad0_norm2 = np.dot(grad0, grad0)
    if not np.isfinite(grad0_norm2):
        return x_k, 'computational_error', history

    start_time = time.perf_counter()
    previous_alpha = None

    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_k_norm2 = np.dot(grad_k, grad_k)
        if grad_k_norm2 <= tolerance * grad0_norm2:
            return x_k, 'success', history

        d_k = -grad_k
        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        previous_alpha = alpha

        if trace:
            _update_history(history, oracle, x_k, oracle.grad(x_k), start_time)
            history['alpha'].append(alpha)

        if display:
            print('iter={}, alpha={:.3e}, grad_norm={:.3e}'.format(k, alpha, np.sqrt(grad_k_norm2)))

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad0 = oracle.grad(x_k)
    grad0_norm2 = np.dot(grad0, grad0)
    if not np.isfinite(grad0_norm2):
        return x_k, 'computational_error', history

    start_time = time.perf_counter()

    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_k_norm2 = np.dot(grad_k, grad_k)
        if grad_k_norm2 <= tolerance * grad0_norm2:
            return x_k, 'success', history

        try:
            c, lower = scipy.linalg.cho_factor(oracle.hess(x_k), check_finite=True)
            d_k = -scipy.linalg.cho_solve((c, lower), grad_k)
        except LinAlgError:
            return x_k, 'newton_direction_error', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        if trace:
            _update_history(history, oracle, x_k, oracle.grad(x_k), start_time)
            history['alpha'].append(alpha)

        if display:
            print('iter={}, alpha={:.3e}, grad_norm={:.3e}'.format(k, alpha, np.sqrt(grad_k_norm2)))

    return x_k, 'iterations_exceeded', history


def linear_conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    n = x_k.size
    max_iter = n if max_iter is None else max_iter
    b_norm = np.linalg.norm(b)
    threshold = tolerance * b_norm
    start_time = time.perf_counter()

    g_k = matvec(x_k) - b
    d_k = -g_k
    g_norm = np.linalg.norm(g_k)

    if trace:
        history['time'].append(0.0)
        history['residual_norm'].append(g_norm)
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    if g_norm <= threshold:
        return x_k, 'success', history

    for k in range(max_iter):
        ad_k = matvec(d_k)
        denom = np.dot(ad_k, d_k)
        if denom <= 0 or not np.isfinite(denom):
            return x_k, 'computational_error', history

        g_norm2 = np.dot(g_k, g_k)
        alpha = g_norm2 / denom
        x_k = x_k + alpha * d_k
        g_next = g_k + alpha * ad_k
        g_next_norm = np.linalg.norm(g_next)

        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['residual_norm'].append(g_next_norm)
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print('iter={}, residual_norm={:.3e}'.format(k, g_next_norm))

        if g_next_norm <= threshold:
            return x_k, 'success', history

        beta = np.dot(g_next, g_next) / g_norm2
        d_k = -g_next + beta * d_k
        g_k = g_next

    return x_k, 'iterations_exceeded', history


def _beta_pr(g_next, g_k, d_k):
    return np.dot(g_next, g_next - g_k) / np.dot(g_k, g_k)


def _beta_hz(g_next, g_k, d_k):
    y_k = g_next - g_k
    denom = np.dot(d_k, y_k)
    if abs(denom) < 1e-16:
        return 0.0
    return np.dot(y_k - 2 * d_k * np.dot(y_k, y_k) / denom, g_next) / denom


def nonlinear_conjugate_gradients(oracle, x_0, tolerance=1e-4, max_iter=500,
                                  line_search_options=None, display=False, trace=False,
                                  beta_formula='PR', powell_restart=False,
                                  powell_threshold=0.2, reset_negative_beta=True,
                                  descent_restart=True):
    """
    Nonlinear Conjugate Gradients method for optimization.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_k = oracle.grad(x_k)
    grad0_norm2 = np.dot(grad_k, grad_k)
    if not np.isfinite(grad0_norm2):
        return x_k, 'computational_error', history

    d_k = -grad_k
    previous_alpha = None
    start_time = time.perf_counter()
    beta_formula = beta_formula.upper()

    for k in range(max_iter):
        grad_k_norm2 = np.dot(grad_k, grad_k)
        if grad_k_norm2 <= tolerance * grad0_norm2:
            return x_k, 'success', history

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_next = x_k + alpha * d_k
        grad_next = oracle.grad(x_next)
        cos_theta = abs(np.dot(grad_next, grad_k)) / (np.linalg.norm(grad_next) * np.linalg.norm(grad_k) + 1e-16)

        if beta_formula == 'PR':
            beta = _beta_pr(grad_next, grad_k, d_k)
        elif beta_formula == 'HZ':
            beta = _beta_hz(grad_next, grad_k, d_k)
        else:
            raise ValueError('Unknown beta formula {}'.format(beta_formula))

        restart = False
        reason = ''
        if powell_restart and cos_theta > powell_threshold:
            beta = 0.0
            restart = True
            reason = 'powell'
        elif reset_negative_beta and beta < 0:
            beta = 0.0
            restart = True
            reason = 'negative_beta'

        d_next = -grad_next + beta * d_k
        if descent_restart and np.dot(grad_next, d_next) >= 0:
            d_next = -grad_next
            restart = True
            reason = 'not_descent'

        x_k = x_next
        grad_k = grad_next
        d_k = d_next
        previous_alpha = alpha

        if trace:
            _update_history(history, oracle, x_k, grad_k, start_time)
            history['alpha'].append(alpha)
            history['beta'].append(beta)
            history['cos'].append(cos_theta)
            history['restart'].append(restart)
            history['restart_reason'].append(reason)

        if display:
            print('iter={}, alpha={:.3e}, grad_norm={:.3e}, cos={:.3e}'.format(
                k, alpha, np.linalg.norm(grad_k), cos_theta))

    return x_k, 'iterations_exceeded', history


def _lbfgs_direction(grad, s_hist, y_hist):
    if len(s_hist) == 0:
        return -grad

    q = np.copy(grad)
    alphas = []
    rhos = []
    for s, y in zip(reversed(s_hist), reversed(y_hist)):
        rho = 1.0 / np.dot(y, s)
        alpha = rho * np.dot(s, q)
        q = q - alpha * y
        alphas.append(alpha)
        rhos.append(rho)

    s_last, y_last = s_hist[-1], y_hist[-1]
    gamma = np.dot(s_last, y_last) / np.dot(y_last, y_last)
    r = gamma * q

    for s, y, alpha, rho in zip(s_hist, y_hist, reversed(alphas), reversed(rhos)):
        beta = rho * np.dot(y, r)
        r = r + s * (alpha - beta)

    return -r


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False, callback=None):
    """
    Limited-memory BFGS method for optimization.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad_k = oracle.grad(x_k)
    grad0_norm2 = np.dot(grad_k, grad_k)
    if not np.isfinite(grad0_norm2):
        return x_k, 'computational_error', history

    s_hist = deque(maxlen=memory_size)
    y_hist = deque(maxlen=memory_size)
    previous_alpha = None
    start_time = time.perf_counter()

    for k in range(max_iter):
        grad_k_norm2 = np.dot(grad_k, grad_k)
        if grad_k_norm2 <= tolerance * grad0_norm2:
            return x_k, 'success', history

        d_k = _lbfgs_direction(grad_k, s_hist, y_hist) if memory_size > 0 else -grad_k
        if np.dot(grad_k, d_k) >= 0:
            d_k = -grad_k

        alpha = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_next = x_k + alpha * d_k
        grad_next = oracle.grad(x_next)
        s_k = x_next - x_k
        y_k = grad_next - grad_k
        curvature = np.dot(s_k, y_k)
        if memory_size > 0 and curvature > 1e-12 * np.dot(s_k, s_k):
            s_hist.append(s_k)
            y_hist.append(y_k)

        x_k = x_next
        grad_k = grad_next
        previous_alpha = alpha

        if trace:
            _update_history(history, oracle, x_k, grad_k, start_time)
            history['alpha'].append(alpha)
            history['memory'].append(len(s_hist))

        if callback is not None:
            callback(x_k, grad_k, k, time.perf_counter() - start_time)

        if display:
            print('iter={}, alpha={:.3e}, grad_norm={:.3e}'.format(k, alpha, np.linalg.norm(grad_k)))

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian-free Newton method for optimization.
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    grad0 = oracle.grad(x_k)
    grad0_norm2 = np.dot(grad0, grad0)
    if not np.isfinite(grad0_norm2):
        return x_k, 'computational_error', history

    start_time = time.perf_counter()

    for k in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_k_norm = np.linalg.norm(grad_k)
        if grad_k_norm ** 2 <= tolerance * grad0_norm2:
            return x_k, 'success', history

        eta = min(0.5, np.sqrt(grad_k_norm))
        d0 = -grad_k
        cg_iters = 0
        for _ in range(12):
            matvec = lambda v: oracle.hess_vec(x_k, v)
            d_k, msg, cg_hist = linear_conjugate_gradients(
                matvec,
                -grad_k,
                d0,
                tolerance=eta,
                max_iter=x_k.size,
                trace=trace,
            )
            cg_iters += 0 if cg_hist is None else max(0, len(cg_hist['residual_norm']) - 1)
            if msg == 'success' and np.dot(grad_k, d_k) < 0:
                break
            eta *= 0.1
            d0 = d_k
        else:
            d_k = -grad_k

        alpha = line_search_tool.line_search(oracle, x_k, d_k, 1.0)
        if alpha is None or not np.isfinite(alpha):
            return x_k, 'computational_error', history

        x_k = x_k + alpha * d_k
        if trace:
            _update_history(history, oracle, x_k, oracle.grad(x_k), start_time)
            history['alpha'].append(alpha)
            history['cg_iters'].append(cg_iters)

        if display:
            print('iter={}, alpha={:.3e}, grad_norm={:.3e}, cg_iters={}'.format(
                k, alpha, grad_k_norm, cg_iters))

    return x_k, 'iterations_exceeded', history
