"""Official BUT VBx core (Apache-2.0), Landini / Burget / Diez.

Source: https://github.com/BUTSpeechFIT/VBx/blob/master/VBx/VBx.py
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def VBx(
    X,
    Phi,
    loopProb=0.9,
    Fa=1.0,
    Fb=1.0,
    pi=10,
    gamma=None,
    maxIters=10,
    epsilon=1e-4,
    alphaQInit=1.0,
    return_model=False,
    alpha=None,
    invL=None,
):
    """Bayesian HMM clustering of x-vector sequences.

    X:   (T, D) features after PLDA transform
    Phi: (D,) across-class covariance diagonal (plda psi)
    """
    D = X.shape[1]

    # Accept Python int and numpy integer scalars (np.int32 has no len()).
    if isinstance(pi, (int, np.integer)):
        pi = np.ones(int(pi)) / float(int(pi))

    if gamma is None:
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    assert gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0]

    G = -0.5 * (np.sum(X**2, axis=1, keepdims=True) + D * np.log(2 * np.pi))
    V = np.sqrt(Phi)
    rho = X * V
    Li = []
    for ii in range(maxIters):
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi)
            alpha = Fa / Fb * invL * gamma.T.dot(rho)
        log_p_ = Fa * (rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(Phi) + G)
        tr = np.eye(len(pi)) * loopProb + (1 - loopProb) * pi
        gamma, log_pX_, logA, logB = forward_backward(log_p_, tr, pi)
        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)
        pi = gamma[0] + (1 - loopProb) * pi * np.sum(
            np.exp(
                logsumexp(logA[:-1], axis=1, keepdims=True)
                + log_p_[1:]
                + logB[1:]
                - log_pX_
            ),
            axis=0,
        )
        pi = pi / pi.sum()
        Li.append([ELBO])

        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())


def forward_backward(lls, tr, ip):
    eps = 1e-8
    ltr = np.log(tr + eps)
    lfw = np.empty_like(lls)
    lbw = np.empty_like(lls)
    lfw[:] = -np.inf
    lbw[:] = -np.inf
    lfw[0] = lls[0] + np.log(ip + eps)
    lbw[-1] = 0.0

    for ii in range(1, len(lls)):
        lfw[ii] = lls[ii] + logsumexp(lfw[ii - 1] + ltr.T, axis=1)

    for ii in reversed(range(len(lls) - 1)):
        lbw[ii] = logsumexp(ltr + lls[ii + 1] + lbw[ii + 1], axis=1)

    tll = logsumexp(lfw[-1], axis=0)
    pi = np.exp(lfw + lbw - tll)
    return pi, tll, lfw, lbw


def twoGMMcalib_lin(s, niters=20):
    """Two-Gaussian score calibration → AHC threshold (BUT diarization_lib)."""
    from scipy.special import softmax

    s = np.asarray(s, dtype=np.float64).ravel()
    weights = np.array([0.5, 0.5])
    means = np.mean(s) + np.std(s) * np.array([-1.0, 1.0])
    var = np.var(s)
    threshold = np.inf
    for _ in range(niters):
        lls = (
            np.log(weights)
            - 0.5 * np.log(var + 1e-12)
            - 0.5 * (s[:, np.newaxis] - means) ** 2 / (var + 1e-12)
        )
        gammas = softmax(lls, axis=1)
        cnts = np.sum(gammas, axis=0) + 1e-12
        weights = cnts / cnts.sum()
        means = s.dot(gammas) / cnts
        var = ((s**2).dot(gammas) / cnts - means**2).dot(weights)
        var = max(float(var), 1e-12)
        threshold = (
            -0.5
            * (np.log(weights**2 / var) - means**2 / var).dot([1, -1])
            / ((means / var).dot([1, -1]) + 1e-12)
        )
    return float(threshold), None
