"""Gaussian processes."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import probnum
from probnum import backend, linops, problems, randprocs, randvars
from probnum.backend.random import RNGState
from probnum.backend.typing import ArrayLike, ShapeLike
from probnum.linalg import solvers

from . import linops as itergp_linops, methods

__all__ = ["GaussianProcess", "ConditionalGaussianProcess"]

GaussianProcess = randprocs.GaussianProcess


class ConditionalGaussianProcess(GaussianProcess):
    r"""Conditional Gaussian Process.

    Gaussian process conditioned on a dataset :math:`(X, y)` of training inputs
    :math:`X` and corresponding outputs :math:`y`. Given a Gaussian process prior
    :math:`f \sim \mathcal{GP}(\mu, k)` and likelihood 
    :math:`y \mid f(x) = f(x) + b = \mathcal{N}(y; f(x) + \mathbb{E}[b], \operatorname{Cov}(b))`, 
    the posterior at a new input :math:`x_\star` is given by 
    :math:`p(f_\star \mid X, y)` with

    .. math ::
        \begin{align}
            \mathbb{E}[f_\star] & = \mu(x_\star) + k(x_\star, X)\hat{K}^{-1} (y - \mu(X)),                 \\
		    \operatorname{Cov}(f_\star) & = k(x_\star,x_\star) - k(x_\star, X)\hat{K}^{-1} k(X, x_\star),
        \end{align}

    where :math:`\hat{K} = K + \operatorname{Cov}(b)` is the Gram matrix.

    Parameters
    ----------
    prior
        Gaussian process prior.
    Xs
        Training inputs.
    Ys
        Training outputs.
    bs
        Observation noise.
    ks
        Kernel functions.
    gram_Xs_Xs_inv
        Factor of the Gram matrix :math:`\hat{K} = K + \operatorname{Cov}(b)`.
    representer_weights
        Representer weights :math:`\hat{K}^{-1}y`.

    See Also
    --------
    GaussianProcess : Gaussian Processes.
    """

    def __init__(
        self,
        prior: GaussianProcess,
        Xs: Sequence[backend.Array],
        Ys: Sequence[backend.Array],
        bs: Sequence[randvars.Normal],
        ks: Sequence[randprocs.kernels.Kernel],
        gram_Xs_Xs_inv: Tuple[backend.Array, bool],
        representer_weights: randvars.RandomVariable,
    ):
        if prior.output_shape != ():
            raise ValueError("Currently, only scalar conditioning is supported.")

        self._prior = prior

        self._bs = tuple(bs)

        self._Xs = tuple(Xs)
        self._Ys = tuple(Ys)

        self._ks = tuple(ks)

        self._k_Xs: Callable[
            [backend.Array], backend.Array
        ] = lambda x: backend.concatenate(
            [
                k(backend.expand_dims(x, axis=-self._prior._input_ndim - 1), X)
                for k, X in zip(self._ks, self._Xs)
            ],
            axis=-1,
        )

        self._gram_Xs_Xs_inv = gram_Xs_Xs_inv
        self._representer_weights = representer_weights

        super().__init__(
            mean=ConditionalGaussianProcess.Mean(
                prior=self._prior,
                X=self._Xs[0],
                representer_weights=self._representer_weights,
            ),
            cov=ConditionalGaussianProcess.Kernel(
                prior_kernel=self._prior.cov,
                k_Xs=self._k_Xs,
                gram_Xs_Xs_inv=self._gram_Xs_Xs_inv,
            ),
        )

    @classmethod
    def from_data(
        cls,
        prior: GaussianProcess,
        X: backend.Array,
        Y: backend.Array,
        b: Optional[randvars.Normal] = None,
        approx_method: methods.ApproximationMethod = methods.Cholesky(),
    ):
        """Construct a :class:`ConditionalGaussianProcess` from data.

        Parameters
        ----------
        prior
            Gaussian process prior.
        X
            Training inputs.
        Y
            Training outputs.
        b
            Observation noise.
        approx_method
            (Iterative) Approximation method to compute the linear solve
            :math:`v \mapsto \hat{K}^{-1} v`.
        """
        X, Y, k, pred_mean_X, gram_XX = cls._preprocess_observations(
            prior=prior,
            X=X,
            Y=Y,
            b=b,
        )

        # Compute representer weights
        P = linops.Zero(shape=(X.shape[0], X.shape[0]))
        Pinv = linops.Zero(shape=(X.shape[0], X.shape[0]))
        problem = problems.LinearSystem(gram_XX, Y - pred_mean_X)
        linsys_prior = solvers.beliefs.LinearSystemBelief(
            x=randvars.Normal(mean=Pinv @ problem.b, cov=linops.aslinop(gram_XX).inv()),
            Ainv=Pinv,
            A=P,
        )
        qoi_belief, _ = approx_method.solve(problem=problem, prior=linsys_prior)
        gram_XX_inv = qoi_belief.Ainv
        representer_weights = qoi_belief.x

        return cls(
            prior=prior,
            bs=(b,),
            Xs=(X,),
            Ys=(Y,),
            ks=(k,),
            gram_Xs_Xs_inv=gram_XX_inv,
            representer_weights=representer_weights,
        )

    @property
    def computational_predictive(self) -> GaussianProcess:
        """Computational predictive distribution.

        Returns the Gaussian process quantifying computational uncertainty induced by the numerical
        approximation method used to compute the representer weights.
        """
        return GaussianProcess(
            mean=ConditionalGaussianProcess.Mean(
                prior=self._prior,
                X=self._Xs[0],
                representer_weights=self._representer_weights,
            ),
            cov=ConditionalGaussianProcess.ComputationalApproximationKernel(
                input_shape=self._prior.cov.input_shape,
                output_shape=self._prior.cov.output_shape,
                k_Xs=self._k_Xs,
                repr_weights_cov=self._representer_weights.cov,
            ),
        )

    def predictive(self, X: backend.Array) -> randvars.Normal:
        """Compute the predictive distribution.

        Parameters
        ----------
        X
            Inputs to predict at.
        """
        raise NotImplementedError

    class Mean(probnum.Function):
        """Mean function of a Gaussian process conditioned on data.

        Parameters
        ----------
        prior
            Gaussian process prior.
        X
            Input data.
        representer_weights
            Representer weights :math:`\hat{K}^{-1}y`.
        """

        def __init__(
            self,
            prior: GaussianProcess,
            X: Tuple[backend.Array],
            representer_weights: randvars.RandomVariable,
        ) -> None:
            self._prior = prior
            self._X = X
            self._representer_weights = representer_weights

            super().__init__(
                input_shape=self._prior.input_shape,
                output_shape=self._prior.output_shape,
            )

        def _evaluate(self, x: backend.Array) -> backend.Array:
            m_x = self._prior.mean(x)
            k_x_X = self._prior.cov.linop(x, self._X)

            return m_x + k_x_X @ self._representer_weights.mean

    class Kernel(randprocs.kernels.Kernel):
        """Kernel of a Gaussian process conditioned on data.

        Parameters
        ----------
        prior_kernel
            Kernel of the Gaussian process prior.
        k_Xs
            Cross covariance :math:`k(\cdot, X)` with the training data.
        gram_Xs_Xs_inv
            Factor of the Gram matrix :math:`\hat{K} = K + \operatorname{Cov}(b)`.
        """

        def __init__(
            self,
            prior_kernel: randprocs.kernels.Kernel,
            k_Xs: Callable[[backend.Array], backend.Array],
            gram_Xs_Xs_inv: backend.Array,
        ):
            self._prior_kernel = prior_kernel

            self._k_Xs = k_Xs
            self._gram_Xs_Xs_inv = gram_Xs_Xs_inv

            super().__init__(
                input_shape=prior_kernel.input_shape,
                output_shape=prior_kernel.output_shape,
            )

        def _evaluate(
            self, x0: backend.Array, x1: Optional[backend.Array] = None
        ) -> backend.Array:
            k_xx = self._prior_kernel(x0, x1)
            k_x0_Xs = self._k_Xs(x0)
            k_x1_Xs = self._k_Xs(x1) if x1 is not None else k_x0_Xs

            return (
                k_xx
                - (
                    k_x0_Xs[..., None, :]
                    @ (self._gram_Xs_Xs_inv @ k_x1_Xs[..., :, None])
                )[..., 0, 0]
            )  # TODO: replace k_x_Xs with KernelMatrix linear operator

    class ComputationalApproximationKernel(randprocs.kernels.Kernel):
        """Kernel describing computational uncertainty induced by a GP approximation.

        Parameters
        ----------
        input_shape
            Shape of the :class:`Kernel`'s input.
        output_shape
            Shape of the :class:`Kernel`'s output.
        k_Xs
            Cross covariance :math:`k(\cdot, X)` with the training data.
        repr_weights_cov
            Covariance of the representer weights computed via a
            :class:`~probnum.linalg.ProbabilisticLinearSolver`..
        """

        def __init__(
            self,
            input_shape: ShapeLike,
            output_shape: ShapeLike,
            k_Xs: Callable[[backend.Array], backend.Array],
            repr_weights_cov: backend.Array,
        ):
            self._k_Xs = k_Xs
            self._repr_weights_cov = repr_weights_cov

            super().__init__(
                input_shape=input_shape,
                output_shape=output_shape,
            )

        def _evaluate(
            self, x0: backend.Array, x1: Optional[backend.Array] = None
        ) -> backend.Array:
            k_x0_Xs = self._k_Xs(x0)
            k_x1_Xs = self._k_Xs(x1) if x1 is not None else k_x0_Xs

            return (
                k_x0_Xs[..., None, :] @ (self._repr_weights_cov @ k_x1_Xs[..., :, None])
            )[..., 0, 0]

    @classmethod
    def _preprocess_observations(
        cls,
        prior: GaussianProcess,
        X: backend.Array,
        Y: backend.Array,
        b: Optional[Union[randvars.Normal, randvars.Constant]],
    ) -> tuple[
        backend.Array,
        backend.Array,
        randprocs.kernels.Kernel,
        backend.Array,
        backend.Array,
    ]:
        # Reshape to (N, input_dim) and (N,)
        X = backend.asarray(X)
        Y = backend.asarray(Y)

        assert prior.output_shape == ()
        assert (
            X.ndim >= 1 and X.shape[X.ndim - prior._input_ndim :] == prior.input_shape
        )
        assert Y.shape == X.shape[: X.ndim - prior._input_ndim] + prior.output_shape

        X = X.reshape((-1,) + prior.input_shape, order="C")
        Y = Y.reshape((-1,), order="C")

        # Apply measurement operator to prior
        f = prior
        k = prior.cov

        # Compute predictive mean and kernel Gram matrix
        pred_mean_X = f.mean(X)
        gram_XX = itergp_linops.KernelMatrix(kernel=k, x0=X)

        if b is not None:
            assert isinstance(b, (randvars.Constant, randvars.Normal))
            assert b.shape == Y.shape

            pred_mean_X += b.mean.reshape((-1,), order="C")
            # This assumes that the covariance matrix is raveled in C-order
            gram_XX += linops.aslinop(b.cov)

        return X, Y, k, pred_mean_X, gram_XX

    def plot(
        self,
        X: backend.Array,
        data: Optional[Tuple[backend.Array, backend.Array]] = None,
        stdevs: Tuple[float, ...] = (2,),
        computational_predictive: bool = False,
        samples: int = 0,
        rng_state: RNGState = backend.random.rng_state(0),
        ax: Optional[matplotlib.axis.Axis] = None,
        color="C0",
        **kwargs,
    ) -> matplotlib.axis.Axis:
        """Plot the Gaussian process.

        Parameters
        ----------
        X
            Input points to plot at.
        stdevs
            Number of standard deviations to plot around the mean of the
            Gaussian process. Can be a tuple to plot more than one shaded
            credible region.
        computational_predictive
            Whether to plot the computational predictive distribution.
        samples
            Number of samples to plot.
        rng_state
            Random number generator state.
        ax
            Axis to plot into
        kwargs
            Key word arguments passed onto :func:`matplotlib.pyplot.plot`.
        """
        # TODO: refactor this to avoid code duplication with GaussianProcess.plot
        if ax is None:
            _, ax = plt.subplots()

        if X.ndim > 1:
            raise NotImplementedError("Only 1D plotting supported.")
        else:
            X = backend.sort(X)

        # Training data
        if data is not None:
            ax.scatter(data[0], data[1], marker=".", color="black", label="Data")

        # Posterior
        mean = self.mean(X)
        std = self.std(X)

        ax.plot(X, mean, label="GP mean", **kwargs)

        # Samples
        if samples > 0:
            samples = self.sample(rng_state=rng_state, sample_shape=samples, args=X)
            ax.plot(X, samples.T, lw=0.5, color=color, **kwargs)

        # Plot computational predictive distribution
        if computational_predictive:

            if X.ndim <= 1:
                X = backend.sort(X)

            mean = self.mean(X)
            gp_computational_pred = self.computational_predictive
            gp_computational_std = gp_computational_pred.std(X)

            for i in stdevs:

                # Observational uncertainty
                ax.fill_between(
                    x=X,
                    y1=mean - i * std,
                    y2=mean - i * gp_computational_std,
                    lw=0.1,
                    alpha=0.4,
                    label=f"GP {i}$\\times$stdev",
                    color=color,
                    **kwargs,
                )
                ax.fill_between(
                    x=X,
                    y1=mean + i * gp_computational_std,
                    y2=mean + i * std,
                    lw=0.1,
                    alpha=0.4,
                    color=color,
                    **kwargs,
                )

                # Computational uncertainty
                ax.fill_between(
                    x=X,
                    y1=mean - i * gp_computational_std,
                    y2=mean + i * gp_computational_std,
                    lw=0.0,
                    alpha=0.4,
                    color="C2",
                    label="Comp. uncertainty",
                    zorder=0,
                )
        else:
            for i in stdevs:
                ax.fill_between(
                    x=X,
                    y1=mean - i * std,
                    y2=mean + i * std,
                    lw=0.1,
                    alpha=0.4,
                    label=f"GP {i}$\\times$stdev",
                    color=color,
                    **kwargs,
                )

        return ax


def _condition_on_data(
    self,
    X: backend.Array,
    Y: backend.Array,
    b: Optional[randvars.Normal] = None,
    approx_method=methods.Cholesky(),
) -> ConditionalGaussianProcess:
    """Condition the Gaussian process on data.

    Given a Gaussian process prior :math:`f \sim \mathcal{GP}(\mu, k)`, condition on
    data :math:`(X, y)` assuming a likelihood of the form
    :math:`y \mid f(x) = f(x) + b = \mathcal{N}(y; f(x) + \mathbb{E}[b], \operatorname{Cov}(b))`.

    Parameters
    ----------
    X
        Training inputs.
    Y
        Training outputs.
    b
        Observation noise.
    approx_method
        (Iterative) approximation method to compute the conditional distribution.
    """
    if isinstance(self, ConditionalGaussianProcess):
        raise NotImplementedError("Conditioning multiple times is not (yet) supported.")

    return ConditionalGaussianProcess.from_data(
        prior=self, X=X, Y=Y, b=b, approx_method=approx_method
    )


def _plot(
    self,
    X: backend.Array,
    data: Optional[Tuple[backend.Array, backend.Array]] = None,
    stdevs: Tuple[float, ...] = (2,),
    samples: int = 0,
    rng_state: RNGState = backend.random.rng_state(0),
    ax: Optional[matplotlib.axis.Axis] = None,
    **kwargs,
) -> matplotlib.axis.Axis:
    """Plot the Gaussian process.

    Parameters
    ----------
    X
        Input points to plot at.
    data
        Training data.
    stdevs
        Number of standard deviations to plot around the mean of the
        Gaussian process. Can be a tuple to plot more than one shaded
        credible region.
    samples
        Number of samples to plot.
    rng_state
        Random number generator state for plotting samples.
    ax
        Plot axis.
    """
    if ax is None:
        _, ax = plt.subplots()

    if X.ndim > 1:
        raise NotImplementedError("Only 1D plotting supported.")
    else:
        X = backend.sort(X)

    # Training data
    if data is not None:
        ax.scatter(data[0], data[1], marker=".", color="black", label="Data")

    # Posterior
    mean = self.mean(X)
    std = self.std(X)

    ax.plot(X, mean, label="GP mean", **kwargs)

    for i in stdevs:
        ax.fill_between(
            x=X,
            y1=mean - i * std,
            y2=mean + i * std,
            lw=0.0,
            alpha=0.4,
            label=f"GP {i}$\\times$stdev",
            **kwargs,
        )

    # Samples
    if samples > 0:
        samples = self.sample(rng_state=rng_state, sample_shape=samples, args=X)
        ax.plot(X, samples.T, lw=0.5, **kwargs)

    return ax


def _std(self, args: ArrayLike) -> backend.Array:
    var = self.var(args=args)
    clip_mask = (var > 0.0) | (var < -(10**-12))
    std = backend.zeros_like(var)
    std[clip_mask] = backend.sqrt(var[clip_mask])
    return std


GaussianProcess.condition_on_data = _condition_on_data
GaussianProcess.std = _std
GaussianProcess.plot = _plot
