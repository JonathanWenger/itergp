"""Belief update for projected residual observations."""

from __future__ import annotations

from probnum import backend, randvars
from probnum.linalg.solvers import belief_updates, beliefs

from itergp import linops


class ProjectedResidualBeliefUpdate(belief_updates.LinearSolverBeliefUpdate):
    r"""Gaussian belief update given projected residual information.

    Updates the belief over the quantities of interest of a linear system :math:`Ax=b` 
    given a Gaussian prior over the solution :math:`x \sim \mathcal{N}(x_i, A^{-1} - C_i)`, such that
    :math:`x_i = C_ib` and 
    information of the form :math:`s^\top r_i = s^\top (b - Ax_i)=s^\top A(x- x_i)`. The 
    belief update computes the posterior belief about the solution, given by 
    :math:`p(x \mid y) = \mathcal{N}(x; x_{i+1}, \Sigma_{i+1})`, such that

    .. math ::
        \begin{align}
            x_{i+1} &= x_i + \Sigma_i A^\top s (s^\top A \Sigma_i A^\top s)^\dagger s^\top r_i\\
            \Sigma_{i+1} &= \Sigma_i - \Sigma_i A^\top s (s^\top A \Sigma_i A s)^\dagger s^\top A \Sigma_i
        \end{align}
    """

    def __call__(
        self, solver_state: "probnum.linalg.solvers.LinearSolverState"
    ) -> beliefs.LinearSystemBelief:

        # Search direction
        A_action = solver_state.problem.A @ solver_state.action
        Ainv0_A_action = solver_state.belief.Ainv @ A_action
        search_dir = solver_state.action - Ainv0_A_action  # assumes Sigma_0 = A^{-1}
        A_search_dir = A_action - solver_state.problem.A @ Ainv0_A_action

        # Normalization constant
        gram = solver_state.action.T @ A_search_dir
        gram_pinv = 1.0 / gram if gram > 0.0 else 0.0

        return self.updated_linsys_belief(
            search_dir=search_dir,
            A_search_dir=A_search_dir,
            gram_pinv=gram_pinv,
            solver_state=solver_state,
        )

    def Ainv_update(
        self,
        normalized_search_dir: backend.Array,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> linops.LowRankMatrix:
        r"""Update the system matrix approximation.

        Parameters
        ----------
        normalized_search_dir
            Normalized search direction :math:`\fraction{1}{\sqrt{s_i^\top A \Sigma_{i-1} A s_i}}As_i`.
        solver_state
            State of the linear solver.
        """
        if backend.ndim(normalized_search_dir) == 1:
            normalized_search_dir = backend.reshape(normalized_search_dir, (-1, 1))

        if solver_state.step == 0:
            Ainv_update = linops.LowRankMatrix(U=normalized_search_dir)
        else:
            Ainv_update = linops.LowRankMatrix(
                U=backend.hstack(
                    [solver_state.belief.Ainv._summands[1].U, normalized_search_dir]
                )
            )
        return Ainv_update

    def A_update(
        self,
        A_normalized_search_dir: backend.Array,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> linops.LowRankMatrix:
        r"""Update the system matrix approximation.

        Parameters
        ----------
        A_normalized_search_dir
            :math:`A` multiplied normalized search direction :math:`\fraction{1}{\sqrt{s_i^\top A \Sigma_{i-1} A s_i}}As_i`.
        solver_state
            State of the linear solver.
        """
        if backend.ndim(A_normalized_search_dir) == 1:
            A_normalized_search_dir = backend.reshape(A_normalized_search_dir, (-1, 1))

        if solver_state.step == 0:
            A_update = linops.LowRankMatrix(U=A_normalized_search_dir)
        else:
            A_update = linops.LowRankMatrix(
                U=backend.hstack(
                    [solver_state.belief.A._summands[1].U, A_normalized_search_dir]
                )
            )
        return A_update

    def updated_linsys_belief(
        self,
        search_dir: backend.Array,
        A_search_dir: backend.Array,
        gram_pinv: backend.Scalar,
        solver_state: "probnum.linalg.solvers.LinearSolverState",
    ) -> beliefs.LinearSystemBelief:
        r"""Update the belief over the quantities of interest.

        Parameters
        ----------
        search_dir
            Search direction :math:`\Sigma_{i-1}As_i`.
        A_search_dir
            :math:`A`-multiplied search direction :math:`A\Sigma_{i-1}As_i`.
        gram_pinv
            Pseudo inverse of the Gramian, i.e. the normalization constant
            :math:`\fraction{1}{\sqrt{s_i^\top A \Sigma_{i-1} A s_i}}`
        solver_state
            State of the linear solver.
        """

        # Update belief about inverse matrix
        Ainv_update = self.Ainv_update(
            normalized_search_dir=search_dir * backend.sqrt(gram_pinv),
            solver_state=solver_state,
        )

        # Update belief about matrix
        A_update = self.A_update(
            A_normalized_search_dir=A_search_dir * backend.sqrt(gram_pinv),
            solver_state=solver_state,
        )

        # Update belief about solution
        x = randvars.Normal(
            mean=solver_state.belief.x.mean
            + solver_state.observation * gram_pinv * search_dir,
            cov=solver_state.prior.x.cov - Ainv_update,
        )

        return beliefs.LinearSystemBelief(
            x=x,
            A=solver_state.prior.A + A_update,
            Ainv=solver_state.prior.Ainv + Ainv_update,
            b=solver_state.belief.b,
        )
