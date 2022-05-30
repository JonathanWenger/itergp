"""Iterative approximation method with a mixed strategy."""

from typing import Tuple

from probnum import linalg
from probnum.linalg.solvers import (
    ProbabilisticLinearSolver,
    information_ops,
    stopping_criteria as pn_stopping_criteria,
)

from itergp.methods import belief_updates, policies, stopping_criteria


class MixedStrategy(ProbabilisticLinearSolver):
    r"""Iterative approximation method with a mixed strategy.

    Parameters
    ----------
    base_policies
        Policies which make up the :class:`MixedPolicy`.
    iters
        Until which iteration (non-inclusive) to use the policy in the corresponding
        position in ``base_policies``. Assumed to be sorted in increasing order. If
        ``iters`` has one fewer entry than ``base_policies``, the last policy is used
        for all remaining iterations.
    atol :
        Absolute tolerance.
    rtol :
        Relative tolerance.
    maxiter
        Maximum number of iterations.
    """

    def __init__(
        self,
        base_policies: Tuple[linalg.solvers.policies.LinearSolverPolicy],
        iters: Tuple[int],
        atol: float = 1e-6,
        rtol: float = 1e-6,
        maxiter: int = None,
    ):
        if len(iters) == len(base_policies) and maxiter is None:
            maxiter = iters[-1]

        super().__init__(
            policy=policies.MixedPolicy(base_policies=base_policies, iters=iters),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
            stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=maxiter, problem_size_factor=1
            )
            | pn_stopping_criteria.ResidualNormStoppingCriterion(atol=atol, rtol=rtol),
        )
