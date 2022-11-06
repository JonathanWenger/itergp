"""Pseudo-input Method."""

from probnum import backend
from probnum.linalg.solvers import ProbabilisticLinearSolver, information_ops

from itergp.methods import belief_updates, policies, stopping_criteria


class PseudoInput(ProbabilisticLinearSolver):
    r"""Pseudo-input approximation method.

    Parameters
    ----------
    pseudo_inputs
        Pseudo inputs, also known as inducing points, at which kernel function actions are centered.
    """

    def __init__(self, pseudo_inputs: backend.Array):
        self._pseudo_inputs = pseudo_inputs
        super().__init__(
            policy=policies.PseudoInputPolicy(pseudo_inputs=pseudo_inputs),
            information_op=information_ops.ProjectedResidualInformationOp(),
            belief_update=belief_updates.ProjectedResidualBeliefUpdate(),
            stopping_criterion=stopping_criteria.MaxIterationsStoppingCriterion(
                maxiter=pseudo_inputs.shape[0], problem_size_factor=1
            ),
        )

    @property
    def pseudo_inputs(self) -> backend.Array:
        """Pseudo inputs, at which kernel function actions are centered."""
        return self._pseudo_inputs
