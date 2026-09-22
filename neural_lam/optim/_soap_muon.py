"""Vendored copy of the SOAP-Muon optimizer.

SOAP-Muon applies one-sided (or two-sided) SOAP preconditioning and then
Muon-style orthogonalization of the resulting preconditioned gradient --
"iterative whitening". Source: ``github.com/nikhilvyas/SOAP_Muon``,
accompanying `Improving SOAP using Iterative Whitening and Muon
<https://nikhilvyas.github.io/SOAP_Muon.pdf>`_ (Vyas et al.).

That repository ships two variants of the same optimizer. This file follows
``nanogpt_optimizer.py`` (the more developed one: it has the square-root
correction, the Nesterov option and a configurable Newton-Schulz step count)
with one deliberate substitution, taken from ``olmo_optimimzer.py``: the
update is normalized to unit RMS (``sqrt(numel)`` norm) rather than being
given modded-nanogpt's Muon layer-wise scaling. The reason is that this repo
drives *all* parameters with a single learning rate and a single cosine
schedule. nanogpt's scaling yields update RMS ``1/sqrt(fan_in)``, which only
works when the remaining layers are driven by a separately tuned Adam
learning rate; unit RMS matches AdamW's update scale for every group, which
is also how the authors ran their single-learning-rate OLMo experiments.

Other edits relative to upstream:

* ``step()`` takes a closure and runs it under ``torch.enable_grad()``, as
  required by PyTorch Lightning's automatic optimization.
* Parameters that are not 2D (or that live in a group with
  ``orthogonalize=False``) get the plain SOAP update instead of
  ``print(...); exit(0)``. This is how the paired MLIP study splits its
  groups: hidden weight matrices are orthogonalized, embeddings, the output
  layer and 1D parameters are not.
* the projection type is recomputed every step (upstream did this too, in
  ``init_preconditioner2``) and kept out of the optimizer state. It must
  stay out: ``torch.optim.Optimizer.load_state_dict`` maps its cast over
  anything iterable, which turns a state string into a generator's repr and
  silently corrupts resumed runs.
* ``nestrov`` is spelled ``nesterov``; the ``type`` argument shadowing the
  builtin is named ``side``.
"""

# Third-party
import torch
from torch.optim import Optimizer


def _zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """Newton-Schulz iteration computing the zeroth power of ``G``.

    A quintic iteration whose coefficients are chosen to maximize the slope
    at zero. It does not produce exactly ``UV^T`` but something like
    ``US'V^T`` with ``S'_ii ~ Uniform(0.5, 1.5)``, which empirically does not
    hurt. Unlike Muon's version this runs in the dtype of ``G`` (fp32) rather
    than bf16, and adds two Newton-Schulz steps of the classic (1.5, -0.5)
    iteration to tighten the spectrum.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.clone()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B

    # Keeps the interaction with the update variance simple.
    for _ in range(2):
        X = 1.5 * X - 0.5 * (X @ X.T) @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class SoapMuon(Optimizer):
    """Implements the SOAP-Muon algorithm.

    Parameters
    ----------
    params : iterable
        Parameters to optimize or dicts defining parameter groups.
        ``orthogonalize`` can be set per group to pick which parameters get
        the Muon step on top of SOAP.
    lr : float
        Learning rate (default: 1e-3). Updates are unit-RMS, so this is on
        the same scale as an AdamW learning rate.
    betas : tuple of two floats
        Adam's beta parameters (default: (0.95, 0.98)).
    shampoo_beta : float
        If >= 0, used for the preconditioner moving average instead of
        ``betas[1]`` (default: -1).
    eps : float
        Adam's epsilon for numerical stability (default: 1e-15).
    weight_decay : float
        Decoupled weight decay coefficient (default: 0.0).
    correct_bias : bool
        Whether to use Adam bias correction (default: True).
    projection_type : {"full", "left", "right", "std", "identity"}
        Which side(s) to precondition. "full" preconditions both, "std"
        picks the smaller side (one-sided SOAP, as in the paper).
    precondition_frequency : int
        How often (in steps) to refresh the eigenbasis. Because the Muon
        step runs every step, this can be much larger than for plain SOAP
        (the paper uses 10-40) (default: 10).
    max_precond_dim : int
        Dimensions larger than this are left unpreconditioned.
    newton_schulz_steps : int
        Number of Newton-Schulz iterations (default: 5).
    nesterov : bool
        Use Nesterov momentum in the orthogonalized group (default: False).
    sqrt_correction : bool
        Apply the elementwise square root (the paper's singular-value power
        ``rho = 0.5``) after orthogonalization. This is the stable default;
        ``False`` corresponds to ``rho = 0``, which is cheaper in the
        original formulation but can be unstable (default: True).
    orthogonalize : bool
        Group default for whether parameters receive the Muon step. It is
        ignored (treated as False) for parameters that are not 2D.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.95, 0.98),
        shampoo_beta: float = -1,
        eps: float = 1e-15,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        projection_type: str = "full",
        precondition_frequency: int = 10,
        max_precond_dim: int = 10000,
        newton_schulz_steps: int = 5,
        nesterov: bool = False,
        sqrt_correction: bool = True,
        orthogonalize: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "shampoo_beta": shampoo_beta,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
            "projection_type": projection_type,
            "precondition_frequency": precondition_frequency,
            "max_precond_dim": max_precond_dim,
            "newton_schulz_steps": newton_schulz_steps,
            "nesterov": nesterov,
            "sqrt_correction": sqrt_correction,
            "orthogonalize": orthogonalize,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        ``closure`` is re-entered with grad enabled: under Lightning's
        automatic optimization it runs the backward pass and gradient
        clipping.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                # Determined by the gradient shape and the group options, so
                # recomputing it is free and keeps it out of the state (see
                # the module docstring).
                projection = self.projection_type(grad, group)
                shampoo_beta = (
                    group["shampoo_beta"]
                    if group["shampoo_beta"] >= 0
                    else group["betas"][1]
                )

                if "ortho_matrix" not in state:
                    self.init_preconditioner(grad, state, projection)
                    self.update_preconditioner(
                        grad,
                        state,
                        state["step"],
                        projection,
                        shampoo_beta,
                        group["precondition_frequency"],
                    )
                    # First step is skipped so that we never use the current
                    # gradients in the projection.
                    continue

                grad_projected = self.project(grad, state, projection)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # exp_avg is kept in the original space and projected on the
                # fly; exp_avg_sq lives in the projected space.
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).add_(
                    grad_projected.square(), alpha=(1.0 - beta2)
                )

                denom = exp_avg_sq.sqrt().add_(group["eps"])

                exp_avg_projected = self.project(exp_avg, state, projection)

                step_size = group["lr"]
                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** (state["step"])
                    bias_correction2 = 1.0 - beta2 ** (state["step"])
                    step_size = (
                        step_size * (bias_correction2**0.5) / bias_correction1
                    )

                # Newton-Schulz needs a matrix; everything else (1D params,
                # and any group opting out) takes the plain SOAP update.
                orthogonalize = group["orthogonalize"] and p.dim() == 2

                if group["nesterov"] and orthogonalize:
                    update = (
                        exp_avg_projected + (1.0 - beta1) * grad_projected
                    ) / denom
                else:
                    update = exp_avg_projected / denom

                if orthogonalize:
                    update = _zeropower_via_newtonschulz5(
                        update, steps=group["newton_schulz_steps"]
                    )
                    if group["sqrt_correction"]:
                        update = torch.sqrt(torch.abs(update)) * torch.sign(
                            update
                        )

                update = self.project_back(update, state, projection)
                # Unit-RMS update, i.e. sqrt(numel) norm (see module docstring)
                update = update / (1e-30 + torch.mean(update**2) ** 0.5)

                p.add_(update, alpha=-step_size)

                # Decoupled (AdamW-style) weight decay: decay the weights in a
                # way that does not interact with the m/v estimates.
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                self.update_preconditioner(
                    grad,
                    state,
                    state["step"],
                    projection,
                    shampoo_beta,
                    group["precondition_frequency"],
                )

        return loss

    # The projection code below is a modified version of GaLore's projector,
    # github.com/jiaweizzhao/GaLore (galore_torch/galore_projector.py).

    @staticmethod
    def projection_type(grad, group):
        """Which side(s) of this parameter to precondition.

        A pure function of the gradient shape and the group options, so it is
        recomputed rather than stored in the optimizer state.
        """
        projection_type = group["projection_type"]
        max_precond_dim = group["max_precond_dim"]

        if grad.dim() != 2:
            # Newton-Schulz and the Kronecker factors both need a matrix.
            return "identity"
        if projection_type == "std":
            # One-sided SOAP on the smaller side, as in the paper.
            if grad.shape[0] < grad.shape[1]:
                return "left"
            return "right"
        if projection_type == "full" and max_precond_dim >= 0:
            too_tall = grad.shape[0] > max_precond_dim
            too_wide = grad.shape[1] > max_precond_dim
            if too_tall and too_wide:
                return "identity"
            if too_tall:
                return "right"
            if too_wide:
                return "left"
        return projection_type

    def init_preconditioner(self, grad, state, projection_type):
        """Allocate the preconditioner matrices (L and R in the paper)."""
        if projection_type == "right":
            state["GG"] = torch.zeros(
                grad.shape[1], grad.shape[1], device=grad.device
            )
        elif projection_type == "left":
            state["GG"] = torch.zeros(
                grad.shape[0], grad.shape[0], device=grad.device
            )
        elif projection_type == "full":
            state["GG"] = [
                torch.zeros(grad.shape[0], grad.shape[0], device=grad.device),
                torch.zeros(grad.shape[1], grad.shape[1], device=grad.device),
            ]

        state["ortho_matrix"] = None

    def project(self, grad, state, projection_type):
        """Project the gradient into the preconditioner's eigenbasis."""
        if projection_type == "identity":
            return grad
        if projection_type == "left":
            return torch.matmul(state["ortho_matrix"].t(), grad)
        if projection_type == "right":
            return torch.matmul(grad, state["ortho_matrix"].t())
        return (
            torch.matmul(state["ortho_matrix"][0].t(), grad)
            @ state["ortho_matrix"][1].t()
        )

    def project_back(self, grad, state, projection_type):
        """Project the update back to the original space."""
        if projection_type == "identity":
            return grad
        if projection_type == "right":
            return torch.matmul(grad, state["ortho_matrix"])
        if projection_type == "left":
            return torch.matmul(state["ortho_matrix"], grad)
        return (
            torch.matmul(state["ortho_matrix"][0], grad)
            @ state["ortho_matrix"][1]
        )

    def update_preconditioner(
        self,
        grad,
        state,
        step,
        projection_type,
        shampoo_beta,
        precondition_frequency,
    ):
        """Update L/R and, every ``precondition_frequency`` steps, their
        eigenbases."""
        refresh = step > 0 and step % precondition_frequency == 0
        if projection_type == "right":
            state["GG"].lerp_(grad.T @ grad, 1 - shampoo_beta)
            if state["ortho_matrix"] is None:
                state["ortho_matrix"] = self.get_orthogonal_matrix(
                    state["GG"], side="right"
                )
            if refresh:
                state["ortho_matrix"] = self.get_orthogonal_matrix_qr(
                    state, side="right"
                )
        elif projection_type == "left":
            state["GG"].lerp_(grad @ grad.T, 1 - shampoo_beta)
            if state["ortho_matrix"] is None:
                state["ortho_matrix"] = self.get_orthogonal_matrix(
                    state["GG"], side="left"
                )
            if refresh:
                state["ortho_matrix"] = self.get_orthogonal_matrix_qr(
                    state, side="left"
                )
        elif projection_type == "full":
            state["GG"][0].lerp_(grad @ grad.T, 1 - shampoo_beta)
            state["GG"][1].lerp_(grad.T @ grad, 1 - shampoo_beta)
            if state["ortho_matrix"] is None:
                state["ortho_matrix"] = self.get_orthogonal_matrix(
                    state["GG"], side="full"
                )
            if refresh:
                state["ortho_matrix"][0] = self.get_orthogonal_matrix_qr(
                    state, side="left", full=True
                )
                state["ortho_matrix"][1] = self.get_orthogonal_matrix_qr(
                    state, side="right", full=True
                )

    def get_orthogonal_matrix(self, weights, side):
        """Eigenbasis of the preconditioner, via ``eigh``."""
        if side == "full":
            float_data = weights[0].data.dtype == torch.float
            original_type = weights[0].data.dtype
            original_device = weights[0].data.device
            matrix = [weights[0].data.float(), weights[1].data.float()]
        else:
            float_data = weights.data.dtype == torch.float
            original_type = weights.data.dtype
            original_device = weights.data.device
            matrix = weights.data.float()

        if side == "right":
            _, Q = torch.linalg.eigh(matrix)
            B = torch.flip(Q, [1]).T
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        if side == "left":
            _, Q = torch.linalg.eigh(matrix)
            A = torch.flip(Q, [1])
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        if side == "full":
            _, Q = torch.linalg.eigh(matrix[0])
            A = torch.flip(Q, [1])
            _, Q = torch.linalg.eigh(matrix[1])
            B = torch.flip(Q, [1]).T
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        raise ValueError("side should be left, right or full")

    def get_orthogonal_matrix_qr(self, state, side, full=False):
        """Refresh the eigenbasis with one round of power iteration followed
        by a QR decomposition, reordering ``exp_avg_sq`` to match."""
        preconditioner = state["GG"]
        orth = state["ortho_matrix"]
        if full:
            index = 0 if side == "left" else 1
            preconditioner = state["GG"][index]
            orth = state["ortho_matrix"][index]

        float_data = preconditioner.data.dtype == torch.float
        original_type = preconditioner.data.dtype
        original_device = preconditioner.data.device
        matrix = preconditioner.data.float()
        orth_matrix = orth.data.float()

        if side == "right":
            est_eig = torch.diag(orth_matrix @ matrix @ orth_matrix.T)
            sort_idx = torch.argsort(est_eig, descending=True)
            state["exp_avg_sq"] = state["exp_avg_sq"].T[sort_idx].T
            orth_matrix = orth_matrix[sort_idx].T
            power_iter = (orth_matrix.T @ matrix).T
            Q, _ = torch.linalg.qr(power_iter)
            B = Q.T
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        if side == "left":
            est_eig = torch.diag(orth_matrix.T @ matrix @ orth_matrix)
            sort_idx = torch.argsort(est_eig, descending=True)
            state["exp_avg_sq"] = state["exp_avg_sq"][sort_idx]
            orth_matrix = orth_matrix.T[sort_idx].T
            power_iter = (orth_matrix.T @ matrix).T
            Q, _ = torch.linalg.qr(power_iter)
            A = Q
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        raise ValueError("side should be left or right")
