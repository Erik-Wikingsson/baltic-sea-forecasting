"""Regression tests for restoring training state mid-schedule."""

# Third-party
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

# First-party
from neural_lam.models.ar_model import ARModel
from neural_lam.optim import MuonAuxAdam


def _build(lr=1e-3, t_max=20):
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1)
    )
    opt = MuonAuxAdam(
        [p for p in net.parameters() if p.ndim == 2],
        [],
        [p for p in net.parameters() if p.ndim < 2],
        lr=lr,
        weight_decay=0.0,
        momentum=0.95,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=t_max, eta_min=1e-5
    )
    return net, opt, sched


def test_muon_aux_adam_restores_scheduled_lr():
    """The wrapper must round-trip its own (scheduled) LR, not just the
    sub-optimizer states. Without it a resumed run silently continues from
    the initial LR."""
    net, opt, sched = _build()
    for _ in range(7):
        opt.zero_grad()
        net(torch.randn(4, 8)).square().mean().backward()
        opt.step()
        sched.step()
    lr, opt_state, sched_state = (
        opt.param_groups[0]["lr"],
        opt.state_dict(),
        sched.state_dict(),
    )
    assert lr < 1e-3, "LR should have been annealed away from the initial LR"

    _, opt2, sched2 = _build()
    opt2.load_state_dict(opt_state)
    sched2.load_state_dict(sched_state)
    assert opt2.param_groups[0]["lr"] == lr

    # Momentum/moment estimates must come back too
    key = opt._muon.param_groups[0]["params"][0]
    key2 = opt2._muon.param_groups[0]["params"][0]
    assert torch.equal(
        opt._muon.state[key]["momentum_buffer"],
        opt2._muon.state[key2]["momentum_buffer"],
    )

    # ... and the schedule continues identically
    sched.step()
    sched2.step()
    assert opt.param_groups[0]["lr"] == opt2.param_groups[0]["lr"]


def test_legacy_muon_checkpoint_lr_recovered():
    """Checkpoints written before the wrapper saved its own param_groups have
    the scheduled LR recovered from the saved scheduler state."""
    net, opt, sched = _build()
    for _ in range(7):
        opt.zero_grad()
        net(torch.randn(4, 8)).square().mean().backward()
        opt.step()
        sched.step()
    legacy = {k: v for k, v in opt.state_dict().items() if k != "base"}
    checkpoint = {
        "optimizer_states": [legacy],
        "lr_schedulers": [sched.state_dict()],
    }
    ARModel._fix_legacy_muon_lr(checkpoint)

    _, opt2, _ = _build()
    with pytest.warns(UserWarning, match="no 'base' state"):
        opt2.load_state_dict(checkpoint["optimizer_states"][0])
    assert opt2.param_groups[0]["lr"] == sched.state_dict()["_last_lr"][0]


def test_fix_legacy_muon_lr_leaves_other_checkpoints_alone():
    modern = {
        "optimizer_states": [
            {"base": {}, "muon": {"param_groups": [{"lr": 0.5}]}}
        ],
        "lr_schedulers": [{"_last_lr": [0.1]}],
    }
    ARModel._fix_legacy_muon_lr(modern)
    assert modern["optimizer_states"][0]["muon"]["param_groups"][0][
        "lr"
    ] == 0.5

    adamw = {
        "optimizer_states": [{"state": {}, "param_groups": [{"lr": 0.5}]}],
        "lr_schedulers": [{"_last_lr": [0.1]}],
    }
    ARModel._fix_legacy_muon_lr(adamw)
    assert adamw["optimizer_states"][0]["param_groups"][0]["lr"] == 0.5

    ARModel._fix_legacy_muon_lr({})  # no optimizer/scheduler state at all


class _TinyModel(pl.LightningModule):
    """Minimal module using the same optimizer/scheduler setup as ARModel."""

    lrs = None

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(8, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1)
        )

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.lrs.append(
                (
                    self.current_epoch,
                    round(self.optimizers().param_groups[0]["lr"], 12),
                )
            )
        x, y = batch
        return (self.net(x) - y).square().mean()

    def configure_optimizers(self):
        opt = MuonAuxAdam(
            [p for p in self.parameters() if p.ndim == 2],
            [],
            [p for p in self.parameters() if p.ndim < 2],
            lr=1e-3,
            weight_decay=0.0,
            momentum=0.95,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=20, eta_min=1e-5
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }


def test_resume_mid_stage_continues_lr_schedule(tmp_path):
    """Interrupting and resuming a stage must reproduce the LR schedule of an
    uninterrupted run (the checkpoint is written at every train epoch end)."""
    torch.manual_seed(1)
    loader = DataLoader(
        TensorDataset(torch.randn(16, 8), torch.randn(16, 1)),
        batch_size=8,
        shuffle=False,
    )

    def _trainer(dirpath, max_epochs):
        callback = pl.callbacks.ModelCheckpoint(
            dirpath=dirpath,
            save_top_k=0,
            save_last=True,
            save_on_train_epoch_end=True,
        )
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[callback],
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator="cpu",
            deterministic=True,
        )
        return trainer, callback

    def _run(dirpath, max_epochs, ckpt_path=None):
        model = _TinyModel()
        model.lrs = []
        trainer, callback = _trainer(str(dirpath), max_epochs)
        trainer.fit(model, loader, ckpt_path=ckpt_path)
        return model.lrs, callback.last_model_path

    reference, _ = _run(tmp_path / "ref", 8)
    first_half, last_ckpt = _run(tmp_path / "run", 5)
    assert last_ckpt, "last.ckpt should be written at every train epoch end"
    second_half, _ = _run(tmp_path / "run", 8, ckpt_path=last_ckpt)

    assert first_half + second_half == reference
