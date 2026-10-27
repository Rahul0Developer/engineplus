"""Deep sequence models (PyTorch): 1-D CNN, GRU and a small Transformer.

All three share one training loop (``_fit_torch_model``) and the same
normalised-target trick: fit on y scaled to roughly [0, 1] so plain MSE with
a sigmoid-free linear head behaves. Weights are saved as state dicts plus a
JSON sidecar — no pickled classes, which matters when an API loads them.

Install with `pip install 'enginepulse[deep]'` (pulls torch). The module
raises a clear ImportError at construction time if torch is absent, so the
rest of the package never touches it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from enginepulse.models.base import RULModel, clip_rul
from enginepulse.models.registry import register

try:
    import torch
    from torch import nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in slim installs
    _HAS_TORCH = False


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError("deep models need torch; install with pip install 'enginepulse[deep]'")


if _HAS_TORCH:

    class _CNNRegressor(nn.Module):
        """Three conv blocks + global pooling + linear head."""

        def __init__(self, n_features: int, hidden: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(n_features, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):  # x: (b, w, f)
            return self.net(x.transpose(1, 2)).squeeze(-1)

    class _GRURegressor(nn.Module):
        def __init__(self, n_features: int, hidden: int = 64, layers: int = 1, dropout: float = 0.0, mc_dropout: bool = False):
            super().__init__()
            self.gru = nn.GRU(n_features, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.head = nn.Linear(hidden, 1)
            # MC-dropout mode keeps dropout active even during eval() when set.
            self.mc_dropout = mc_dropout

        def forward(self, x):
            out, _ = self.gru(x)
            h = out[:, -1, :]
            if self.training or getattr(self, "_mc_active", False):
                h = torch.nn.functional.dropout(h, p=0.2)
            return self.head(h).squeeze(-1)

    class _TransformerRegressor(nn.Module):
        """Compact encoder: value embedding + positional encoding + 2 layers."""

        def __init__(self, n_features: int, d_model: int = 32, heads: int = 4, layers: int = 2):
            super().__init__()
            self.proj = nn.Linear(n_features, d_model)
            self.pos = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=heads, dim_feedforward=2 * d_model,
                dropout=0.1, batch_first=True, activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x):
            b, w, _ = x.shape
            h = self.proj(x) + self.pos[:, :w, :]
            h = self.encoder(h)
            return self.head(h[:, -1, :]).squeeze(-1)

    def _fit_torch_model(
        model: "nn.Module",
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        device: str = "cpu",
    ) -> dict:
        """One shared Adam/MSE loop with early stopping on val RMSE (patience 5)."""
        torch.manual_seed(seed)
        dev = torch.device(device)
        model.to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        lossf = torch.nn.MSELoss()

        xt = torch.as_tensor(X, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
        ds = torch.utils.data.TensorDataset(xt, yt)
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

        hist = {"train_loss": [], "val_rmse": []}
        best_state, best_val, patience = None, float("inf"), 5
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = model(xb)
                loss = lossf(pred, yb)
                loss.backward()
                # Small grad clip: keeps GRU/Transformer runs stable without tuning.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += float(loss) * len(xb)
            hist["train_loss"].append(tot / len(ds))
            if X_val is not None and len(X_val):
                model.eval()
                with torch.no_grad():
                    pv = model(torch.as_tensor(X_val, dtype=torch.float32, device=dev)).cpu().numpy()
                # y_val arrives already scaled to [0,1]; early-stop metric is relative anyway.
                vrmse = float(np.sqrt(np.mean((pv - y_val) ** 2)))
                hist["val_rmse"].append(vrmse)
                if vrmse < best_val - 1e-4:
                    best_val, patience = vrmse, 5
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    patience -= 1
                    if patience <= 0:
                        break
        if best_state is not None:
            model.load_state_dict(best_state)
        return hist


class _TorchModelBase(RULModel):
    """Common fit/predict/save/load plumbing for the three deep models."""

    arch_kwargs = ("epochs", "batch_size", "lr")

    def __init__(self, seed: int = 42, epochs: int = 60, batch_size: int = 128,
                 lr: float = 1e-3, max_rul: float | None = 125.0, device: str = "cpu", **kw):
        super().__init__(seed=seed, epochs=epochs, batch_size=batch_size, lr=lr,
                         max_rul=max_rul, **kw)
        _require_torch()
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.max_rul = max_rul
        self.device = device
        self.model_: "nn.Module | None" = None

    def _build(self, n_features: int) -> "nn.Module":
        raise NotImplementedError

    def fit(self, X, y, X_val=None, y_val=None):
        scale = float(self.max_rul) if self.max_rul else float(max(y.max(), 1.0))
        ys = np.asarray(y, dtype=np.float32) / scale
        vs = None if y_val is None else np.asarray(y_val, dtype=np.float32) / scale
        self.y_scale_ = scale  # must exist before _build()/save() reference it
        self.model_ = self._build(X.shape[-1])
        self.hist_ = _fit_torch_model(
            self.model_, X, ys, X_val, vs,
            epochs=self.epochs, batch_size=self.batch_size, lr=self.lr,
            seed=self.seed, device=self.device,
        )
        self.is_fitted_ = True
        return self

    def predict(self, X):
        self._check_fitted()
        self.model_.eval()
        with torch.no_grad():
            out = self.model_(torch.as_tensor(X, dtype=torch.float32, device=self.device)).cpu().numpy()
        return clip_rul(out * self.y_scale_, self.max_rul)

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model_.state_dict(), p / "weights.pt")
        self._write_meta(p, {"y_scale": self.y_scale_, "n_features": int(self.model_input_dim_)})

    def load(self, path):
        p = Path(path)
        meta = json.loads((p / "meta.json").read_text())
        self.y_scale_ = meta["y_scale"]
        self.model_ = self._build(meta["n_features"])
        self.model_.load_state_dict(torch.load(p / "weights.pt", map_location=self.device))
        self.is_fitted_ = True
        return self


@register("cnn")
class CNNModel(_TorchModelBase):
    """1-D CNN over raw windows."""

    name = "cnn"

    def __init__(self, seed: int = 42, hidden: int = 32, **kw):
        super().__init__(seed=seed, hidden=hidden, **kw)
        self.hidden = hidden

    def _build(self, n_features: int):
        self.model_input_dim_ = n_features
        return _CNNRegressor(n_features, self.hidden)


@register("gru")
class GRUModel(_TorchModelBase):
    """Single-layer GRU with optional MC-dropout intervals."""

    name = "gru"

    def __init__(self, seed: int = 42, hidden: int = 64, layers: int = 1,
                 mc_dropout: bool = False, mc_samples: int = 25, **kw):
        super().__init__(seed=seed, hidden=hidden, layers=layers,
                         mc_dropout=mc_dropout, mc_samples=mc_samples, **kw)
        self.hidden, self.layers = hidden, layers
        self.mc_dropout, self.mc_samples = mc_dropout, mc_samples

    def _build(self, n_features: int):
        self.model_input_dim_ = n_features
        return _GRURegressor(n_features, self.hidden, self.layers,
                             dropout=0.0, mc_dropout=self.mc_dropout)

    def predict_quantiles(self, X, quantiles):
        """MC-dropout: run T stochastic forward passes, take empirical quantiles."""
        self._check_fitted()
        if not self.mc_dropout:
            return super().predict_quantiles(X, quantiles)
        xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        samples = []
        self.model_.train()          # re-enables dropout
        self.model_._mc_active = True
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = self.model_(xt).cpu().numpy() * self.y_scale_
                samples.append(out)
        self.model_.eval()
        self.model_._mc_active = False
        S = np.stack(samples)        # (T, n)
        return {q: clip_rul(np.quantile(S, q, axis=0), self.max_rul) for q in quantiles}


@register("transformer")
class TransformerModel(_TorchModelBase):
    """Small encoder-only transformer; keep it tiny — FD001 has ~16k windows."""

    name = "transformer"

    def __init__(self, seed: int = 42, d_model: int = 32, heads: int = 4, layers: int = 2, **kw):
        super().__init__(seed=seed, d_model=d_model, heads=heads, layers=layers, **kw)
        self.d_model, self.heads, self.layers = d_model, heads, layers

    def _build(self, n_features: int):
        self.model_input_dim_ = n_features
        return _TransformerRegressor(n_features, self.d_model, self.heads, self.layers)
