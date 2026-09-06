"""Shared machinery for the three PyTorch families.

Everything that is genuinely common to a sequence model lives here so that
``lstm.py``, ``cnn.py`` and ``dlinear.py`` contain only their architecture.
The three hard parts are all solved once, in this module:

1. Turning a ``(asset, date)`` indexed feature matrix into
   ``(batch, lookback, n_features)`` windows that never straddle two assets.
2. Giving the test window the history it is entitled to, which lives in the
   training set, without letting anything from the future in.
3. A stable asset code mapping so the embedding table means the same thing at
   fit time and at predict time.

A note on the target scale. For the regression targets the network is trained
on a standardised target and ``predict`` inverts the transform, because
``base.py`` promises the point forecast is on the target's own scale. The
reason to standardise at all is that ``SmoothL1Loss`` has ``beta=1.0``: on raw
daily log returns, which are on the order of 0.03, every residual sits in the
quadratic region and huber silently degenerates into MSE, which defeats the
point of asking for huber. On the standardised target ``beta=1.0`` means one
standard deviation, which is the intended behaviour. Classification targets are
never standardised.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.base import CLASSIFICATION, REGRESSION, FoldData, Model
from src.schema import ASSET, DATE

# Label used for the extra embedding row that catches assets never seen during
# fit. Keeping a dedicated slot is cheaper than crashing on an unseen asset.
UNKNOWN_ASSET = "<unknown>"


def set_torch_seed(seed: int) -> None:
    """Seed every generator that can change a fit's output.

    ``torch.use_deterministic_algorithms`` is deliberately not called: this
    project is CPU only, where the default kernels are already deterministic,
    and switching it on makes unrelated ops raise.
    """
    import random

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)


@dataclass
class AssetCodec:
    """The stable asset code mapping, built once at fit time.

    ``asset_codes`` is the mapping the embedding table is indexed by. It is
    built from the sorted unique assets present in the training fold, so the
    same asset gets the same row in every fold and every seed. Anything not in
    it maps to ``unknown_code``, which is the last row of the table.
    """

    asset_codes: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_labels(cls, labels: object) -> "AssetCodec":
        unique = sorted({str(label) for label in np.asarray(labels).ravel().tolist()})
        return cls(asset_codes={name: code for code, name in enumerate(unique)})

    @property
    def unknown_code(self) -> int:
        return len(self.asset_codes)

    @property
    def n_codes(self) -> int:
        """Rows the embedding table needs, including the unknown slot."""
        return len(self.asset_codes) + 1

    def encode(self, labels: object) -> np.ndarray:
        unknown = self.unknown_code
        flat = np.asarray(labels).ravel().tolist()
        return np.asarray([self.asset_codes.get(str(label), unknown) for label in flat], dtype=np.int64)


def _per_asset_arrays(frame: pd.DataFrame, columns: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Split a frame into ``{asset: (dates, values)}`` for cheap lookups."""
    if frame is None or frame.empty:
        return {}
    aligned = frame.reindex(columns=columns)
    labels = np.asarray(aligned.index.get_level_values(ASSET)).astype(str)
    dates = aligned.index.get_level_values(DATE).to_numpy()
    values = np.nan_to_num(aligned.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in pd.unique(labels):
        rows = np.flatnonzero(labels == name)
        out[str(name)] = (dates[rows], values[rows])
    return out


def make_windows(
    X: pd.DataFrame,
    lookback: int,
    history: pd.DataFrame | None = None,
    *,
    codec: AssetCodec | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.MultiIndex]:
    """Build one lookback window per row of ``X``.

    Returns ``(windows, asset_codes, index)`` where ``windows`` is
    ``(len(X), lookback, n_features)`` float32, ``asset_codes`` is an int64
    array for the embedding, and ``index`` is ``X.index`` row for row in the
    same order. The caller can therefore line predictions straight back up
    against ``X`` without a join.

    A window is the previous ``lookback`` ROWS for that asset, not the previous
    ``lookback`` calendar days. The panel is not a perfect daily grid: a handful
    of gaps survive the build and are reported by the validator. Defining the
    window positionally keeps every window the same length and treats a gap the
    way a practitioner would, as the previous observation. Pretending the index
    is contiguous would be the actual error.

    ``history`` supplies rows that precede ``X`` for the same asset. Only rows
    strictly earlier than the first date of that asset in ``X`` are used, and at
    most ``lookback`` of them.

    ``codec`` is optional. When it is omitted the codes are derived from the
    sorted unique assets in ``X``, which is fine for a standalone call but not
    for a fitted model, which must pass the codec it learned at fit time.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if X.empty:
        raise ValueError("cannot build windows from an empty frame")

    columns = list(X.columns)
    n_rows = len(X)
    n_features = len(columns)

    if codec is None:
        codec = AssetCodec.from_labels(X.index.get_level_values(ASSET))

    labels = np.asarray(X.index.get_level_values(ASSET)).astype(str)
    dates = X.index.get_level_values(DATE).to_numpy()
    values = np.nan_to_num(X.to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    history_blocks = _per_asset_arrays(history, columns) if history is not None else {}

    windows = np.zeros((n_rows, lookback, n_features), dtype=np.float32)
    filled = np.zeros(n_rows, dtype=bool)

    for name in pd.unique(labels):
        # Positions of this asset's rows inside X. Grouping on the asset level
        # is the whole defence against a window spanning two assets: every
        # window is cut from this block and nothing else.
        rows = np.flatnonzero(labels == name)
        block = values[rows]
        first_date = dates[rows[0]]

        past = np.empty((0, n_features), dtype=np.float32)
        if str(name) in history_blocks:
            hist_dates, hist_values = history_blocks[str(name)]
            # Strictly earlier than the first row of X for this asset. Those
            # rows are training or validation observations that the model has
            # already been allowed to see, so handing them back as context is
            # the model being given the past it is entitled to, not leakage.
            # Nothing at or after the first test date is ever admitted here.
            earlier = np.flatnonzero(hist_dates < first_date)
            if earlier.size:
                past = hist_values[earlier[-lookback:]]

        combined = np.vstack([past, block]) if past.size else block
        offset = combined.shape[0] - block.shape[0]

        # Left pad with zeros so that early rows still produce a full window.
        # Zero is the neutral value after per fold standardisation, so a padded
        # timestep contributes nothing rather than a made up observation. Rows
        # near the start of the training fold, and any asset with no history
        # available at predict time, are padded this way.
        padded = np.zeros((lookback - 1 + combined.shape[0], n_features), dtype=np.float32)
        padded[lookback - 1 :] = combined

        strided = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=0)
        starts = offset + np.arange(block.shape[0])
        windows[rows] = np.ascontiguousarray(strided[starts].transpose(0, 2, 1))
        filled[rows] = True

    # Coverage check. If an asset group were skipped or written twice this is
    # where it shows up, before a silently wrong tensor reaches the network.
    assert filled.all(), "make_windows left rows unfilled"
    assert windows.shape == (n_rows, lookback, n_features)

    asset_codes = codec.encode(labels)
    assert asset_codes.shape == (n_rows,)
    return windows, asset_codes, X.index


class SequenceDataset(Dataset):
    """Windows, asset codes and targets, as tensors."""

    def __init__(self, windows: np.ndarray, asset_codes: np.ndarray, y: np.ndarray | None = None) -> None:
        self.windows = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        self.asset_codes = torch.from_numpy(np.ascontiguousarray(asset_codes, dtype=np.int64))
        if y is None:
            self.y = torch.zeros(len(self.windows), dtype=torch.float32)
        else:
            self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[item], self.asset_codes[item], self.y[item]


def resolve_loss(task: str, loss_name: str, target: str | None = None) -> nn.Module:
    """Map a config loss name onto a criterion.

    ``bce`` gives ``BCEWithLogitsLoss`` for direction, ``huber`` gives
    ``SmoothL1Loss`` for return regression and ``mse`` gives ``MSELoss`` for
    volatility. ``auto`` resolves to bce for classification, mse for ``vol_1d``
    and huber for everything else regression shaped.
    """
    name = str(loss_name or "auto").lower()
    if name == "auto":
        if task == CLASSIFICATION:
            name = "bce"
        elif target == "vol_1d":
            name = "mse"
        else:
            name = "huber"

    if name == "bce":
        if task != CLASSIFICATION:
            raise ValueError("bce is only valid for a classification task")
        return nn.BCEWithLogitsLoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"unknown loss {loss_name!r}; expected one of auto, bce, huber, mse")


@dataclass
class TorchTrainer:
    """Adam, gradient clipping, early stopping, best state restored.

    Kept deliberately plain. There is no scheduler and no progress reporting:
    the runner fits this model several hundred times and anything printed per
    epoch would drown the log.
    """

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 12
    grad_clip: float = 1.0
    batch_size: int = 128
    seed: int = 42

    def fit(
        self,
        network: nn.Module,
        criterion: nn.Module,
        train_dataset: SequenceDataset,
        val_dataset: SequenceDataset | None = None,
    ) -> tuple[float, int]:
        """Train and return ``(best_validation_loss, last_epoch_run)``."""
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            generator=generator,
        )
        val_loader = None
        if val_dataset is not None and len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=max(self.batch_size, 256), shuffle=False, num_workers=0)

        optimiser = torch.optim.Adam(network.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float("inf")
        best_state = copy.deepcopy(network.state_dict())
        best_epoch = 0
        since_improved = 0
        epoch = 0

        for epoch in range(1, int(self.max_epochs) + 1):
            network.train()
            for windows, codes, targets in train_loader:
                optimiser.zero_grad(set_to_none=True)
                loss = criterion(network(windows, codes), targets)
                loss.backward()
                if self.grad_clip and self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(network.parameters(), self.grad_clip)
                optimiser.step()

            # With no validation rows there is nothing to early stop on, so the
            # training loss stands in. That happens only in degenerate folds.
            monitor_loader = val_loader if val_loader is not None else train_loader
            epoch_loss = _evaluate_loss(network, criterion, monitor_loader)

            if epoch_loss < best_loss - 1e-6:
                best_loss = epoch_loss
                best_state = copy.deepcopy(network.state_dict())
                best_epoch = epoch
                since_improved = 0
            else:
                since_improved += 1
                if since_improved >= int(self.patience):
                    break

        network.load_state_dict(best_state)
        self.best_epoch_ = best_epoch
        return best_loss, epoch


def _evaluate_loss(network: nn.Module, criterion: nn.Module, loader: DataLoader) -> float:
    network.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for windows, codes, targets in loader:
            loss = criterion(network(windows, codes), targets)
            total += float(loss) * len(targets)
            count += len(targets)
    return total / max(count, 1)


class BaseTorchModel(Model):
    """Fit and predict for any of the three sequence families.

    A concrete family supplies ``_build_network`` and nothing else. Capacity is
    held down on purpose everywhere below: roughly two thousand daily
    observations per asset against a signal to noise ratio near zero means an
    over parameterised network memorises the training fold and reports noise as
    a finding.
    """

    name = "torch"
    is_stochastic = True

    def _build_network(self, n_features: int, n_assets: int) -> nn.Module:
        """Return the family's network. ``n_assets`` includes the unknown slot."""
        raise NotImplementedError

    # -- configuration -------------------------------------------------

    @property
    def lookback(self) -> int:
        return int(self.params.get("lookback", 30))

    def _trainer(self) -> TorchTrainer:
        return TorchTrainer(
            lr=float(self.params.get("lr", 1e-3)),
            weight_decay=float(self.params.get("weight_decay", 1e-4)),
            max_epochs=int(self.params.get("max_epochs", 100)),
            patience=int(self.params.get("patience", 12)),
            grad_clip=float(self.params.get("grad_clip", 1.0)),
            batch_size=int(self.params.get("batch_size", 128)),
            seed=self.seed,
        )

    # -- fit -----------------------------------------------------------

    def fit(self, fold: FoldData) -> "BaseTorchModel":
        started = time.perf_counter()
        set_torch_seed(self.seed)

        # A refit must not serve logits cached by the previous one.
        self._cached_index = None
        self._cached_logits = None

        lookback = self.lookback
        self.feature_names_ = list(fold.X_train.columns)
        self.target_ = fold.target

        train_labels = fold.X_train.index.get_level_values(ASSET)
        if len(fold.X_val):
            train_labels = train_labels.append(fold.X_val.index.get_level_values(ASSET))
        self.codec_ = AssetCodec.from_labels(train_labels)

        train_windows, train_codes, _ = make_windows(fold.X_train, lookback, codec=self.codec_)
        y_train = fold.y_train.to_numpy(dtype=np.float64)

        val_windows = val_codes = None
        y_val = None
        if len(fold.X_val):
            # The validation windows get the training rows as history for the
            # same reason the test windows do: those rows are strictly earlier
            # and already seen, so withholding them would only degrade the
            # early stopping signal with thirty zero padded rows per asset.
            val_windows, val_codes, _ = make_windows(
                fold.X_val, lookback, history=fold.X_train, codec=self.codec_
            )
            y_val = fold.y_val.to_numpy(dtype=np.float64)

        # Target scaling is learned on the training rows only. Classification
        # keeps the raw 0/1 label because BCEWithLogitsLoss expects it.
        finite_train = np.isfinite(y_train)
        if self.task == CLASSIFICATION:
            self.y_mean_, self.y_std_ = 0.0, 1.0
        else:
            values = y_train[finite_train]
            self.y_mean_ = float(values.mean()) if values.size else 0.0
            std = float(values.std(ddof=0)) if values.size else 1.0
            self.y_std_ = std if std > 1e-12 else 1.0

        # Rows with a missing target are dropped only after windowing. They are
        # still legitimate history for later windows, so filtering them out of
        # X first would corrupt every window that spans them.
        train_dataset = SequenceDataset(
            train_windows[finite_train],
            train_codes[finite_train],
            (y_train[finite_train] - self.y_mean_) / self.y_std_,
        )
        val_dataset = None
        if val_windows is not None:
            finite_val = np.isfinite(y_val)
            if finite_val.any():
                val_dataset = SequenceDataset(
                    val_windows[finite_val],
                    val_codes[finite_val],
                    (y_val[finite_val] - self.y_mean_) / self.y_std_,
                )

        self.network_ = self._build_network(len(self.feature_names_), self.codec_.n_codes)
        criterion = resolve_loss(self.task, self.params.get("loss", "auto"), self.target_)
        trainer = self._trainer()
        self.best_val_loss_, self.stopped_epoch_ = trainer.fit(
            self.network_, criterion, train_dataset, val_dataset
        )
        self.best_epoch_ = getattr(trainer, "best_epoch_", self.stopped_epoch_)
        # Distinguish "patience expired" from "ran out of epochs". The second
        # case means the fit was truncated rather than converged, and it is the
        # only case where max_epochs is the binding constraint on runtime.
        hit_ceiling = self.stopped_epoch_ >= int(self.params.get("max_epochs", 100))

        # The history buffer handed to predict. Only the last ``lookback`` rows
        # per asset are needed, and every one of them predates the test window.
        seen = fold.X_train if not len(fold.X_val) else pd.concat([fold.X_train, fold.X_val])
        self.history_ = (
            seen.sort_index().groupby(level=ASSET, observed=True, group_keys=False).tail(lookback)
        )

        self.fitted_ = True
        print(
            f"{self.name} fold {fold.fold_id} seed {self.seed}: "
            f"best_val_loss={self.best_val_loss_:.5f} best_epoch={self.best_epoch_} "
            f"stopped_epoch={self.stopped_epoch_}{' hit_max_epochs' if hit_ceiling else ''} "
            f"n_train={len(train_dataset)} secs={time.perf_counter() - started:.1f}"
        )
        return self

    # -- predict ---------------------------------------------------------

    def _logits(self, X: pd.DataFrame) -> np.ndarray:
        """Forward pass, cached per ``X.index``.

        ``run_fold`` calls ``predict`` and then ``predict_proba`` on the same
        frame. Caching means the network runs once rather than twice.
        """
        if not getattr(self, "fitted_", False):
            raise RuntimeError(f"{self.name}.predict called before fit")

        cached_index = getattr(self, "_cached_index", None)
        if cached_index is not None and len(cached_index) == len(X) and cached_index.equals(X.index):
            return self._cached_logits

        frame = X if list(X.columns) == self.feature_names_ else X.reindex(columns=self.feature_names_)
        windows, codes, index = make_windows(
            frame, self.lookback, history=self.history_, codec=self.codec_
        )
        assert index.equals(X.index)

        dataset = SequenceDataset(windows, codes)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        self.network_.eval()
        chunks = []
        with torch.no_grad():
            for batch_windows, batch_codes, _ in loader:
                chunks.append(self.network_(batch_windows, batch_codes).numpy())
        logits = np.concatenate(chunks).astype(np.float64)

        self._cached_index = X.index
        self._cached_logits = logits
        return logits

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        logits = self._logits(X)
        if self.task == CLASSIFICATION:
            return (logits >= 0.0).astype(float)
        # Back onto the target's own scale, which is what base.py promises.
        return logits * self.y_std_ + self.y_mean_

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray | None:
        if self.task != CLASSIFICATION:
            return None
        # float64 sigmoid: the float32 one saturates to exactly 1.0 near a
        # logit of 17, and a probability of exactly 1 breaks log loss.
        return 1.0 / (1.0 + np.exp(-self._logits(X)))


def build_torch_model(family: str, params: dict, task: str, seed: int) -> Model:
    """Instantiate one of the three families by name."""
    # Local import: each family module imports BaseTorchModel from here, so a
    # module level import in either direction would be circular.
    from src.models.cnn import CNNModel
    from src.models.dlinear import DLinearModel
    from src.models.lstm import LSTMModel

    registry: dict[str, type[Model]] = {
        "lstm": LSTMModel,
        "cnn": CNNModel,
        "dlinear": DLinearModel,
    }
    key = str(family).lower()
    if key not in registry:
        raise ValueError(f"unknown torch family {family!r}; expected one of {sorted(registry)}")
    return registry[key](params=params, task=task, seed=seed)


TORCH_FAMILIES = ("lstm", "cnn", "dlinear")

__all__ = [
    "ASSET",
    "CLASSIFICATION",
    "REGRESSION",
    "TORCH_FAMILIES",
    "UNKNOWN_ASSET",
    "AssetCodec",
    "BaseTorchModel",
    "SequenceDataset",
    "TorchTrainer",
    "build_torch_model",
    "make_windows",
    "resolve_loss",
    "set_torch_seed",
]
