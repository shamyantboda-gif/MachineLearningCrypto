"""Purged and embargoed walk-forward splitter.

``sklearn.model_selection.TimeSeriesSplit`` respects time order but does not
purge, which is why this exists.

Purging and embargo are two different things and it is worth being precise
about which does what, because the usual explanation conflates them:

**Purging** removes training rows whose *label* overlaps the test period. The
label at date ``t`` is a fact about ``t + horizon``. With a one day horizon,
the training row dated the day before the test window opens has a label drawn
from the first test day, so it has to go. Purging is governed by the label
horizon and nothing else. Backward-looking feature windows do not need purging,
however long they are, because a rolling mean ending at ``t`` contains no
information from after ``t``.

**Embargo** removes a further buffer of training rows immediately before the
test window. It has nothing to do with label overlap. It exists because
returns and especially volatility are serially correlated, so a training row
sitting right against the test boundary is close to being the same
observation as the first test row even when its label is clean.

The two stack: the gap between the last training date and the first test date
is ``horizon + embargo_days``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd

from src import schema


@dataclass
class Fold:
    """One walk-forward fold, expressed as boolean masks over the panel index."""

    fold_id: int
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_purged: int
    n_embargoed: int

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_val(self) -> int:
        return int(self.val_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())

    def __str__(self) -> str:
        return (
            f"fold {self.fold_id:>2d}  "
            f"train {self.train_start.date()}..{self.train_end.date()} ({self.n_train:>5d})  "
            f"val {self.val_start.date()}..{self.val_end.date()} ({self.n_val:>4d})  "
            f"test {self.test_start.date()}..{self.test_end.date()} ({self.n_test:>4d})  "
            f"purged {self.n_purged} embargoed {self.n_embargoed}"
        )


class PurgedWalkForward:
    """Expanding or rolling walk-forward splits with purging and embargo.

    Parameters mirror the ``splits`` block of ``config/base.yaml``.

    ``scheme='expanding'`` keeps the training window anchored at
    ``train_start`` and lets it grow. ``scheme='rolling'`` slides a fixed
    ``rolling_train_years`` window forward. Running both is the robustness
    check: if they disagree sharply, the model is regime sensitive and the
    writeup has to say so.
    """

    def __init__(
        self,
        train_start: str | pd.Timestamp,
        first_test_start: str | pd.Timestamp,
        test_months: int = 3,
        step_months: int = 3,
        scheme: str = "expanding",
        rolling_train_years: int = 3,
        embargo_days: int = 5,
        horizon: int = 1,
        inner_val_frac: float = 0.2,
        min_train_rows: int = 500,
    ):
        if scheme not in {"expanding", "rolling"}:
            raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
        if not 0.0 < inner_val_frac < 0.5:
            raise ValueError("inner_val_frac must sit between 0 and 0.5")

        self.train_start = pd.Timestamp(train_start, tz="UTC")
        self.first_test_start = pd.Timestamp(first_test_start, tz="UTC")
        self.test_months = test_months
        self.step_months = step_months
        self.scheme = scheme
        self.rolling_train_years = rolling_train_years
        self.embargo_days = embargo_days
        self.horizon = horizon
        self.inner_val_frac = inner_val_frac
        self.min_train_rows = min_train_rows

    @property
    def gap_days(self) -> int:
        """Calendar days removed between the last training row and the test window."""
        return self.horizon + self.embargo_days

    def split(self, index: pd.MultiIndex) -> Iterator[Fold]:
        """Yield folds for a panel index of (asset, date) pairs."""
        if list(index.names) != schema.INDEX_NAMES:
            raise ValueError(f"expected index names {schema.INDEX_NAMES}, got {list(index.names)}")

        dates = pd.Series(index.get_level_values(schema.DATE), index=index)
        last_date = dates.max()

        fold_id = 0
        test_start = self.first_test_start

        while test_start < last_date:
            test_end = test_start + pd.DateOffset(months=self.test_months)

            # Purge first, then embargo. Keeping them separate lets the fold
            # report how many rows each one cost, which is what makes the
            # leakage test able to check them independently.
            purge_boundary = test_start - pd.Timedelta(days=self.horizon)
            embargo_boundary = purge_boundary - pd.Timedelta(days=self.embargo_days)

            window_start = (
                self.train_start
                if self.scheme == "expanding"
                else max(self.train_start, test_start - pd.DateOffset(years=self.rolling_train_years))
            )

            in_window = (dates >= window_start) & (dates < test_start)
            after_purge = in_window & (dates < purge_boundary)
            after_embargo = in_window & (dates < embargo_boundary)

            n_purged = int(in_window.sum() - after_purge.sum())
            n_embargoed = int(after_purge.sum() - after_embargo.sum())

            full_train = after_embargo
            test_mask = (dates >= test_start) & (dates < test_end)

            if full_train.sum() >= self.min_train_rows and test_mask.sum() > 0:
                train_mask, val_mask, val_start = self._carve_validation(dates, full_train)
                train_dates = dates[train_mask]
                val_dates = dates[val_mask]
                test_dates = dates[test_mask]

                yield Fold(
                    fold_id=fold_id,
                    train_mask=train_mask,
                    val_mask=val_mask,
                    test_mask=test_mask,
                    train_start=train_dates.min(),
                    train_end=train_dates.max(),
                    val_start=val_dates.min() if len(val_dates) else val_start,
                    val_end=val_dates.max() if len(val_dates) else val_start,
                    test_start=test_dates.min(),
                    test_end=test_dates.max(),
                    n_purged=n_purged,
                    n_embargoed=n_embargoed,
                )
                fold_id += 1

            test_start = test_start + pd.DateOffset(months=self.step_months)

    def _carve_validation(
        self, dates: pd.Series, full_train: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Timestamp]:
        """Split the training window into an inner train and validation part.

        The validation slice is the most recent ``inner_val_frac`` of the
        training window by calendar date, not by row count, so every asset
        sees the same validation period. The boundary between inner train and
        validation is purged and embargoed exactly like the outer boundary,
        otherwise early stopping is tuned on rows that overlap the data it is
        stopping on.
        """
        train_dates = dates[full_train]
        unique_dates = pd.DatetimeIndex(sorted(train_dates.unique()))
        cut = unique_dates[int(len(unique_dates) * (1.0 - self.inner_val_frac))]

        val_mask = full_train & (dates >= cut)
        inner_boundary = cut - pd.Timedelta(days=self.gap_days)
        train_mask = full_train & (dates < inner_boundary)

        return train_mask, val_mask, cut

    def describe(self, index: pd.MultiIndex) -> pd.DataFrame:
        """Summarise every fold as a table, useful in the report and in tests."""
        rows = []
        for fold in self.split(index):
            rows.append(
                {
                    "fold": fold.fold_id,
                    "train_start": fold.train_start.date(),
                    "train_end": fold.train_end.date(),
                    "n_train": fold.n_train,
                    "val_start": fold.val_start.date(),
                    "val_end": fold.val_end.date(),
                    "n_val": fold.n_val,
                    "test_start": fold.test_start.date(),
                    "test_end": fold.test_end.date(),
                    "n_test": fold.n_test,
                    "n_purged": fold.n_purged,
                    "n_embargoed": fold.n_embargoed,
                }
            )
        return pd.DataFrame(rows)


def splitter_from_config(config: dict) -> PurgedWalkForward:
    """Build a splitter from the ``splits`` and ``targets`` config blocks."""
    splits = config["splits"]
    return PurgedWalkForward(
        train_start=splits["train_start"],
        first_test_start=splits["first_test_start"],
        test_months=splits["test_months"],
        step_months=splits["step_months"],
        scheme=splits.get("scheme", "expanding"),
        rolling_train_years=splits.get("rolling_train_years", 3),
        embargo_days=splits.get("embargo_days", 5),
        horizon=config.get("targets", {}).get("horizon", 1),
        inner_val_frac=splits.get("inner_val_frac", 0.2),
        min_train_rows=splits.get("min_train_rows", 500),
    )
