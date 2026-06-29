"""Composable feature-engineering transformers.

Design note (interview-ready):
Each transformer does *one* thing and shares the same tiny interface
(``transform``). A :class:`FeaturePipeline` then composes them in order. This is
the same idea as scikit-learn pipelines and the classic "Pipe and Filter"
pattern: small, independently testable units assembled into a bigger one.
Adding a new feature = writing one class, no edits to existing code
(Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

TARGET = "price"


class FeatureTransformer(ABC):
    """Stateless transformation from a data frame to an augmented data frame."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return ``df`` with new feature columns added."""
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class FeaturePipeline:
    """Apply an ordered list of :class:`FeatureTransformer` objects."""

    def __init__(self, transformers: list[FeatureTransformer]) -> None:
        self.transformers = transformers

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for t in self.transformers:
            out = t.transform(out)
        return out

    def feature_names(self, df: pd.DataFrame) -> list[str]:
        """Column names produced by the pipeline, excluding raw/target columns."""
        transformed = self.transform(df)
        excluded = {"timestamp", TARGET}
        return [c for c in transformed.columns if c not in excluded]
