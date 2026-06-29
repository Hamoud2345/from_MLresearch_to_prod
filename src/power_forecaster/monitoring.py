"""Détection de drift via le PSI (Population Stability Index).

En prod un modèle se dégrade sans prévenir quand la distribution des données
change par rapport à l'entraînement (nouveau régime de marché, hiver froid, price
cap...). Le PSI est un indicateur simple : on découpe en buckets un échantillon
de référence et un échantillon courant et on mesure de combien la population a
bougé. Au-dessus de ~0.2 on considère le drift significatif et il faut réentraîner.
"""

from __future__ import annotations

import numpy as np


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    """Calcule le PSI entre un échantillon de référence et un échantillon courant (1D)."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    # plancher pour éviter division par zéro / log(0) sur les buckets vides
    ref_pct = np.clip(ref_pct, 1e-4, None)
    cur_pct = np.clip(cur_pct, 1e-4, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
