# ⚡ Day-Ahead Power Price Forecaster

> Prévision du prix horaire de l'électricité sur le marché *day-ahead*, et **backtest d'une stratégie de trading**, le tout **industrialisé en MLOps** (API, Docker, MLflow, CI/CD, monitoring).

Ce projet n'est pas un notebook : c'est un **modèle ML packagé comme un vrai logiciel de production**. Il est pensé pour le métier des équipes de *trading / Global Energy Management* (prévoir le prix du lendemain, prendre une position, mesurer le PnL et le risque).

---

## 🎯 Ce que le projet démontre

| Compétence | Où la voir dans le code |
|---|---|
| **POO & design patterns** | `DataSource`, `Forecaster`, `TradingStrategy`, `FeatureTransformer` sont des **classes abstraites** ; **Strategy**, **Factory** (`create_model`, `get_data_source`), **Pipeline** composable, injection de dépendances |
| **ML séries temporelles** | Feature engineering (lags, calendrier cyclique, fondamentaux), LightGBM **quantile** → intervalles de prédiction (= notion de risque) |
| **Évaluation honnête** | Backtest **walk-forward** (train passé → test futur), aucune fuite de données |
| **Domaine trading** | Stratégies long/short, PnL, Sharpe, max drawdown, bandes de confiance |
| **Industrialisation (MLOps)** | API **FastAPI** + **Pydantic**, **Docker** multi-stage, **MLflow** (tracking + registry), **CI/CD GitHub Actions**, **monitoring** (Prometheus + détection de *drift* PSI) |
| **Qualité d'ingénieur** | Tests **pytest** sur chaque couche, config typée, linting **ruff**, `Makefile` |

---

## 🏗️ Architecture

Le code est organisé en **couches découplées**. Chaque couche dépend d'une *abstraction*, pas d'une implémentation concrète — c'est ce qui rend le système testable et évolutif.

```
src/power_forecaster/
├── data/            # Couche données — d'OÙ viennent les données
│   ├── base.py          DataSource (classe abstraite + validation du schéma)
│   ├── synthetic.py     Générateur réaliste hors-ligne (déterministe → CI)
│   └── entsoe.py        Adaptateur vraies données ENTSO-E (même interface)
├── features/        # Feature engineering — pattern Pipeline composable
│   ├── base.py          FeatureTransformer (ABC) + FeaturePipeline
│   └── transformers.py  Calendrier, lags, rolling, fondamentaux
├── models/          # Modèles — pattern Strategy + Factory
│   ├── base.py          Forecaster (ABC) + Prediction (médiane + intervalle)
│   ├── baseline.py      Seasonal-naive (la référence à battre)
│   ├── gbm.py           LightGBM quantile
│   └── __init__.py      Registry + factory create_model()
├── backtest/        # Évaluation walk-forward + stratégies de trading
│   ├── engine.py        WalkForwardBacktester (agnostique au modèle)
│   ├── strategy.py      TradingStrategy (ABC) + directionnelle / confidence-gated
│   └── metrics.py       RMSE, sMAPE, pinball, Sharpe, max drawdown
├── training/        # Pipeline d'entraînement + MLflow
│   └── train.py         Load → features → backtest → fit → log + register
├── serving/         # Mise en production
│   ├── api.py           FastAPI (/predict, /health, /ready, /metrics)
│   ├── schemas.py       Contrats d'entrée/sortie Pydantic
│   └── predictor.py     Chargement du modèle + features au serving
├── monitoring.py    # Détection de drift (Population Stability Index)
├── config.py        # Configuration typée (pydantic-settings)
└── cli.py           # Point d'entrée : ppf train / backtest / serve
```

### Le fil rouge POO : *programmer contre des interfaces*

L'exemple le plus parlant est la couche données. Tout le système consomme un `DataSource` **abstrait** :

```python
class DataSource(ABC):
    @abstractmethod
    def _fetch(self, history_days: int) -> pd.DataFrame: ...
    def load(self, history_days: int) -> pd.DataFrame:   # valide le schéma
```

Il existe deux implémentations : `SyntheticDataSource` (hors-ligne, pour développer/tester) et `EntsoeDataSource` (vraies données marché). **Passer de l'une à l'autre = changer une variable d'environnement** (`PPF_DATA_SOURCE=entsoe`), sans toucher au reste. C'est le principe d'**inversion des dépendances** : on conçoit pour le changement.

Le même schéma se répète pour les **modèles** (`Forecaster`) et les **stratégies** (`TradingStrategy`) → on peut comparer/échanger les algorithmes sans modifier le moteur de backtest.

---

## 🚀 Démarrage rapide

```bash
# 1. Installer (Python 3.11+)
make install            # ou: pip install -e ".[dev]"

# 2. Entraîner + backtester + tout logger dans MLflow
make train              # ou: ppf train --model lightgbm

# 3. Comparer le modèle ML à la baseline
ppf backtest --model lightgbm

# 4. Lancer l'API de prédiction
make serve              # http://localhost:8000/docs (Swagger)

# 5. Tester l'API avec un payload prêt à l'emploi
ppf sample-request > req.json
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @req.json
```

### Tout lancer avec Docker (API + serveur MLflow)

```bash
docker compose up --build
# API     → http://localhost:8000/docs
# MLflow  → http://localhost:5000
```

---

## 🔌 L'API

| Route | Description |
|---|---|
| `POST /predict` | Prévision horaire avec intervalle de confiance + indicateur de drift |
| `GET /health` | Liveness probe (le process tourne) |
| `GET /ready` | Readiness probe (le modèle est chargé) |
| `GET /metrics` | Métriques Prometheus (latence, volume, drift) |
| `GET /docs` | Documentation Swagger auto-générée |

Exemple de réponse :

```json
{
  "model_name": "lightgbm",
  "predictions": [
    {"timestamp": "2024-12-31T23:00:00+00:00", "median": 58.4, "lower": 49.1, "upper": 71.8}
  ],
  "drift_psi": 0.03,
  "drift_alert": false
}
```

---

## 🔬 Le volet MLOps en détail

- **MLflow** — chaque entraînement logge params, métriques (RMSE, Sharpe, PnL…), feature importances, le modèle sérialisé et un échantillon de référence ; le modèle est **enregistré** (versionné) pour permettre rollback et promotion.
- **Monitoring** — l'API calcule à chaque requête le **PSI** entre les prix reçus et la distribution d'entraînement. Un PSI > 0,2 lève `drift_alert: true` → signal pour ré-entraîner.
- **CI/CD** — GitHub Actions : lint (ruff) → tests + couverture → smoke-test du pipeline → build Docker.
- **Reproductibilité** — données synthétiques *seedées*, config typée, image Docker multi-stage tournant en utilisateur non-root.

---

## 🧪 Tests

```bash
make test     # pytest + couverture sur les 6 couches
```

Couverture : data (schéma, déterminisme, fuites), features (pas de look-ahead), models (intervalles ordonnés, save/load, ML > baseline), backtest (PnL, Sharpe), monitoring (PSI), API (bout-en-bout via TestClient).

---

## 📝 Limites assumées (honnêteté technique)

- Les données par défaut sont **synthétiques** (mais structurellement réalistes) ; l'adaptateur ENTSO-E est fourni pour brancher des données réelles.
- Le marché de trading est un **modèle-jouet** (1 unité long/short vs prix de référence) : l'objectif est de démontrer la chaîne ML→prod, pas une stratégie déployable.
- Au serving, les statistiques *rolling* sont approximées depuis les lags fournis (documenté dans `predictor.py`).

---

*Stack : Python 3.11 · LightGBM · pandas · MLflow · FastAPI · Pydantic · Docker · pytest · GitHub Actions · ruff*
