"""Fold-local sklearn pipelines and a small, predeclared candidate search."""

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def candidates(config):
    """No early stopping on test; each complete estimator is fit inside its fold."""
    choices = {"Median_baseline": DummyRegressor(strategy="median")}
    for alpha in (10.0, 100.0):
        choices[f"Ridge_a{alpha:g}"] = Ridge(alpha=alpha)
    for leaf in (3, 8):
        choices[f"RandomForest_leaf{leaf}"] = RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=leaf,
            max_features=0.75,
            random_state=config.seed,
            n_jobs=config.n_jobs,
        )
    for depth, trees in ((3, 450), (4, 450), (4, 700)):
        choices[f"XGBoost_d{depth}_n{trees}"] = XGBRegressor(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=0.035,
            min_child_weight=8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=10.0,
            reg_alpha=0.2,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=config.seed,
            n_jobs=config.n_jobs,
        )
    return choices


def make_pipeline(estimator):
    # Pipeline.fit calls fit_transform on TRAIN ONLY; predict calls transform.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("variance", VarianceThreshold(threshold=1e-12)),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]
    )


def predict_nonnegative(pipeline, features):
    # Upper clipping is NOT applied: raw NASA truth must remain uncapped.
    return np.maximum(0.0, pipeline.predict(features))
