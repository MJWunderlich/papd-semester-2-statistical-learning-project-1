"""
Modelo de regresión para predicción de precios de energía eléctrica.
Entrena y testea solo con data horaria de 2023.

Train: Enero 2023 → Septiembre 2023
Test:  Octubre 2023 → Diciembre 2023

Uso: python train_hourly_2023.py
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Cargar datos
# ─────────────────────────────────────────────────────────────────────────────
print("=== CARGANDO DATOS HORARIOS 2023 ===")
data = pd.read_csv('../data/HourlyData-2023.csv', parse_dates=['dia'])
data = data.sort_values(['dia', 'hora']).reset_index(drop=True)
data = data.dropna()

print(f"Shape: {data.shape}")
print(f"Date range: {data['dia'].min().date()} → {data['dia'].max().date()}")
print(f"Precio describe:\n{data['precio'].describe()}")

# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────
EXCLUDE  = ['precio', 'dia', 'hora', 'dia_semanal',
            'dia_anual', 'dia_mensual', 'mes', 'year']
FEATURES = [c for c in data.columns if c not in EXCLUDE]
TARGET   = 'precio'

print(f"\nFeatures ({len(FEATURES)}):\n{FEATURES}")

# ─────────────────────────────────────────────────────────────────────────────
# Train / Test split — 75% train, 25% test chronologically
# ─────────────────────────────────────────────────────────────────────────────
split_date = pd.Timestamp('2023-10-01')
train_mask = data['dia'] < split_date
test_mask  = data['dia'] >= split_date

X = data[FEATURES]
y = data[TARGET]

X_train, y_train = X[train_mask], y[train_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nTrain: {len(X_train)} rows ({data.loc[train_mask, 'dia'].min().date()} → {data.loc[train_mask, 'dia'].max().date()})")
print(f"Test:  {len(X_test)} rows  ({data.loc[test_mask,  'dia'].min().date()} → {data.loc[test_mask,  'dia'].max().date()})")

# ─────────────────────────────────────────────────────────────────────────────
# Sliding window CV
# ─────────────────────────────────────────────────────────────────────────────
def create_sliding_folds(dates, train_size_days=60, val_size_days=30, slide_days=30):
    folds    = []
    dates    = dates.reset_index(drop=True)
    min_date = dates.min()
    max_date = dates.max()
    fold_start = min_date

    while True:
        train_end = fold_start + pd.Timedelta(days=train_size_days)
        val_end   = train_end  + pd.Timedelta(days=val_size_days)
        if val_end > max_date:
            break
        train_idx = dates[(dates >= fold_start) & (dates < train_end)].index.tolist()
        val_idx   = dates[(dates >= train_end)  & (dates < val_end)].index.tolist()
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
        fold_start += pd.Timedelta(days=slide_days)
    return folds

dates_train = data.loc[X_train.index, 'dia'].reset_index(drop=True)
custom_cv   = create_sliding_folds(dates_train, train_size_days=60, val_size_days=30, slide_days=30)

print(f"\n=== FOLDS DE VALIDACIÓN CRUZADA ===")
for i, (ti, vi) in enumerate(custom_cv):
    td = dates_train.iloc[ti]
    vd = dates_train.iloc[vi]
    print(f"Fold {i}: Train [{td.min().date()} → {td.max().date()}] ({len(ti)} rows) "
          f"Val [{vd.min().date()} → {vd.max().date()}] ({len(vi)} rows)")
print(f"Total folds: {len(custom_cv)}")

# ─────────────────────────────────────────────────────────────────────────────
# Modelos y parámetros
# ─────────────────────────────────────────────────────────────────────────────
pipelines = {
    'Ridge':      Pipeline([('scaler', StandardScaler()), ('model', Ridge())]),
    'Lasso':      Pipeline([('scaler', StandardScaler()), ('model', Lasso(max_iter=100000))]),
    'ElasticNet': Pipeline([('scaler', StandardScaler()), ('model', ElasticNet(max_iter=100000))]),
    'RandomForest':    Pipeline([('model', RandomForestRegressor(n_jobs=-1, random_state=42))]),
    'GradientBoosting': Pipeline([('model', GradientBoostingRegressor(random_state=42))]),
}

param_grids = {
    'Ridge':      {'model__alpha': [0.01, 0.1, 1, 10, 100, 1000]},
    'Lasso':      {'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]},
    'ElasticNet': {
        'model__alpha':    [0.001, 0.01, 0.1, 1, 10],
        'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'RandomForest': {
        'model__n_estimators':     [100, 300],
        'model__max_depth':        [None, 10, 20],
        'model__min_samples_leaf': [1, 5, 10],
    },
    'GradientBoosting': {
        'model__n_estimators':  [100, 300],
        'model__max_depth':     [3, 5, 7],
        'model__learning_rate': [0.01, 0.1, 0.2],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
results = {}
for name, pipeline in pipelines.items():
    print(f"\nFitting {name}...")
    grid = GridSearchCV(pipeline, param_grids[name], cv=custom_cv,
                        scoring='r2', n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    y_pred   = grid.predict(X_test)
    test_r2  = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    results[name] = {
        'best_params': grid.best_params_,
        'cv_r2':       grid.best_score_,
        'test_r2':     test_r2,
        'test_mae':    test_mae,
        'model':       grid.best_estimator_,
    }

    print(f"  Best params : {grid.best_params_}")
    print(f"  CV R²       : {grid.best_score_:.4f}")
    print(f"  Test R²     : {test_r2:.4f}")
    print(f"  Test MAE    : {test_mae:.4f} Q/MWh")

# ─────────────────────────────────────────────────────────────────────────────
# Resumen
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"{'Model':<20} {'CV R²':>8} {'Test R²':>10} {'Test MAE':>10}")
print("-" * 55)
for name, r in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
    print(f"{name:<20} {r['cv_r2']:>8.4f} {r['test_r2']:>10.4f} {r['test_mae']:>10.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Feature importance for best model
# ─────────────────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['test_r2'])
best_model = results[best_name]['model']
print(f"\n=== FEATURE IMPORTANCE: {best_name} ===")

if hasattr(best_model.named_steps['model'], 'coef_'):
    coefs = pd.DataFrame({
        'feature':     FEATURES,
        'coefficient': best_model.named_steps['model'].coef_
    }).sort_values('coefficient', key=abs, ascending=False)
    print(coefs[coefs['coefficient'] != 0].head(20).to_string(index=False))

elif hasattr(best_model.named_steps['model'], 'feature_importances_'):
    importances = pd.DataFrame({
        'feature':    FEATURES,
        'importance': best_model.named_steps['model'].feature_importances_
    }).sort_values('importance', ascending=False)
    print(importances.head(20).to_string(index=False))
