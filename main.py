"""Train or predict with the final 45-model ensemble."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from common import build_model

BASE = Path(__file__).resolve().parent

def cat_features(data, recipe):
    x = data[recipe['input_columns']].copy()
    for c in recipe['categorical_features']:
        x[c] = x[c].fillna('missing').astype(str)
    return x

def main():
    recipe = json.loads((BASE / 'recipe.json').read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'predict'])
    parser.add_argument('--train', type=Path, default=BASE / 'hard_train.csv')
    parser.add_argument('--test', type=Path, default=BASE / 'hard_test.csv')
    parser.add_argument('--variant', choices=list(recipe['variants']) + ['all'], default='all')
    parser.add_argument('--model-dir', type=Path, default=BASE / 'weights')
    parser.add_argument('--output-dir', type=Path, default=BASE / 'submissions')
    args = parser.parse_args()
    base = recipe['base']
    cat = recipe['catboost']
    if args.mode == 'train':
        data = pd.read_csv(args.train)
        y = data.protection_score.to_numpy(float)
        assert np.isfinite(y).all() and ((y >= 0) & (y <= 100)).all()
        np.random.seed(next(iter(recipe['variants'].values()))['seed'])
        args.model_dir.mkdir(exist_ok=True, parents=True)
        for member in base['members']:
            model = build_model(base, member)
            model.fit(data[base['input_columns']], y)
            joblib.dump(model, args.model_dir / member['file'], compress=3)
        x = cat_features(data, cat)
        for number, spec in enumerate(cat['models'], 1):
            parameters = dict(spec['parameters'])
            parameters.update(loss_function='RMSE', eval_metric='RMSE', nan_mode='Min',
                              random_seed=spec['seed'], thread_count=4,
                              cat_features=cat['categorical_features'],
                              verbose=False, allow_writing_files=False)
            model = CatBoostRegressor(**parameters)
            model.fit(x, y)
            model.save_model(str(args.model_dir / spec['file']))
            print(f'CatBoost {number}/15', flush=True)
        return
    data = pd.read_csv(args.test)
    required = ['customer_id'] + base['input_columns']
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f'Missing input columns: {missing}')
    if len(data) == 0 or not data.customer_id.is_unique or data.customer_id.isna().any():
        raise ValueError('Test data must contain nonempty, unique customer_id values')
    base_pred = np.zeros(len(data))
    for member in base['members']:
        model = joblib.load(args.model_dir / member['file'])
        raw = model.predict(data[base['input_columns']])
        assert np.isfinite(raw).all()
        base_pred += member['weight'] * np.clip(raw, 0, 100)
    x = cat_features(data, cat)
    cat_pred = np.zeros(len(data))
    for spec in cat['models']:
        model = CatBoostRegressor()
        model.load_model(str(args.model_dir / spec['file']))
        cat_pred += spec['weight'] * model.predict(x)
    assert np.isfinite(cat_pred).all()
    cat_pred = np.clip(cat_pred, 0, 100)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    variants = recipe['variants'] if args.variant == 'all' else {args.variant: recipe['variants'][args.variant]}
    for name, variant in variants.items():
        w = variant['catboost_weight']
        prediction = np.clip((1-w)*base_pred + w*cat_pred, 0, 100)
        out = args.output_dir / f"submission_seed_{variant['seed']}.csv"
        pd.DataFrame({'customer_id': data.customer_id, 'protection_score': prediction}).to_csv(out, index=False, float_format='%.10f')
        print(name, out.resolve(), flush=True)

if __name__ == '__main__':
    main()
