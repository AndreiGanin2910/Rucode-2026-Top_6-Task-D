"""Only the two final bagged model families; preprocessing is fitted on train."""
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from diverse_models import DiverseBag

def build_model(recipe, member):
    categories = recipe['categorical_features']
    numeric = [c for c in recipe['input_columns'] if c not in categories]
    preprocessing = ColumnTransformer([
        ('numeric', SimpleImputer(strategy='median'), numeric),
        ('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categories),
    ])
    if member['factory_name'] == 'lgb_featurebag':
        estimator = LGBMRegressor(n_estimators=200, learning_rate=.05,
            num_leaves=15, min_child_samples=30, reg_lambda=3,
            colsample_bytree=.8, n_jobs=4, verbosity=-1,
            deterministic=True, force_col_wise=True)
    elif member['factory_name'] == 'hgb_featurebag':
        estimator = HistGradientBoostingRegressor(max_iter=250, learning_rate=.05,
            max_leaf_nodes=15, l2_regularization=3, early_stopping=False, max_features=.8)
    else:
        raise ValueError(member['factory_name'])
    ensemble = DiverseBag(estimator, member['seed'], member['bag_members'], member['diversity'])
    return Pipeline([('preprocessing', preprocessing), ('model', ensemble)])
