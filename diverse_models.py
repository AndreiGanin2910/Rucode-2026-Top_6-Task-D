"""Controlled diversity over rows or tree size, with fixed random seeds."""
import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator, clone

class DiverseBag(RegressorMixin, BaseEstimator):
    def __init__(self, estimator, seed, members=15, style='rows85'):
        self.estimator=estimator
        self.seed=seed
        self.members=members
        self.style=style

    def fit(self, x, y):
        seeds=np.random.default_rng(self.seed).integers(1,2**31-1,size=self.members)
        y=np.asarray(y)
        self.models_=[]
        for number,seed in enumerate(seeds):
            model=clone(self.estimator).set_params(random_state=int(seed))
            rng=np.random.default_rng(int(seed))
            rows=np.arange(len(y))
            if self.style=='rows85':
                rows=rng.choice(len(y),size=int(.85*len(y)),replace=False)
            elif self.style=='bootstrap':
                rows=rng.integers(0,len(y),size=len(y))
            elif self.style=='leafmix':
                leaves=[7,11,15,23,31][number % 5]
                parameter='num_leaves' if 'num_leaves' in model.get_params() else 'max_leaf_nodes'
                model.set_params(**{parameter:leaves})
            else:
                raise ValueError(self.style)
            model.fit(x[rows],y[rows])
            self.models_.append(model)
        return self

    def predict(self, x):
        return np.mean([np.clip(model.predict(x),0,100) for model in self.models_],axis=0)
