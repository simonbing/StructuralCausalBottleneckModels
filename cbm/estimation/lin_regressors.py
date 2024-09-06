import numpy as np
from sklearn.linear_model import LinearRegression

from cbm.estimation.base_regressor import BaseRegressor


class LinRegressor(BaseRegressor):
    def __init__(self, d_micro_in, d_micro_out, d_bottleneck):
        super().__init__(d_micro_in, d_micro_out, d_bottleneck)

        self.model = LinearRegression()

    def fit(self, X, Y, X_cond=[]):
        if len(X_cond) == 0:  # no conditioning
            self.model.fit(X, Y)
        else:
            self.X_dim = X.shape[-1]  # need this later in get_bottleneck_fct

            X_cat = np.concatenate((X, X_cond), axis=1)
            # print(f'X_cat: {X_cat[9000, :]}')
            # print(f'Y: {Y[9000, :]}')
            self.model.fit(X_cat, Y)

    def get_bottleneck_fct(self):
        # using the first d_bottleneck entries...double check if this makes sense
        # TODO: this should be the d_bottleneck lin indep rows!
        try:
            # print(self.model.coef_)
            linear_map = self.model.coef_[:self.d_bottleneck, :self.X_dim].T
        except AttributeError:  # go here if self.X_dim is not defined, i.e. no conditioning
            linear_map = self.model.coef_[:self.d_bottleneck, :].T
        print(f'Linear map:\n{linear_map}')
        fct = lambda x: x @ linear_map

        return fct
