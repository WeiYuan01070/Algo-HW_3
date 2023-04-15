import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from pyRMT import clipped, optimalShrinkage

class TAQ_Cov_Clean:
    def __init__(self, df):
        self.df = df

    def calculate_mid_quote_returns(self):
        resample_df = self.df.resample('5T')['midQuote'].agg(['first', 'last'])
        resample_df['return'] = resample_df['last'] / resample_df['first'] - 1
        return resample_df['return']

    def calculate_cov(self):
        returns = self.calculate_mid_quote_returns()
        empirical_cov = EmpiricalCovariance().fit(returns)
        empirical_cov_matrix = empirical_cov.covariance_
        return empirical_cov_matrix

    def clean_cov(self):
        returns = self.calculate_mid_quote_returns()
        clipped_cov = clipped().fit(returns)
        clipped_cov_matrix = clipped_cov.covariance_
        return clipped_cov_matrix

    def shrinkage_cov(self):
        returns = self.calculate_mid_quote_returns()
        shrinkage_cov = optimalShrinkage().fit(returns)
        optimal_shrinkage_cov_matrix = shrinkage_cov.covariance_
        return optimal_shrinkage_cov_matrix

# Define the predictors g
def min_variance_portfolio(N):
    return np.ones(N) / N

def omniscient_portfolio(realized_returns):
    return realized_returns

def random_long_short_predictors(N):
    random_vector = np.random.randn(N)
    return random_vector / np.linalg.norm(random_vector)

# Calculate the cleaned covariance matrix using the shrinkage function and cleaning scheme
# ...

# Compute the optimal portfolio weights using Equation (14)
def optimal_portfolio_weights(cleaned_cov_matrix, g):
    inv_cleaned_cov_matrix = np.linalg.inv(cleaned_cov_matrix)
    weights = inv_cleaned_cov_matrix @ g / (g.T @ inv_cleaned_cov_matrix @ g)
    return weights

# Normalize predictors
normalized_predictors = {
    'minimum_variance': min_variance_portfolio(N),
    'omniscient': omniscient_portfolio(realized_returns),
    'random_long_short': random_long_short_predictors(N)
}

# Calculate the optimal portfolio weights for each predictor
optimal_weights = {}
for predictor_name, g in normalized_predictors.items():
    optimal_weights[predictor_name] = optimal_portfolio_weights(cleaned_cov_matrix, g)

# Compute the realized risk of the optimal portfolios
# ...




