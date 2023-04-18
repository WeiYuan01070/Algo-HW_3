import unittest
import pandas as pd
import numpy as np

from TAQ_rolling import PortfolioOptimizer


class TestPortfolioOptimizer(unittest.TestCase):

    def setUp(self):
        data = np.random.rand(1600, 10)
        self.df = pd.DataFrame(data, columns=[f'Stock{i}' for i in range(10)])
        self.optimizer = PortfolioOptimizer(self.df, covariance_method='empirical', g_method='mvp')

    def test_calculate_covariance(self):
        data = self.df.iloc[:1000, :]
        cov_matrix = self.optimizer.calculate_covariance(data)
        self.assertEqual(cov_matrix.shape, (10, 10))

    def test_calculate_g(self):
        data = self.df.iloc[:1000, :]
        cov_matrix = self.optimizer.calculate_covariance(data)
        g = self.optimizer.calculate_g(data, cov_matrix)
        self.assertEqual(g.shape, (10,))

    def test_calculate_portfolio_weights(self):
        train_df = self.df.iloc[:1000, :]
        weights = self.optimizer.calculate_portfolio_weights(train_df)
        self.assertEqual(weights.shape, (10,))

    def test_calculate_portfolio_vol(self):
        out_of_sample_df = self.df.iloc[1000:1500, :]
        train_df = self.df.iloc[:1000, :]
        weights = self.optimizer.calculate_portfolio_weights(train_df)
        portfolio_vol = self.optimizer.calculate_portfolio_vol(out_of_sample_df, weights)
        self.assertIsInstance(portfolio_vol, float)

    def test_rolling_portfolio_optimization(self):
        weights_list, portfolio_vol_list = self.optimizer.rolling_portfolio_optimization()
        self.assertEqual(len(weights_list), 101)
        self.assertEqual(len(portfolio_vol_list), 101)

if __name__ == '__main__':
    unittest.main()
