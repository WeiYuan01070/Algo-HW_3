import pandas as pd
import numpy as np
import os
from sklearn.covariance import EmpiricalCovariance
from pyRMT import clipped, optimalShrinkage
from tqdm import tqdm
from CollectTicker import collect_ticker
from TAQQuotesReader import TAQQuotesReader

data_path = 'C:\\Users\\Admin\\PycharmProjects\\Algo-HW_3\\data\\quotes'


def calculate_mid_quote_returns(df):
    resample_df = df.resample('5T')['midQuote'].agg(['first', 'last'])
    resample_df['return'] = resample_df['last'] / resample_df['first'] - 1
    return resample_df['return']


def clean_cov(data):
    clipped_cov = clipped().fit(data)
    clipped_cov_matrix = clipped_cov.covariance_
    return clipped_cov_matrix


def calculate_cov(data):
    empirical_cov = EmpiricalCovariance().fit(data)
    empirical_cov_matrix = empirical_cov.covariance_
    return empirical_cov_matrix


def shrinkage_cov(data):
    shrinkage_cov = optimalShrinkage().fit(data)
    optimal_shrinkage_cov_matrix = shrinkage_cov.covariance_
    return optimal_shrinkage_cov_matrix


class PortfolioOptimizer:
    def __init__(self, df, window_size=1500, train_size=1000, covariance_method='standard', g_method='default'):
        self.df = df
        self.window_size = window_size
        self.train_size = train_size
        self.covariance_method = covariance_method
        self.g_method = g_method

    def calculate_covariance(self, data):

        if self.covariance_method == 'empirical':
            return calculate_cov(data)
        elif self.covariance_method == 'clipped':
            # Implement the covariance calculation for method_3
            return clean_cov(data)
        elif self.covariance_method == 'shrinkage':
            return shrinkage_cov(data)
        else:
            raise ValueError("Invalid covariance_method. Choose from 'standard', 'method_2', or 'method_3'.")

    def calculate_g(self, data, cov):
        if self.g_method == 'mvp':
            return np.ones(data.shape[1])
        elif self.g_method == 'omniscient':
            sigma = np.sqrt(np.diag(cov))
            g = data.iloc[-1, :] / sigma * np.sqrt(data.shape[1])
            return g
        elif self.g_method == 'lsp':
            N = data.shape[1]
            random_vector = np.random.randn(N)
            unit_vector = random_vector / np.linalg.norm(random_vector)
            g = np.sqrt(N) * unit_vector
            return g
        else:
            raise ValueError("Invalid g_method. Choose from 'default', 'method_2', or 'method_3'.")

    def calculate_portfolio_weights(self, train_df):
        cov_matrix = calculate_cov(train_df)
        g = self.calculate_g(train_df, cov_matrix)
        weights = (cov_matrix @ g) / (g.T @ cov_matrix @ g)
        return weights

    def calculate_portfolio_vol(self, out_of_sample_df, weights):
        cov_matrix = self.calculate_covariance(out_of_sample_df)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return np.sqrt(portfolio_variance * 252 * 78)

    def rolling_portfolio_optimization(self):
        weights_list = []
        portfolio_vol_list = []

        for start_idx in range(0, len(self.df) - self.window_size + 1):
            sub_df = self.df[start_idx:start_idx + self.window_size]
            train_df = sub_df[:self.train_size]
            out_of_sample_df = sub_df[self.train_size:]

            weights = self.calculate_portfolio_weights(train_df)
            portfolio_vol = self.calculate_portfolio_vol(out_of_sample_df, weights)

            weights_list.append(weights)
            portfolio_vol_list.append(portfolio_vol)

        return weights_list, portfolio_vol_list


if __name__ == "__main__":

    # ticker_list = collect_ticker(path='C:\\Users\\Admin\\PycharmProjects\\Algo-HW_3\\data\\s&p500.xlsx')
    #
    #
    # df_list = {ticker: pd.DataFrame() for ticker in ticker_list}
    # for root, dir, file in os.walk(data_path):
    #     for date in tqdm(dir):
    #         for subroot, subdir, subfiles in os.walk(os.path.join(root, date)):
    #             for f in tqdm(subfiles):
    #                 ticker = f.split('_quotes')[0]
    #                 if ticker not in ticker_list:
    #                     continue
    #                 q_reader = TAQQuotesReader(os.path.join(subroot, f))
    #                 q_df = q_reader.get_df(date, ticker)
    #                 q_df['midQuote'] = (q_df['AskPrice'] + q_df['BidPrice']) / 2
    #                 q_df.set_index('Date', inplace=True)
    #                 q_df = calculate_mid_quote_returns(q_df)
    #                 df_list[ticker] = pd.concat([df_list[ticker], q_df], axis=0)

    # df = pd.concat(df_list, axis=1)
    #
    # df.to_pickle('my_dataframe.pkl')

    df = pd.read_pickle('my_dataframe.pkl')
    df = df.fillna(0)
    for cov_method in tqdm(['empirical', 'clipped', 'shrinkage']):
        for g_method in tqdm(['mvp', 'omniscient', 'lsp']):
            optim = PortfolioOptimizer(df, g_method=g_method, covariance_method=cov_method)
            weights_list, portfolio_vol_list = optim.rolling_portfolio_optimization()
            print(f'{cov_method} + {g_method}: {np.mean(portfolio_vol_list)}')


