import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
from datetime import datetime

data= input('Enter the stocks here ').upper()

for datum in data:
    if datum == ',':
        data = data.replace(datum, '')

stocks = data.split()

stock_data = pd.DataFrame()

start_date = '2015-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
for stock in stocks:
    stock_data[stock] = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)['Adj Close']

dailyReturns = stock_data.pct_change()

number_of_portfolios = 20
RF = 0

portfolio_returns = []
portfolio_risk = []
sharpe_ratio_port = []
portfolio_weights = []

for portfolio in range(number_of_portfolios):
    # generate a w random weight of length of number of stocks
    weights = np.random.random_sample(len(stock_data.columns))

    weights = weights / np.sum(weights)
    annualized_return = np.sum((dailyReturns.mean() * weights) * 252)
    portfolio_returns.append(annualized_return)
    # variance
    matrix_covariance_portfolio = (dailyReturns.cov()) * 252
    portfolio_variance = np.dot(weights.T, np.dot(matrix_covariance_portfolio, weights))
    portfolio_standard_deviation = np.sqrt(portfolio_variance)
    portfolio_risk.append(portfolio_standard_deviation)
    # sharpe_ratio
    sharpe_ratio = ((annualized_return - RF) / portfolio_standard_deviation)
    sharpe_ratio_port.append(sharpe_ratio)

    portfolio_weights.append(weights)

portfolio_risk = np.array(portfolio_risk)
portfolio_returns = np.array(portfolio_returns)
sharpe_ratio_port = np.array(sharpe_ratio_port)

plt.figure(figsize=(10, 5))
plt.scatter(portfolio_risk, portfolio_returns, c=portfolio_returns / portfolio_risk)
plt.xlabel('volatility')
plt.ylabel('returns')
plt.colorbar(label='Sharpe ratio')

porfolio_metrics = [portfolio_returns, portfolio_risk, sharpe_ratio_port, portfolio_weights]

portfolio_dfs = pd.DataFrame(porfolio_metrics)
portfolio_dfs = portfolio_dfs.T
portfolio_dfs.columns = ['Port Returns', 'Port Risk', 'Sharpe Ratio', 'Portfolio Weights']

# convert from object to float the first three columns.
for col in ['Port Returns', 'Port Risk', 'Sharpe Ratio']:
    portfolio_dfs[col] = portfolio_dfs[col].astype(float)

# portfolio with the highest Sharpe Ratio
Highest_sharpe_port = portfolio_dfs.iloc[portfolio_dfs['Sharpe Ratio'].idxmax()]
# portfolio with the minimum risk
min_risk = portfolio_dfs.iloc[portfolio_dfs['Port Risk'].idxmin()]

# Highest_sharpe_port
print('-----------------------------------------------------------------------')
print(f"{Highest_sharpe_port['Portfolio Weights']} in the order, {stocks}")
print(f"Your Portfolio Risk is: {round(Highest_sharpe_port['Port Risk'], 2)}%")
print(f"Your Sharpe Ratio is: {round(Highest_sharpe_port['Sharpe Ratio'], 2)}%")
print('-----------------------------------------------------------------------')
print(f"{min_risk['Portfolio Weights']} in the order, {stocks}")
print(f"Your Portfolio Risk is: {round(min_risk['Port Risk'], 2)}%")
print(f"Your Sharpe Ratio is: {round(min_risk['Sharpe Ratio'], 2)}%")
print('-----------------------------------------------------------------------')