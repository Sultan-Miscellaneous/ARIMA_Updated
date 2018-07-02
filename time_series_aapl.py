import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from pyramid.arima import auto_arima
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error


def load_data(file):
    data = pd.read_csv(file, index_col=0)
    data.head()
    data.index = pd.to_datetime(data.index)
    # data.columns = ['Production']
    data.columns = ['open', 'high', 'low', 'close', 'volume', 'nan']
    data = data.drop('high', 1)
    data = data.drop('low', 1)
    data = data.drop('close', 1)
    data = data.drop('volume', 1)
    data = data.drop('nan', 1)
    data = data.loc['2007-01-01':]
    return data


def test_stationarity(data):
    print 'running Dickey-Fuller Test:'
    dftest = adfuller(data, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    print dfoutput


def apply_differencing(data, lag=1):
    df = data - data.shift(lag)
    df = df.iloc[lag:]
    return df


def main():
    print 'loading data'
    data = load_data('aapl.us.txt')
    data.plot()
    plt.title('data before differencing')
    plt.show()

    test_stationarity(data['open'])

    first_difference = apply_differencing(data)

    test_stationarity(first_difference['open'])

    first_difference.plot()
    plt.title('data after differencing')
    plt.show()
    autocorrelation_plot(first_difference)
    plt.title('autocorrelation of data after differencing')
    plt.show()

    random_array = pd.DataFrame(np.random.rand(1, len(first_difference.index))).transpose()
    autocorrelation_plot(random_array)
    plt.title('autocorrelation of random number array')
    plt.show()

    print 'auto correlation shows near random signal'

    # result = seasonal_decompose(data, model='multiplicative', freq = 6)

    # result.plot()
    # plt.show()

    # print 'residuals 5 number summary'
    #
    # print result.resid.describe()

    train = data.loc['2007-01-01':'2016-12-31']
    test = data.loc['2017-01-01':]

    stepwise_model = auto_arima(train, start_p=0, start_q=0,
                                max_p=50, max_q=50, max_d=50, m=365,
                                start_P=0, seasonal=False,
                                d=1, D=1, trace=True,
                                suppress_warnings=True,
                                stepwise=True)

    stepwise_model.fit(train)
    future_forecast = stepwise_model.predict(n_periods=len(test['open']))

    # print(future_forecast)
    future_forecast = pd.DataFrame(future_forecast, index=test.index, columns=['Prediction'])

    future_forecast.plot()
    plt.title('forecast for 2017')
    plt.show()

    test.plot()
    plt.title('actual data for 2017')
    plt.show()

    prediction_error = pd.DataFrame((future_forecast['Prediction'] - test['open'])**2, index=test.index)
    prediction_error.plot()
    plt.title('error')
    plt.show()


if __name__ == '__main__':
    main()
