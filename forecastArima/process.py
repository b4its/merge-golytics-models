import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import normaltest
import itertools
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle

warnings.filterwarnings("ignore")

store = pd.read_csv("rossmann-store-sales/store.csv")
train = pd.read_csv("rossmann-store-sales/train.csv", parse_dates = True, low_memory = False, index_col = 'Date')

train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.isocalendar().week
train['SalePerCustomer'] = train['Sales']/train['Customers']
train.fillna(0, inplace = True)

os.makedirs("output/plot", exist_ok=True)
os.makedirs("output/forecast", exist_ok=True)

sns.set_theme(style = "ticks")
c = '#386B7F'
plt.figure(figsize = (12, 6))

# First ECDF - Sales distribution
plt.subplot(311)
ecdf_sales = stats.ecdf(train['Sales'])
x_sales = np.sort(train['Sales'])
y_sales = np.arange(1, len(x_sales) + 1) / len(x_sales)
plt.plot(x_sales, y_sales, label = "scipy", color = c);
plt.xlabel('Sales'); plt.ylabel('ECDF');
plt.title('Empirical Cumulative Distribution Function - Sales')

# Second ECDF - Customers distribution  
plt.subplot(312)
ecdf_customers = stats.ecdf(train['Customers'])
x_customers = np.sort(train['Customers'])
y_customers = np.arange(1, len(x_customers) + 1) / len(x_customers)
plt.plot(x_customers, y_customers, label = "scipy", color = c);
plt.xlabel('Customers');
plt.title('Empirical Cumulative Distribution Function - Customers')

# Third ECDF - Sales per Customer distribution
plt.subplot(313)
ecdf_sale_per_customer = stats.ecdf(train['SalePerCustomer'])
x_sale_per_customer = np.sort(train['SalePerCustomer'])
y_sale_per_customer = np.arange(1, len(x_sale_per_customer) + 1) / len(x_sale_per_customer)
plt.plot(x_sale_per_customer, y_sale_per_customer, label = "scipy", color = c);
plt.xlabel('Sale per Customer');
plt.title('Empirical Cumulative Distribution Function - Sales per Customer')

plt.tight_layout()

plt.savefig("output/plot/ecdf_analysis.png", dpi=300, bbox_inches='tight')
plt.close()

print("Plot saved as 'output/plot/ecdf_analysis.png'")

train[(train.Open == 0) & (train.Sales == 0)]
zero_sales = train[(train.Open != 0) & (train.Sales == 0)]
print("In total: ", zero_sales.shape)

train = train[(train["Open"] != 0) & (train['Sales'] != 0)]
train=train.drop(columns=train[(train.Open == 1) & (train.Sales == 0)].index)
{"Mean":np.mean(train.Sales),"Median":np.median(train.Sales)}
{"Mean":np.mean(train.Customers),"Median":np.median(train.Customers)}

store.fillna(0, inplace = True)

train_store = pd.merge(train, store, how = 'inner', on = 'Store')

print("In total: ", train_store.shape)
train_store.head(100)

train_store.groupby('StoreType')[['Customers', 'Sales']].sum()

sns.set_theme(style = "ticks")
c = '#386B7F'
sns.catplot(data = train_store, x = 'Month', y = "Sales",
               col = 'StoreType',
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo',
               kind='point')
plt.title('Sales vs Month by Store Type and Promo')
plt.savefig('output/plot/sales_month_storetype_promo_catplot.png')
sns.catplot(data = train_store, x = 'Month', y = "Customers",
               col = 'StoreType',
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo',
               kind='point')
plt.savefig('output/plot/customers_month_storetype_promo_catplot.png')
sns.catplot(data = train_store, x = 'Month', y = "Sales",
               col = 'StoreType',
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo',
               kind='point')
plt.savefig('output/plot/sales_month_storetype_promo_catplot.png')
sns.catplot(data = train_store, x = 'Month', y = "SalePerCustomer",
               col = 'StoreType',
               palette = 'plasma',
               hue = 'StoreType',
               row = 'Promo',
               kind='point')
plt.savefig('output/plot/salepercustomer_month_storetype_promo_catplot.png')
sns.catplot(data = train_store, x = 'DayOfWeek', y = "Sales",
               col = 'DayOfWeek',
               palette = 'plasma',
               hue = 'StoreType',
               row = 'StoreType',
               kind = 'point')
plt.savefig('output/plot/sales_dayofweek_storetype_catplot.png')

train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].unique()

train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
        (train_store.Month - train_store.CompetitionOpenSinceMonth)
    
train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
        (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

train_store.fillna(0, inplace = True)

train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()

numeric_cols = train_store.select_dtypes(include=np.number).columns
corr_all = train_store[numeric_cols].drop('Open', axis = 1).corr()

mask = np.zeros_like(corr_all, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize = (11, 9))

sns.heatmap(corr_all, mask = mask, annot = True, square = True, linewidths = 0.5, ax = ax, cmap = "BrBG", fmt='.2f')
plt.savefig('output/plot/correlation_heatmap.png')

sns.catplot(data = train_store, x = 'DayOfWeek', y = "Sales",
               col = 'Promo',
               row = 'Promo2',
               hue = 'Promo2',
               kind = 'point')
plt.savefig('output/plot/sales_dayofweek_promo_promo2_catplot.png')

train['Sales'] = train['Sales'] * 1.0

sales_a = train[train.Store == 2]['Sales']
sales_b = train[train.Store == 85]['Sales']
sales_c = train[train.Store == 1]['Sales']
sales_d = train[train.Store == 13]['Sales']

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

sales_a.resample('W').sum().plot(color = c, ax = ax1)
ax1.set_title('Weekly Sales Trend - Store A')
sales_b.resample('W').sum().plot(color = c, ax = ax2)
ax2.set_title('Weekly Sales Trend - Store B')
sales_c.resample('W').sum().plot(color = c, ax = ax3)
ax3.set_title('Weekly Sales Trend - Store C')
sales_d.resample('W').sum().plot(color = c, ax = ax4)
ax4.set_title('Weekly Sales Trend - Store D')

plt.tight_layout()
plt.savefig('output/plot/weekly_sales_trends.png')

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize = (12, 13))

# weekly
decomposition_a = seasonal_decompose(sales_a, model = 'additive', period = 7)
decomposition_a.trend.plot(color = c, ax = ax1)
ax1.set_title('Weekly Trend from Decomposition - Store A')

decomposition_b = seasonal_decompose(sales_b, model = 'additive', period = 7)
decomposition_b.trend.plot(color = c, ax = ax2)
ax2.set_title('Weekly Trend from Decomposition - Store B')

decomposition_c = seasonal_decompose(sales_c, model = 'additive', period = 7)
decomposition_c.trend.plot(color = c, ax = ax3)
ax3.set_title('Weekly Trend from Decomposition - Store C')

decomposition_d = seasonal_decompose(sales_d, model = 'additive', period = 7)
decomposition_d.trend.plot(color = c, ax = ax4)
ax4.set_title('Weekly Trend from Decomposition - Store D')

plt.tight_layout()
plt.savefig('output/plot/weekly_decomposition_trends.png')

def test_stationarity(timeseries, window = 12, cutoff = 0.01):

    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig(f'output/plot/rolling_mean_std.png')
    plt.close()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)
    
    print(dfoutput)

def residual_plot(model, model_name="Model"):

    resid = model.resid
    print(normaltest(resid))


    fig = plt.figure(figsize=(12,8))
    ax0 = fig.add_subplot(111)

    sns.histplot(resid, kde=True, stat='density', ax=ax0)
    x = np.linspace(resid.min(), resid.max(), 100)
    y = stats.norm.pdf(x, resid.mean(), resid.std())
    ax0.plot(x, y, 'r-', lw=2, label='Normal distribution')

    (mu, sigma) = stats.norm.fit(resid)

    plt.legend(loc='best')
    plt.ylabel('Frequency')
    plt.title(f'Residual distribution - {model_name}')
    plt.savefig(f'output/plot/residual_distribution_{model_name.replace(" ", "_")}.png')
    plt.close()


    # ACF and PACF
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))
    
    # Plot ACF
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    plot_acf(model.resid, lags=40, ax=ax1)
    ax1.set_title(f'ACF of Residuals - {model_name}')
    
    # Plot PACF
    plot_pacf(model.resid, lags=40, ax=ax2)
    ax2.set_title(f'PACF of Residuals - {model_name}')

    plt.tight_layout()
    plt.savefig(f'output/plot/residual_acf_pacf_{model_name.replace(" ", "_")}.png')
    plt.close()

def test_stationarity(timeseries, window = 12, cutoff = 0.01, series_name="Time Series"):

    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(f'Rolling Mean & Standard Deviation - {series_name}')
    plt.savefig(f'output/plot/rolling_stats_{series_name.replace(" ", "_")}.png')
    plt.close()


    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC', maxlag = 20 )
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    pvalue = dftest[1]
    if pvalue < cutoff:
        print('p-value = %.4f. The series is likely stationary.' % pvalue)
    else:
        print('p-value = %.4f. The series is likely non-stationary.' % pvalue)

    print(dfoutput)

first_diff_a = sales_a - sales_a.shift(1)
first_diff_a = first_diff_a.dropna(inplace = False)
test_stationarity(first_diff_a, window = 12, series_name="Store A")
test_stationarity(sales_b, series_name="Store B")

first_diff_b = sales_b - sales_b.shift(1)
first_diff_b = first_diff_b.dropna(inplace = False)
test_stationarity(first_diff_b, window = 12, series_name="First Difference Store B")

test_stationarity(sales_c, series_name="Store C")
first_diff_c = sales_c - sales_c.shift(1)
first_diff_c = first_diff_c.dropna(inplace = False)
test_stationarity(first_diff_c, window = 12, series_name="First Difference Store C")

test_stationarity(sales_d, series_name="Store D")
first_diff_d = sales_d - sales_d.shift(1)
first_diff_d = first_diff_d.dropna(inplace = False)
test_stationarity(first_diff_d, window = 12, series_name="First Difference Store D")

# figure for sales
fig, axes = plt.subplots(4, 2, figsize=(12, 8))

# acf and pacf for A
plot_acf(sales_a, lags=50, ax=axes[0,0], color=c, title='ACF sales_a')
plot_pacf(sales_a, lags=50, ax=axes[0,1], color=c, title='PACF sales_a')

# acf and pacf for B
plot_acf(sales_b, lags=50, ax=axes[1,0], color=c, title='ACF sales_b')
plot_pacf(sales_b, lags=50, ax=axes[1,1], color=c, title='PACF sales_b')

# acf and pacf for C
plot_acf(sales_c, lags=50, ax=axes[2,0], color=c, title='ACF sales_c')
plot_pacf(sales_c, lags=50, ax=axes[2,1], color=c, title='PACF sales_c')

# acf and pacf for D
plot_acf(sales_d, lags=50, ax=axes[3,0], color=c, title='ACF sales_d')
plot_pacf(sales_d, lags=50, ax=axes[3,1], color=c, title='PACF sales_d')

plt.tight_layout()
plt.savefig('output/plot/acf_pacf_sales.png', dpi=300, bbox_inches='tight')
plt.close()

# figure for first difference
fig, axes = plt.subplots(4, 2, figsize=(12, 8))

# acf and pacf for A
plot_acf(first_diff_a, lags=50, ax=axes[0,0], color=c, title='ACF First Diff sales_a')
plot_pacf(first_diff_a, lags=50, ax=axes[0,1], color=c, title='PACF First Diff sales_a')

# acf and pacf for B
plot_acf(first_diff_b, lags=50, ax=axes[1,0], color=c, title='ACF First Diff sales_b')
plot_pacf(first_diff_b, lags=50, ax=axes[1,1], color=c, title='PACF First Diff sales_b')

# acf and pacf for C
plot_acf(first_diff_c, lags=50, ax=axes[2,0], color=c, title='ACF First Diff sales_c')
plot_pacf(first_diff_c, lags=50, ax=axes[2,1], color=c, title='PACF First Diff sales_c')

# acf and pacf for D
plot_acf(first_diff_d, lags=50, ax=axes[3,0], color=c, title='ACF First Diff sales_d')
plot_pacf(first_diff_d, lags=50, ax=axes[3,1], color=c, title='PACF First Diff sales_d')

plt.tight_layout()
plt.savefig('output/plot/acf_pacf_first_diff_sales.png', dpi=300, bbox_inches='tight')
plt.close()

arima_mod_a = ARIMA(sales_a, order=(11,1,0), exog=None).fit()
try:
    sarima_mod_a = SARIMAX(sales_a, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store A SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_a = SARIMAX(sales_a, trend='n', order=(11,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store A...")
        sarima_mod_a = ARIMA(sales_a, order=(11,1,0)).fit()
residual_plot(arima_mod_a, model_name="ARIMA(11,1,0) for Store A")
residual_plot(sarima_mod_a, model_name="SARIMA(11,1,0) for Store A")


os.makedirs("output/model_summary", exist_ok=True)

with open('output/model_summary/sarima_model_a_summary.txt', 'w') as f:
    f.write(str(sarima_mod_a.summary()))

sales_a_reindex = sales_a.reindex(index=sales_a.index[::-1])
mydata_a = sales_a_reindex
temp_df =pd.DataFrame(mydata_a)
mydata_a = temp_df
try:
    sarima_mod_a_train = SARIMAX(mydata_a, trend='n', order=(11,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store A SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_a_train = SARIMAX(mydata_a, trend='n', order=(11,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store A...")
        sarima_mod_a_train = ARIMA(mydata_a, order=(11,1,0)).fit()

with open('output/model_summary/sarima_model_a_train_summary.txt', 'w') as f:
    f.write(str(sarima_mod_a_train.summary()))

residual_plot(sarima_mod_a_train, model_name="SARIMA Store A (mydata_a)")

plt.figure(figsize=(50,10))
plt.plot(mydata_a, c='red', label='Actual')
plt.plot(sarima_mod_a_train.fittedvalues, c='blue', label='Fitted')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Fitted Sales - SARIMA Store A (mydata_a)")
plt.legend()
plt.savefig('output/plot/sarima_store_a_fitted_vs_actual.png')
plt.close()

plt.figure(figsize=(30,10))
plt.plot(mydata_a.iloc[1:625], label='Actual (Training)', color='red')
forecast = sarima_mod_a_train.predict(start = 625, end = 783, dynamic= False)
plt.plot(forecast, color='blue', label='Forecast')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Forecasted Sales - SARIMA Store A")
plt.legend()
plt.savefig('output/plot/sarima_store_a_forecast.png')
plt.close()

forecast.to_csv('output/forecast/sarima_store_a_forecast.csv')

arima_mod_b = ARIMA(sales_b, order=(1,1,0), exog=None).fit()
residual_plot(arima_mod_b, model_name="ARIMA(1,1,0) for Store B")

try:
    sarima_mod_b = SARIMAX(sales_b, trend='n', order=(1,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store B SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_b = SARIMAX(sales_b, trend='n', order=(1,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store B...")
        sarima_mod_b = ARIMA(sales_b, order=(1,1,0)).fit()
residual_plot(sarima_mod_b, model_name="SARIMA(1,1,0) for Store B")

with open('output/model_summary/sarima_model_b_summary.txt', 'w') as f:
    f.write(str(sarima_mod_b.summary()))

sales_b_reindex = sales_b.reindex(index=sales_b.index[::-1])
mydata_b = sales_b_reindex
temp_df =pd.DataFrame(mydata_b)
mydata_b = temp_df
try:
    sarima_mod_b_train = SARIMAX(mydata_b, trend='n', order=(1,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store B SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_b_train = SARIMAX(mydata_b, trend='n', order=(1,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store B...")
        sarima_mod_b_train = ARIMA(mydata_b, order=(1,1,0)).fit()


with open('output/model_summary/sarima_model_b_train_summary.txt', 'w') as f:
    f.write(str(sarima_mod_b_train.summary()))

residual_plot(sarima_mod_b_train, model_name="SARIMA Store B (mydata_b)")

plt.figure(figsize=(50,10))
plt.plot(mydata_b, color='red', label='Actual')
plt.plot(sarima_mod_b_train.fittedvalues, color='blue', label='Fitted')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Fitted Sales - SARIMA Store B (mydata_b)")
plt.legend()
plt.savefig('output/plot/sarima_store_b_fitted_vs_actual.png')
plt.close()

plt.figure(figsize=(30,10))
plt.plot(mydata_b.iloc[1:755], label='Actual (Training)', color='red')
forecast = sarima_mod_b_train.predict(start = 755, end = 941, dynamic= False)
plt.plot(forecast, color='blue', label='Forecast')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Forecasted Sales - SARIMA Store B")
plt.legend()
plt.savefig('output/plot/sarima_store_b_forecast.png')
plt.close()

forecast.to_csv('output/forecast/sarima_store_b_forecast.csv')

arima_mod_c = ARIMA(sales_c, order=(1,1,0), exog=None).fit()
residual_plot(arima_mod_c, model_name="ARIMA(1,1,0) for Store C")

with open('output/model_summary/arima_model_c_summary.txt', 'w') as f:
    f.write(str(arima_mod_c.summary()))

try:
    sarima_mod_c = SARIMAX(sales_c, trend='n', order=(1,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store C SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_c = SARIMAX(sales_c, trend='n', order=(1,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store C...")
        sarima_mod_c = ARIMA(sales_c, order=(1,1,0)).fit()
residual_plot(sarima_mod_c, model_name="SARIMA(1,1,0) for Store C")

sales_c_reindex = sales_c.reindex(index=sales_c.index[::-1])
mydata_c = sales_c_reindex
temp_df =pd.DataFrame(mydata_c)
mydata_c = temp_df
try:
    sarima_mod_c_train = SARIMAX(mydata_c, trend='n', order=(1,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store C SARIMA model. Trying with simpler parameters...")
    try:
        # Try with simpler seasonal order
        sarima_mod_c_train = SARIMAX(mydata_c, trend='n', order=(1,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store C...")
        sarima_mod_c_train = ARIMA(mydata_c, order=(1,1,0)).fit()

with open('output/model_summary/sarima_model_c_train_summary.txt', 'w') as f:
    f.write(str(sarima_mod_c_train.summary()))

residual_plot(sarima_mod_c_train, model_name="SARIMA Store C (mydata_c)")

plt.figure(figsize=(50,10))
plt.plot(mydata_c, c='red', label='Actual')
plt.plot(sarima_mod_c_train.fittedvalues, c='blue', label='Fitted')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Fitted Sales - SARIMA Store C (mydata_c)")
plt.legend()
plt.savefig('output/plot/sarima_store_c_fitted_vs_actual.png')
plt.close()

plt.figure(figsize=(30,10))
plt.plot(mydata_c.iloc[1:625], label='Actual (Training)', color='red')
forecast = sarima_mod_c_train.predict(start = 625, end = 783, dynamic= False)
plt.plot(forecast, color='blue', label='Forecast')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Forecasted Sales - SARIMA Store C")
plt.legend()
plt.savefig('output/plot/sarima_store_c_forecast.png')
plt.close()

forecast.to_csv('output/forecast/sarima_store_c_forecast.csv')

arima_mod_d = ARIMA(sales_d, order=(1,1,0), exog=None).fit()
residual_plot(arima_mod_d, model_name="ARIMA(1,1,0) for Store D")

with open('output/model_summary/arima_model_d_summary.txt', 'w') as f:
    f.write(str(arima_mod_d.summary()))

try:
    sarima_mod_d = SARIMAX(sales_d, trend='n', order=(1,1,0), seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store D SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_d = SARIMAX(sales_d, trend='n', order=(1,1,0), seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store D...")
        sarima_mod_d = ARIMA(sales_d, order=(1,1,0)).fit()
residual_plot(sarima_mod_d, model_name="SARIMA(1,1,0) for Store D")

with open('output/model_summary/sarima_model_d_summary.txt', 'w') as f:
    f.write(str(sarima_mod_d.summary()))

sales_d_reindex = sales_d.reindex(index=sales_d.index[::-1])
mydata_d = sales_d_reindex
temp_df =pd.DataFrame(mydata_d)
mydata_d = temp_df

try:
    sarima_mod_d_train = SARIMAX(mydata_d, trend='n', order=(11,1,0),seasonal_order=(2,1,0,12)).fit()
except MemoryError:
    print("Memory error for Store D SARIMA model. Trying with simpler parameters...")
    try:
        sarima_mod_d_train = SARIMAX(mydata_d, trend='n', order=(11,1,0),seasonal_order=(1,1,0,12)).fit()
    except MemoryError:
        print("Still memory error. Using ARIMA model instead for Store D...")
        sarima_mod_d_train = ARIMA(mydata_d, order=(11,1,0)).fit()

with open('output/model_summary/sarima_model_d_train_summary.txt', 'w') as f:
    f.write(str(sarima_mod_d_train.summary()))

residual_plot(sarima_mod_d_train, model_name="SARIMA Store D (mydata_d)")

plt.figure(figsize=(50,10))
plt.plot(mydata_d, c='red', label='Actual')
plt.plot(sarima_mod_d_train.fittedvalues, c='blue', label='Fitted')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Fitted Sales - SARIMA Store D (mydata_d)")
plt.legend()
plt.savefig('output/plot/sarima_store_d_fitted_vs_actual.png')
plt.close()

plt.figure(figsize=(30,10))
plt.plot(mydata_d.iloc[1:495], label='Actual (Training)', color='red')
forecast = sarima_mod_d_train.predict(start = 495, end = 620, dynamic= False)
plt.plot(forecast, color='blue', label='Forecast')
plt.ylabel("Sales")
plt.xlabel("Time")
plt.title("Actual vs Forecasted Sales - SARIMA Store D")
plt.legend()
plt.savefig('output/plot/sarima_store_d_forecast.png')
plt.close()

forecast.to_csv('output/forecast/sarima_store_d_forecast.csv')

os.makedirs("models", exist_ok=True)

# Save the SARIMA models
if 'sarima_mod_a_train' in locals() and sarima_mod_a_train is not None:
    with open('models/sarima_model_a_train.pkl', 'wb') as f:
        pickle.dump(sarima_mod_a_train, f)
    print("SARIMA model for Store A saved as models/sarima_model_a_train.pkl")

if 'sarima_mod_b_train' in locals() and sarima_mod_b_train is not None:
    with open('models/sarima_model_b_train.pkl', 'wb') as f:
        pickle.dump(sarima_mod_b_train, f)
    print("SARIMA model for Store B saved as models/sarima_model_b_train.pkl")

if 'sarima_mod_c_train' in locals() and sarima_mod_c_train is not None:
    with open('models/sarima_model_c_train.pkl', 'wb') as f:
        pickle.dump(sarima_mod_c_train, f)
    print("SARIMA model for Store C saved as models/sarima_model_c_train.pkl")

if 'sarima_mod_d_train' in locals() and sarima_mod_d_train is not None:
    with open('models/sarima_model_d_train.pkl', 'wb') as f:
        pickle.dump(sarima_mod_d_train, f)
    print("SARIMA model for Store D saved as models/sarima_model_d_train.pkl")

# Save the ARIMA models
if 'arima_mod_a' in locals() and arima_mod_a is not None:
    with open('models/arima_model_a.pkl', 'wb') as f:
        pickle.dump(arima_mod_a, f)
    print("ARIMA model for Store A saved as models/arima_model_a.pkl")

if 'arima_mod_b' in locals() and arima_mod_b is not None:
    with open('models/arima_model_b.pkl', 'wb') as f:
        pickle.dump(arima_mod_b, f)
    print("ARIMA model for Store B saved as models/arima_model_b.pkl")

if 'arima_mod_c' in locals() and arima_mod_c is not None:
    with open('models/arima_model_c.pkl', 'wb') as f:
        pickle.dump(arima_mod_c, f)
    print("ARIMA model for Store C saved as models/arima_model_c.pkl")

if 'arima_mod_d' in locals() and arima_mod_d is not None:
    with open('models/arima_model_d.pkl', 'wb') as f:
        pickle.dump(arima_mod_d, f)
    print("ARIMA model for Store D saved as models/arima_model_d.pkl")

# Save the SARIMA models (non-train versions)
if 'sarima_mod_a' in locals() and sarima_mod_a is not None:
    with open('models/sarima_model_a.pkl', 'wb') as f:
        pickle.dump(sarima_mod_a, f)
    print("SARIMA model A saved as models/sarima_model_a.pkl")

if 'sarima_mod_b' in locals() and sarima_mod_b is not None:
    with open('models/sarima_model_b.pkl', 'wb') as f:
        pickle.dump(sarima_mod_b, f)
    print("SARIMA model B saved as models/sarima_model_b.pkl")

if 'sarima_mod_c' in locals() and sarima_mod_c is not None:
    with open('models/sarima_model_c.pkl', 'wb') as f:
        pickle.dump(sarima_mod_c, f)
    print("SARIMA model C saved as models/sarima_model_c.pkl")

if 'sarima_mod_d' in locals() and sarima_mod_d is not None:
    with open('models/sarima_model_d.pkl', 'wb') as f:
        pickle.dump(sarima_mod_d, f)
    print("SARIMA model D saved as models/sarima_model_d.pkl")

print("\nAll available models have been saved to models/ directory.")