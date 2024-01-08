import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import ARIMA
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from skforecast.Sarimax import Sarimax
from skforecast.ForecasterSarimax import ForecasterSarimax
from sklearn.metrics import mean_absolute_error
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

sales = pd.read_csv("sales.csv", index_col="Year-Month", parse_dates=True)
df = pd.DataFrame(sales)
df.index.freq = "MS"

df.columns = ["Sales"]
df.plot(figsize=(14,8))
plt.show()

print(df["Sales"].describe())
print("rng ", np.ptp(df["Sales"]))
print("var ", np.var(df["Sales"]))
print("covar ", np.cov(df["Sales"]))

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ["ADF Test Statistic","p-value","Number of Lags Used","Number of Observations Used"]
    for value,label in zip(result,labels):
        print(label + ' : ' + str(value))
    if result[1] <= 0.05:
        print("Los datos son estacionarios") #Se rechaza la hipotesis nula (Ho)
    else:
        print("Los datos NO son estacionarios") #Se acepta la hipotesis nula (H1)

adfuller_test(df["Sales"])
df["Sales Difference"] = df["Sales"] - df["Sales"].shift(1)
adfuller_test(df["Sales Difference"].dropna())

plot_acf(df["Sales Difference"].iloc[2:])
plot_pacf(df["Sales Difference"].iloc[2:])
plt.show()

df["Sales Difference"] = df["Sales Difference"].fillna(0)
decompose = seasonal_decompose(df["Sales"], model="additive", extrapolate_trend="freq")
descompose_differential = seasonal_decompose(df["Sales Difference"], model="additive", extrapolate_trend="freq")
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9, 6), sharex=True)
decompose.observed.plot(ax=axs[0, 0])
axs[0, 0].set_title("Original Sales")
decompose.trend.plot(ax=axs[1, 0])
axs[1, 0].set_title("Tendency")
decompose.seasonal.plot(ax=axs[2, 0])
axs[2, 0].set_title("Stacionality")
decompose.resid.plot(ax=axs[3, 0])
axs[3, 0].set_title("Residues")
descompose_differential.observed.plot(ax=axs[0, 1])
axs[0, 1].set_title("Differenciated Sales")
descompose_differential.trend.plot(ax=axs[1, 1])
axs[1, 1].set_title("Tendency")
descompose_differential.seasonal.plot(ax=axs[2, 1])
axs[2, 1].set_title("Stacionality")
descompose_differential.resid.plot(ax=axs[3, 1])
axs[3, 1].set_title("Residues")
fig.suptitle("Decomposition between original and differenciated sales", fontsize=20)
fig.tight_layout()
plt.show()

model = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model.fit(y=df["Sales"])
model.summary()
predictions = model.predict(steps=12)
fig, ax = plt.subplots(figsize=(7, 3))
df["Sales"].plot(ax=ax, label="Sales")
predictions.columns = ["Predictions"]
predictions.plot(ax=ax, label="Predictions")
ax.set_title("Predictions with SARIMA model")
ax.legend()
plt.show()