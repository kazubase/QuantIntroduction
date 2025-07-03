#%% データ準備とライブラリのインポート
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Prophetはfbprophetまたはprophetとしてインポートされる場合があります
try:
    from prophet import Prophet
except ImportError:
    from fbprophet import Prophet
import japanize_matplotlib  # 日本語プロット対応
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%% データ取得
# Appleの株価（AAPL）を過去5年分取得
symbol = "AAPL"
data = yf.download(symbol, period="5y", group_by='ticker')
# カラム構造を確認
print(data.columns)
# MultiIndexの場合も考慮して調整後終値または終値を取得
ts_data = None
if ("Adj Close" in data.columns):
    ts_data = data["Adj Close"]
elif (symbol, "Adj Close") in data.columns:
    ts_data = data[(symbol, "Adj Close")]
elif (symbol, "Close") in data.columns:
    ts_data = data[(symbol, "Close")]
elif "Close" in data.columns:
    ts_data = data["Close"]
else:
    print("利用可能なカラム:", data.columns)
    raise KeyError("'Adj Close'や'Close'カラムが見つかりません")

#%% トレンドの確認
plt.figure(figsize=(12, 4))
ts_data.plot()
plt.title("AAPLの終値（過去5年）")
plt.xlabel("日付")
plt.ylabel("終値")
plt.show()
# ヒント: 右肩上がりのトレンドが見られ、非定常性が疑われる

#%% d（階差）の決定
# 1階差分を計算
ts_diff = ts_data.diff().dropna()
plt.figure(figsize=(12, 4))
ts_diff.plot()
plt.title("AAPLの終値 1階差分")
plt.xlabel("日付")
plt.ylabel("差分値")
plt.show()
# ヒント: トレンドが除去され、平均0の周りで変動する定常性に近いデータとなった
# → ARIMAモデルのパラメータdは1が妥当だろう

#%% pとqの決定
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plot_acf(ts_diff, ax=plt.gca(), lags=30)
plt.title("ACF（自己相関）")
plt.subplot(1, 2, 2)
plot_pacf(ts_diff, ax=plt.gca(), lags=30, method="ywm")
plt.title("PACF（偏自己相関）")
plt.tight_layout()
plt.show()
# ヒント: PACFを見るとラグXで急に相関がなくなるのでp=Xか？
#         ACFを見るとラグYで急に相関がなくなるのでq=Yか？

#%% ARIMAモデルによる予測
# データ分割（過去4年を学習用、直近1年をテスト用）
split_date = ts_data.index[-252]  # おおよそ1年分（営業日252日）
train = ts_data[:split_date]
test = ts_data[split_date:]

# ARIMAパラメータの決定（例: p=5, d=1, q=0）
p, d, q = 5, 1, 0
# 理由: PACFがラグ5で急減、ACFがすぐ減衰→p=5, q=0, d=1（1階差分で定常化）

# モデル構築と学習
to_fit = train.dropna()  # 欠損値除去
model = ARIMA(to_fit, order=(p, d, q))
model_fit = model.fit()

# テスト期間の予測（start/endを整数で指定）
start = len(train)
end = len(train) + len(test) - 1
pred = model_fit.get_prediction(start=start, end=end)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

# 予測結果のインデックスをテストデータのインデックスに合わせる
pred_mean.index = test.index
pred_ci.index = test.index

# プロット
plt.figure(figsize=(14, 6))
plt.plot(ts_data, label="実績値", color="black")
plt.plot(pred_mean, label="ARIMA予測値", color="red")
plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="pink", alpha=0.3, label="予測95%信頼区間")
plt.axvline(split_date, color="gray", linestyle="--", label="学習/テスト分割")
plt.title("AAPL株価のARIMA予測（p=5, d=1, q=0）")
plt.xlabel("日付")
plt.ylabel("終値")
plt.legend()
plt.tight_layout()
plt.show()

#%% Prophetモデルによる予測
# Prophet用にカラム名を変換
ts_df = ts_data.reset_index()
ts_df.columns = ['ds', 'y']

# データ分割（過去4年を学習用、直近1年をテスト用）
train_df = ts_df.iloc[:-252]
test_df = ts_df.iloc[-252:]

# Prophetモデルのインスタンス化と学習
to_fit_df = train_df.dropna()
model_prophet = Prophet()
model_prophet.fit(to_fit_df)

# 未来のデータフレーム作成（テスト期間分）
future = model_prophet.make_future_dataframe(periods=len(test_df), freq='B')  # 'B'は営業日
forecast = model_prophet.predict(future)

# 実績値と予測値のプロット
fig1 = model_prophet.plot(forecast)
plt.title("AAPL株価のProphet予測")
plt.xlabel("日付")
plt.ylabel("終値")
plt.tight_layout()
plt.show()

# トレンド・季節性の分解プロット
fig2 = model_prophet.plot_components(forecast)
plt.tight_layout()
plt.show()

#%% モデル精度の評価
from sklearn.metrics import mean_squared_error, mean_absolute_error

# テスト期間の実績値
actual = test.values

# ARIMA予測値
arima_pred = pred_mean.values
arima_mse = mean_squared_error(actual, arima_pred)
arima_mae = mean_absolute_error(actual, arima_pred)

# Prophet予測値（forecastのyhat列からテスト期間分を抽出）
prophet_pred = forecast.iloc[-len(test):]["yhat"].values
prophet_mse = mean_squared_error(actual, prophet_pred)
prophet_mae = mean_absolute_error(actual, prophet_pred)

print("==== モデル精度の評価（テスト期間） ====")
print(f"ARIMAモデル:    MSE={arima_mse:.2f}, MAE={arima_mae:.2f}")
print(f"Prophetモデル:  MSE={prophet_mse:.2f}, MAE={prophet_mae:.2f}")

if arima_mse < prophet_mse:
    print("→ ARIMAモデルの方がMSEが小さく、精度が高いです。")
elif arima_mse > prophet_mse:
    print("→ Prophetモデルの方がMSEが小さく、精度が高いです。")
else:
    print("→ 両モデルのMSEは同じです。")

# %%