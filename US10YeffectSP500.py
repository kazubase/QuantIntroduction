#%% データ収集と準備
import yfinance as yf
import pandas_datareader.data as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from datetime import datetime, timedelta
import seaborn as sns
import statsmodels.formula.api as smf
# from fredapi import Fred # fredapiはコメントアウトまたは削除

# 日本語フォント設定
plt.rcParams['font.family'] = 'IPAexGothic'

# 分析期間の設定（過去20年）
end = datetime.today()
start = end - timedelta(days=365*20)

# FRED APIキー設定は不要となるため削除

# 米国10年債利回り（DGS10）データ取得（FRED）
dgs10_data = pdr.DataReader('DGS10', 'fred', start, end)
if not dgs10_data.empty:
    dgs10_series = dgs10_data.iloc[:, 0].squeeze()
    if not isinstance(dgs10_series, pd.Series):
        dgs10_series = pd.Series([dgs10_series], index=[dgs10_data.index[0]]) # 単一値の場合にSeriesに変換
else:
    print("Warning: DGS10 data could not be retrieved or is empty.")
    dgs10_series = pd.Series(dtype=float)
print(f"DEBUG: Type of dgs10_series after init: {type(dgs10_series)}")
print(f"DEBUG: Head of dgs10_series after init:\n{dgs10_series.head()}")

# S&P 500価格データ取得（yfinance）
gspc = yf.download('^GSPC', start=start, end=end)
if not gspc.empty:
    gspc_close_series = gspc['Close'].squeeze()
    if not isinstance(gspc_close_series, pd.Series):
        gspc_close_series = pd.Series([gspc_close_series], index=[gspc.index[0]]) # 単一値の場合にSeriesに変換
else:
    print("Warning: S&P 500 data could not be retrieved or is empty.")
    gspc_close_series = pd.Series(dtype=float)
print(f"DEBUG: Type of gspc_close_series after init: {type(gspc_close_series)}")
print(f"DEBUG: Head of gspc_close_series after init:\n{gspc_close_series.head()}")

# Shillerデータ（ie_data.xls）を読み込む
# 'Data'シートにデータがあると仮定。実際のデータは9行目から始まるため、header=8を設定。
# 必要な列のみを読み込む: Date (A列), Shiller PER (M列)
shiller_df = pd.read_excel(
    'ie_data.xls',
    sheet_name='Data',
    header=8,
    usecols='A,M',
    names=['Date', 'Shiller PER'], # 直接列名を設定
    dtype={'Date': str} # Date列を文字列として読み込む
)

# Date列の文字列をYYYY.MM形式とみなし、日付型に変換
# format='%Y.%m' を明示的に指定
shiller_df['Date'] = pd.to_datetime(shiller_df['Date'], format='%Y.%m', errors='coerce')

# NaNの日付行を削除し、Date列をインデックスに設定
shiller_df = shiller_df.dropna(subset=['Date']).set_index('Date')

# インデックスをソート
shiller_df = shiller_df.sort_index()

# 未来の日付のデータやNaN値を削除
# このフィルタリングは、Dateのパースが正しく行われた後に行う
shiller_df = shiller_df[shiller_df.index <= end].dropna()

print(f"DEBUG: Type of shiller_df after init: {type(shiller_df)}")
print(f"DEBUG: Index of shiller_df after init: {shiller_df.index.dtype}")
print(f"DEBUG: Head of shiller_df after init:\n{shiller_df.head()}")
print(f"DEBUG: Tail of shiller_df after init:\n{shiller_df.tail()}")

# Shiller PERデータが空でないことを確認
if shiller_df.empty or 'Shiller PER' not in shiller_df.columns:
    print("Warning: Shiller PER data could not be retrieved or is empty.")
    shiller_per_monthly = pd.Series(dtype=float)
else:
    # Shiller PERを月次データにリサンプリングし、最後の値を取得
    # resample後にNaNが生じることがあるので、interpolateで補間してからdropna
    shiller_per_monthly = shiller_df['Shiller PER'].resample('ME').last().interpolate().dropna()
    if not isinstance(shiller_per_monthly, pd.Series):
        shiller_per_monthly = pd.Series([shiller_per_monthly], index=[shiller_df.index[0]]) # 単一値の場合にSeriesに変換
print(f"DEBUG: Type of shiller_per_monthly after resample: {type(shiller_per_monthly)}")
print(f"DEBUG: Head of shiller_per_monthly after resample:\n{shiller_per_monthly.head()}")
print(f"DEBUG: Tail of shiller_per_monthly after resample:\n{shiller_per_monthly.tail()}")

# 月次データにリサンプリング
# 金利は月末値、S&P500終値
# ここでrate_monthlyとgspc_monthlyがSeriesであることを確認
if not dgs10_series.empty:
    rate_monthly = dgs10_series.resample('ME').last().squeeze()
    if not isinstance(rate_monthly, pd.Series):
        rate_monthly = pd.Series([rate_monthly], index=[dgs10_series.index[0]])
else:
    rate_monthly = pd.Series(dtype=float)
print(f"DEBUG: Type of rate_monthly after resample: {type(rate_monthly)}")
print(f"DEBUG: Head of rate_monthly after resample:\n{rate_monthly.head()}")

if not gspc_close_series.empty:
    gspc_monthly = gspc_close_series.resample('ME').last().squeeze()
    if not isinstance(gspc_monthly, pd.Series):
        gspc_monthly = pd.Series([gspc_monthly], index=[gspc_close_series.index[0]])
else:
    gspc_monthly = pd.Series(dtype=float)
print(f"DEBUG: Type of gspc_monthly after resample: {type(gspc_monthly)}")
print(f"DEBUG: Head of gspc_monthly after resample:\n{gspc_monthly.head()}")

# すべてのデータの共通インデックスを計算
# ここで各Seriesが空でないか、そしてIndex属性を持つことを保証
if rate_monthly.empty or gspc_monthly.empty or shiller_per_monthly.empty:
    print("Warning: One or more monthly data series are empty. Cannot calculate common index.")
    all_common_index = pd.Index([])
else:
    # startとendの期間でデータをフィルタリング
    rate_monthly = rate_monthly[(rate_monthly.index >= start) & (rate_monthly.index <= end)]
    gspc_monthly = gspc_monthly[(gspc_monthly.index >= start) & (gspc_monthly.index <= end)]
    shiller_per_monthly = shiller_per_monthly[(shiller_per_monthly.index >= start) & (shiller_per_monthly.index <= end)]

    all_common_index = rate_monthly.index.intersection(gspc_monthly.index).intersection(shiller_per_monthly.index)

# データ統合
if not all_common_index.empty:
    result_df = pd.DataFrame({
        'US10Y_Rate': rate_monthly.loc[all_common_index],
        'SP500_Close': gspc_monthly.loc[all_common_index],
        'SP500_ShillerPER': shiller_per_monthly.loc[all_common_index]
    })
    result_df = result_df.dropna() # dropnaを最後にすることで、インデックスが揃っていないことによるNaNを処理する
else:
    print("Warning: Common index is empty. result_df will be an empty DataFrame.")
    result_df = pd.DataFrame(columns=['US10Y_Rate', 'SP500_Close', 'SP500_ShillerPER'])

# データ確認
print(result_df.head())

# デバッグ用の出力追加
print("\n--- Debug Info ---")
print(f"Type of dgs10_series: {type(dgs10_series)}")
print(f"Head of dgs10_series:\n{dgs10_series.head()}")
print(f"Type of gspc_close_series: {type(gspc_close_series)}")
print(f"Head of gspc_close_series:\n{gspc_close_series.head()}")
print(f"Type of shiller_df: {type(shiller_df)}")
print(f"Head of shiller_df:\n{shiller_df.head()}")
print(f"Type of rate_monthly: {type(rate_monthly)}")
print(f"Head of rate_monthly:\n{rate_monthly.head()}")
print(f"Type of gspc_monthly: {type(gspc_monthly)}")
print(f"Head of gspc_monthly:\n{gspc_monthly.head()}")
print(f"Type of shiller_per_monthly: {type(shiller_per_monthly)}")
print(f"Head of shiller_per_monthly:\n{shiller_per_monthly.head()}")
print("------------------")

#%% 外れ値の除去

print(f"外れ値除去前のデータ数: {len(result_df)}")

# シラーPERが極端に低い（大きな負の値）を外れ値と見なす
# グラフの外れ値は-30000近くまで落ち込んでいることから、
# -10000よりも小さい値を外れ値として除去する
threshold_per_lower = -10000 

result_df = result_df[result_df['SP500_ShillerPER'] >= threshold_per_lower]

print(f"外れ値除去後のデータ数: {len(result_df)}")
print(result_df.head())

#%% データの可視化
plt.figure(figsize=(14, 7))
fig, ax1 = plt.subplots(figsize=(14, 7))

# 左Y軸: S&P500のシラーPER
color = 'tab:blue'
ax1.set_xlabel('日付')
ax1.set_ylabel('S&P500 シラーPER', color=color)
l1 = ax1.plot(result_df.index, result_df['SP500_ShillerPER'], color=color, label='S&P500 シラーPER')
ax1.tick_params(axis='y', labelcolor=color)

# 右Y軸: 米国10年債利回り
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('米国10年債利回り（％）', color=color)
l2 = ax2.plot(result_df.index, result_df['US10Y_Rate'], color=color, label='米国10年債利回り')
ax2.tick_params(axis='y', labelcolor=color)

# タイトルと凡例
plt.title('S&P 500シラーPERと米国10年債利回りの推移')
lns = l1 + l2
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc='upper left')

plt.tight_layout()
plt.show()

# %% ここはデータ確認のためにコメントアウトを推奨

#%% 相関分析と回帰分析

# 相関係数の計算
corr = result_df['US10Y_Rate'].corr(result_df['SP500_ShillerPER'])
print(f'10年債利回りとS&P500シラーPERの相関係数: {corr:.3f}')
# 通常、金利とPERは負の相関が期待されます（＝金利が上がるとPERは下がりやすい）

# 散布図と回帰直線
plt.figure(figsize=(10, 6))
sns.regplot(x='US10Y_Rate', y='SP500_ShillerPER', data=result_df)
plt.xlabel('米国10年債利回り（％）')
plt.ylabel('S&P500 シラーPER')
plt.title('米国10年債利回りとS&P500シラーPERの関係')
plt.tight_layout()
plt.show()

# 単回帰分析
model = smf.ols('SP500_ShillerPER ~ US10Y_Rate', data=result_df)
result = model.fit()
print(result.summary())

#%% 理論PERとの比較

# 回帰モデルの係数を取得
intercept = result.params[0]
slope = result.params[1]

# 理論PERの計算
result_df['Theoretical_PER'] = intercept + slope * result_df['US10Y_Rate']

# 実際のPERと理論PERの比較グラフ
plt.figure(figsize=(14, 7))
plt.plot(result_df.index, result_df['SP500_ShillerPER'], label='S&P500 実際のシラーPER', color='blue')
plt.plot(result_df.index, result_df['Theoretical_PER'], label='理論PER', color='red', linestyle='--')
plt.xlabel('日付')
plt.ylabel('PER')
plt.title('S&P500 実際のシラーPERと理論PERの比較')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
