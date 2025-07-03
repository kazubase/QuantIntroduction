#%% データ収集と可視化

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語表示のため
import datetime # datetimeライブラリを追加
import seaborn as sns # seabornライブラリを追加

# 日本語表示の設定
japanize_matplotlib.japanize()

# 分析期間の定義 (リーマンショックを含む2007年以降)
start_date = "2007-01-01"
end_date = datetime.date.today().strftime("%Y-%m-%d") # 今日の日付を終了日として設定

# データの取得
sp500 = yf.download("^GSPC", start=start_date, end=end_date)
VIX = yf.download("^VIX", start=start_date, end=end_date)

# データ取得の確認
if sp500.empty:
    print("S&P 500のデータを取得できませんでした。ティッカーシンボルまたは期間を確認してください。")
    exit() # スクリプトを終了
if VIX.empty:
    print("VIX指数のデータを取得できませんでした。ティッカーシンボルまたは期間を確認してください。")
    exit() # スクリプトを終了

# 終値データの結合
# concatを使用して、日付インデックスに基づいて結合し、後で列名を変更します。
df = pd.concat([sp500["Close"], VIX["Close"]], axis=1)
df.columns = ["SP500_Close", "VIX_Close"]

# 欠損値の確認と処理 (必要であれば)
df.dropna(inplace=True)

# グラフの作成
fig, ax1 = plt.subplots(figsize=(14, 7))

# S&P 500を左Y軸にプロット
color = 'tab:blue'
ax1.set_xlabel('日付')
ax1.set_ylabel('S&P 500 終値', color=color)
ax1.plot(df.index, df['SP500_Close'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# VIX指数を右Y軸にプロット
ax2 = ax1.twinx()  # ax1とX軸を共有する新しいY軸を作成
color = 'tab:red'
ax2.set_ylabel('VIX指数', color=color)  # Y軸ラベル
ax2.plot(df.index, df['VIX_Close'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

# グラフのタイトル
plt.title('S&P 500とVIX指数の推移')

# グリッドの表示
ax1.grid(True)

# グラフの表示
plt.show()

#%% VIXとS&P 500の関係分析

# 日次リターンの計算
df['SP500_Daily_Return'] = df['SP500_Close'].pct_change()
df['VIX_Daily_Return'] = df['VIX_Close'].pct_change()

# 週次リターンの計算 (金曜日を週の終わりとする)
df_weekly = df.resample('W').last() # 週ごとの最終日を取得
df_weekly['SP500_Weekly_Return'] = df_weekly['SP500_Close'].pct_change()
df_weekly['VIX_Weekly_Return'] = df_weekly['VIX_Close'].pct_change()

# 欠損値の除去
df.dropna(inplace=True)
df_weekly.dropna(inplace=True)

# 相関関係の分析
daily_correlation = df['SP500_Daily_Return'].corr(df['VIX_Daily_Return'])
weekly_correlation = df_weekly['SP500_Weekly_Return'].corr(df_weekly['VIX_Weekly_Return'])

print(f"S&P 500とVIX指数の日次リターンの相関: {daily_correlation:.4f}")
print(f"S&P 500とVIX指数の週次リターンの相関: {weekly_correlation:.4f}")

#%% 回帰分析と予測

import statsmodels.api as sm

# 線形回帰モデルの構築
X = df['VIX_Daily_Return']
y = df['SP500_Daily_Return']
X = sm.add_constant(X)  # 定数項を追加

model = sm.OLS(y, X)
results = model.fit()

# 回帰分析結果の表示
print(results.summary())

# 回帰直線のプロット
plt.figure(figsize=(10, 6))
plt.scatter(df['VIX_Daily_Return'], df['SP500_Daily_Return'], alpha=0.5)
plt.plot(df['VIX_Daily_Return'], results.predict(X), color='red')
plt.title('S&P 500とVIX指数の日次リターン回帰分析')
plt.xlabel('VIX日次リターン')
plt.ylabel('S&P 500日次リターン')
plt.grid(True)
plt.show()

#%% ボラティリティ分析

# 20日間の移動相関を計算
df['Rolling_Correlation'] = df['SP500_Daily_Return'].rolling(window=20).corr(df['VIX_Daily_Return'])

# S&P 500の20日間の移動標準偏差（ボラティリティ）を計算
df['SP500_Rolling_Std'] = df['SP500_Daily_Return'].rolling(window=20).std()

# グラフの作成
fig, ax1 = plt.subplots(figsize=(14, 7))

# 移動相関を左Y軸にプロット
color = 'tab:purple'
ax1.set_xlabel('日付')
ax1.set_ylabel('移動相関 (S&P 500 vs VIX)', color=color)
ax1.plot(df.index, df['Rolling_Correlation'], color=color, label='S&P 500 vs VIX 移動相関')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

# S&P 500の移動標準偏差を右Y軸にプロット
ax2 = ax1.twinx()  # ax1とX軸を共有する新しいY軸を作成
color = 'tab:green'
ax2.set_ylabel('S&P 500 移動標準偏差', color=color)  # Y軸ラベル
ax2.plot(df.index, df['SP500_Rolling_Std'], color=color, label='S&P 500 移動標準偏差')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

# グラフのタイトル
plt.title('S&P 500とVIX指数の移動相関とS&P 500ボラティリティ')

# グリッドの表示
ax1.grid(True)

# グラフの表示
plt.show()

#%% リターンとVIXの相関分析

# S&P 500の日次リターンは既に計算済み (df['SP500_Daily_Return'])

# VIX指数の日次変化量（差分）を計算
df['VIX_Daily_Change'] = df['VIX_Close'].diff()

# 欠損値の除去
df_analysis = df.dropna(subset=['SP500_Daily_Return', 'VIX_Daily_Change'])

# 相関係数を計算し、表示
correlation_return_vix_change = df_analysis['SP500_Daily_Return'].corr(df_analysis['VIX_Daily_Change'])
print(f"S&P 500日次リターンとVIX日次変化量の相関: {correlation_return_vix_change:.4f}")
# 一般的に、S&P 500が下落する際にはVIXが上昇するため、強い負の相関が期待されます。

# regplotを使って散布図と回帰直線を可視化
plt.figure(figsize=(10, 7))
sns.regplot(x='SP500_Daily_Return', y='VIX_Daily_Change', data=df_analysis, scatter_kws={'alpha':0.3})
plt.title('S&P 500日次リターン vs VIX日次変化量')
plt.xlabel('S&P 500 日次リターン')
plt.ylabel('VIX 日次変化量')
plt.grid(True)
plt.show()

#%% VIX指数とS&P 500の非対称性分析

# S&P 500の日次リターンとVIX日次変化量がすでに計算されていることを前提とします

# S&P 500の上下動に基づくVIX変化量の非対称性を分析
# S&P 500が上昇した日と下落した日にVIXがどのように変化するかを比較します。

# S&P 500の上昇日と下落日を定義
# ここでは、S&P 500のリターンが0より大きい場合を「上昇日」、0以下の場合を「下落日」とします。
threshold = 0 # 例: 0%以上のリターンを上昇日とする
df_up_days = df_analysis[df_analysis['SP500_Daily_Return'] > threshold]
df_down_days = df_analysis[df_analysis['SP500_Daily_Return'] <= threshold]

# 各グループにおけるVIX日次変化量の平均を計算
mean_vix_change_up = df_up_days['VIX_Daily_Change'].mean()
mean_vix_change_down = df_down_days['VIX_Daily_Change'].mean()

print(f"S&P 500上昇日のVIX日次変化量の平均: {mean_vix_change_up:.4f}")
print(f"S&P 500下落日のVIX日次変化量の平均: {mean_vix_change_down:.4f}")

# 結果を棒グラフで可視化
plt.figure(figsize=(8, 6))
labels = ['S&P 500 上昇日', 'S&P 500 下落日']
values = [mean_vix_change_up, mean_vix_change_down]
colors = ['skyblue', 'salmon']

plt.bar(labels, values, color=colors)
plt.title('S&P 500の騰落別VIX日次変化量の平均')
plt.xlabel('S&P 500の動き')
plt.ylabel('VIX 日次変化量の平均')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#%% VIX水準による市場レジーム分析

# VIX指数の値に基づいて市場のレジームを定義する列を作成
def define_vix_regime(vix_value):
    if vix_value < 20:
        return '平穏（Calm）'
    elif 20 <= vix_value < 30:
        return '警戒（Nervous）'
    else:
        return 'パニック（Panic）'

df['VIX_Regime'] = df['VIX_Close'].apply(define_vix_regime)

# 各レジームにおけるS&P 500の日次リターンの分布をboxplotで可視化
plt.figure(figsize=(12, 7))
sns.boxplot(x='VIX_Regime', y='SP500_Daily_Return', data=df, order=['平穏（Calm）', '警戒（Nervous）', 'パニック（Panic）'])
plt.title('VIX水準別S&P 500日次リターン分布')
plt.xlabel('VIX市場レジーム')
plt.ylabel('S&P 500 日次リターン')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 各レジームにおけるS&P 500の日次リターンの平均値と標準偏差を計算し、表示
regime_analysis = df.groupby('VIX_Regime')['SP500_Daily_Return'].agg(['mean', 'std']).loc[['平穏（Calm）', '警戒（Nervous）', 'パニック（Panic）']]
print("\n各VIXレジームにおけるS&P 500日次リターンの統計:")
print(regime_analysis)

# 結果のコメント
# パニック相場（VIX >= 30）では、リターンの平均はマイナスに傾き、
# ボラティリティ（標準偏差）が劇的に増大することが確認できます。

#%% ボラティリティ・クラスタリングの確認

# S&P 500の日次リターンの絶対値を計算（日々のボラティリティの代理変数）
df['SP500_Daily_Volatility'] = df['SP500_Daily_Return'].abs()

# 時系列グラフをプロット
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['SP500_Daily_Volatility'], label='S&P 500 日次ボラティリティ (絶対値)', color='purple', alpha=0.7)
plt.title('S&P 500 日次ボラティリティの推移')
plt.xlabel('日付')
plt.ylabel('日次ボラティリティ (絶対値)')
plt.grid(True)
plt.legend()
plt.show()

# コメント: ボラティリティ・クラスタリングについて
# グラフから、変動が大きい時期（例えば、リーマンショックやコロナショックなどの危機時）は
# その後も変動が大きい状態が続く傾向があり、逆に変動が小さい時期も固まって発生していることが確認できます。
# このように変動が集中する現象を「ボラティリティ・クラスタリング」と呼びます。
# この特性は、GARCH（Generalized Autoregressive Conditional Heteroskedasticity）モデルのような、
# 時系列にわたるボラティリティの予測モデルが必要とされる主な理由の一つです。

#%% VIXと恐怖指数のスプレッド分析

# S&P 500の終値とVIX指数の比率（スプレッド）を計算
# この比率が高いほど市場は「平穏」であり、低いほど「恐怖」が蔓延していると解釈できます。
df['SP500_VIX_Ratio'] = df['SP500_Close'] / df['VIX_Close']

# スプレッドのヒストグラムをプロット
plt.figure(figsize=(10, 6))
sns.histplot(df['SP500_VIX_Ratio'].dropna(), bins=50, kde=True)
plt.title('S&P 500終値とVIX指数比率の分布')
plt.xlabel('S&P 500 / VIX 比率')
plt.ylabel('頻度')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# コメント: ヒストグラムの解釈
# ヒストグラムを見ると、比率の分布がどの水準に集中しているかがわかります。
# 例えば、非常に低い比率の領域（左側）は、VIXがS&P 500と比較して異常に高い状態、
# すなわち市場の極度の恐怖やパニックを示唆しています。
# 逆に、比率が高い領域（右側）は、市場が比較的平穏であることを示しています。

# %%
