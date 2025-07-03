#%% データ準備とテクニカル指標の計算
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams['font.family'] = 'IPAexGothic'  # 日本語フォント設定（インストール済み前提）
plt.rcParams['axes.unicode_minus'] = False

# 日経平均株価（^N225）の過去10年分データ取得
symbol = '^N225'
df = yf.download(symbol, period='10y')

# 短期移動平均線（50日）と長期移動平均線（200日）を計算
short_window = 50
long_window = 200
df['SMA50'] = df['Close'].rolling(window=short_window).mean()
df['SMA200'] = df['Close'].rolling(window=long_window).mean()

#%% 移動平均線と終値のプロット
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['Close'], label='終値', color='black')
plt.plot(df.index, df['SMA50'], label=f'{short_window}日移動平均', color='blue', alpha=0.7)
plt.plot(df.index, df['SMA200'], label=f'{long_window}日移動平均', color='red', alpha=0.7)
plt.title('日経平均株価と移動平均線')
plt.xlabel('日付')
plt.ylabel('価格')
plt.legend()
plt.grid(True)
plt.show()

#%% 売買シグナルの生成
# ゴールデンクロス: 昨日SMA50 < SMA200 かつ 今日SMA50 > SMA200 → 1（買い）
# デッドクロス: 昨日SMA50 > SMA200 かつ 今日SMA50 < SMA200 → -1（売り）
df['signal'] = 0
condition_buy = (df['SMA50'].shift(1) < df['SMA200'].shift(1)) & (df['SMA50'] > df['SMA200'])
condition_sell = (df['SMA50'].shift(1) > df['SMA200'].shift(1)) & (df['SMA50'] < df['SMA200'])
df['signal'] = np.where(condition_buy, 1, np.where(condition_sell, -1, 0))

# シグナルが発生した日だけ抽出して表示
signals = df[df['signal'] != 0][['Close', 'SMA50', 'SMA200', 'signal']]
print(signals)

#%% バックテストの実行
# シグナルに従いポジションを決定（シグナルが出たらそのポジションを次のシグナルまで維持）
df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)

# 日次リターンの計算
# 市場の日次リターン
df['market_return'] = np.log(df['Close'] / df['Close'].shift(1))
# 戦略の日次リターン（前日のポジションで当日のリターンを得る）
df['strategy_return'] = df['position'].shift(1) * df['market_return']

# 戦略の累積リターン（資産曲線）
df['strategy_equity'] = np.exp(df['strategy_return'].cumsum())
# バイ・アンド・ホールドの資産曲線
df['buy_and_hold'] = np.exp(df['market_return'].cumsum())

#%% パフォーマンスの評価と可視化
# 資産曲線の可視化
plt.figure(figsize=(16, 8))
plt.plot(df.index, df['strategy_equity'], label='ゴールデンクロス戦略')
plt.plot(df.index, df['buy_and_hold'], label='バイ・アンド・ホールド')
plt.title('資産曲線の比較')
plt.xlabel('日付')
plt.ylabel('資産（初期値=1）')
plt.legend()
plt.grid(True)
plt.show()

# パフォーマンス指標の計算
trading_days = 252  # 年間取引日数

# 最終累積リターン
final_strategy = df['strategy_equity'].iloc[-1]
final_bh = df['buy_and_hold'].iloc[-1]

# 年率リターン
annual_return_strategy = df['strategy_return'].mean() * trading_days
annual_return_bh = df['market_return'].mean() * trading_days

# 年率ボラティリティ
annual_vol_strategy = df['strategy_return'].std() * np.sqrt(trading_days)
annual_vol_bh = df['market_return'].std() * np.sqrt(trading_days)

# シャープレシオ（無リスク金利0%と仮定）
sharpe_strategy = annual_return_strategy / annual_vol_strategy if annual_vol_strategy != 0 else np.nan
sharpe_bh = annual_return_bh / annual_vol_bh if annual_vol_bh != 0 else np.nan

# 最大ドローダウン
cummax_strategy = df['strategy_equity'].cummax()
drawdown_strategy = df['strategy_equity'] / cummax_strategy - 1
max_dd_strategy = drawdown_strategy.min()

cummax_bh = df['buy_and_hold'].cummax()
drawdown_bh = df['buy_and_hold'] / cummax_bh - 1
max_dd_bh = drawdown_bh.min()

# 結果の表示
print('--- パフォーマンス指標 ---')
print('戦略\t\t\t最終リターン\t年率リターン\t年率ボラ\tシャープレシオ\t最大ドローダウン')
print(f'ゴールデンクロス\t{final_strategy:.3f}\t\t{annual_return_strategy:.3%}\t{annual_vol_strategy:.3%}\t{sharpe_strategy:.2f}\t\t{max_dd_strategy:.2%}')
print(f'バイ&ホールド\t{final_bh:.3f}\t\t{annual_return_bh:.3%}\t{annual_vol_bh:.3%}\t{sharpe_bh:.2f}\t\t{max_dd_bh:.2%}')

# %%
