#%% データ取得
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import japanize_matplotlib

# 分析対象のティッカーと期間を定義
tickers = ["ARM", "NVDA"]
start_date = "2024-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')

# 株価データを取得（キャッシュ無効化、今日まで）
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    progress=False,
    threads=False,
    interval="1d",
    auto_adjust=False
)

# データの最初の5行と最後の5行を表示
display(data.head())
display(data.tail())


#%% パフォーマンス分析
# Close価格を使って2024年の開始日を1として正規化リターンを計算
close = data['Close']
normalized_returns = close / close.iloc[0]

# 正規化リターンをプロットする関数
def plot_normalized_returns(returns_df):
    plt.figure(figsize=(14, 7))
    for col in returns_df.columns:
        plt.plot(returns_df.index, returns_df[col], label=col)
    plt.title("ARM vs NVIDIA 正規化リターン比較")
    plt.ylabel("リターン倍率")
    plt.axhline(1, color='gray', linestyle='--', linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# グラフを表示
plot_normalized_returns(normalized_returns)

#%% リスク分析
# Close価格から日次リターンを計算（Adj CloseがないためCloseを使用）
daily_returns = close.pct_change().dropna()

# 日次リターンの基本統計量を表示
# 標準偏差（std）はリスク（ボラティリティ）の指標となる
print(daily_returns.describe())

# 日次リターン分布をヒストグラムで可視化する関数
def plot_return_distribution(returns_df):
    plt.figure(figsize=(14, 7))
    for col in returns_df.columns:
        sns.histplot(returns_df[col], bins=50, kde=True, label=col, stat="density", alpha=0.5)
    plt.title("ARM vs NVIDIA 日次リターン分布")
    plt.xlabel("日次リターン")
    plt.ylabel("密度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# グラフを表示
plot_return_distribution(daily_returns)

#%% 相関分析
# 銘柄間の相関行列を計算・表示
correlation_matrix = daily_returns.corr()
print(correlation_matrix)

# ARMとNVIDIAのリターンの関係性を散布図＋回帰直線付きで可視化する関数
def plot_correlation_scatterplot(returns_df, ticker1, ticker2):
    sns.jointplot(
        x=returns_df[ticker1],
        y=returns_df[ticker2],
        kind='reg',
        height=8,
        marginal_kws=dict(bins=30, fill=True)
    )
    plt.suptitle(f"{ticker1} vs {ticker2} 日次リターン相関", y=1.02)
    plt.xlabel(f"{ticker1} 日次リターン")
    plt.ylabel(f"{ticker2} 日次リターン")
    plt.tight_layout()
    plt.show()

# グラフを表示
plot_correlation_scatterplot(daily_returns, "ARM", "NVDA")

#%% 結果サマリー
# 期間最終日の正規化リターン
final_returns = normalized_returns.iloc[-1]

# 年率換算リターンと年率換算ボラティリティ（リスク）
daily_mean = daily_returns.mean()
daily_std = daily_returns.std()
annual_return = daily_mean * 252
annual_volatility = daily_std * np.sqrt(252)

print("==== 結果サマリー ====")
for ticker in close.columns:
    print(f"【{ticker}】")
    print(f"  期間最終日の正規化リターン: {final_returns[ticker]:.3f}")
    print(f"  年率換算リターン: {annual_return[ticker]:.2%}")
    print(f"  年率換算ボラティリティ（リスク）: {annual_volatility[ticker]:.2%}")
    print("")

# 全てのグラフを一つのウィンドウにまとめて表示
plt.show()

