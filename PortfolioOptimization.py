#%% データ準備
# 必要なライブラリのインポート
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import datetime
from scipy.optimize import minimize

# 分析対象のティッカーと期間を定義
# 例: 3年分のデータを取得
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=3*365)
tickers = ["NVDA", "MSFT", "GOOGL", "AMZN", "META"]

# yfinanceでデータ取得
raw_data = yf.download(tickers, start=start_date, end=end_date)

# 'Adj Close'が存在すればそれを使い、なければ'Close'を使う
if 'Adj Close' in raw_data.columns.get_level_values(0):
    data = raw_data['Adj Close']
else:
    data = raw_data['Close']

# 日次リターンを計算
returns = data.pct_change().dropna()

# 年率リターンと年率ボラティリティ（リスク）を計算
annual_return = returns.mean() * 252
annual_volatility = returns.std() * np.sqrt(252)

# 結果を表示
print("年率リターン:")
print(annual_return)
print("\n年率ボラティリティ（リスク）:")
print(annual_volatility)

#%% モンテカルロ・シミュレーション
# シミュレーション回数と銘柄数を定義
num_simulations = 10000
num_assets = len(tickers)

# 結果を保存するリストを準備
portfolio_returns = []
portfolio_risks = []
portfolio_sharpe = []
portfolio_weights = []

# モンテカルロ・シミュレーション
for _ in range(num_simulations):
    # ランダムなウェイトを生成し、合計が1になるよう正規化
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    
    # ポートフォリオの年率リターンとリスクを計算
    port_return = np.dot(annual_return, weights)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    
    # シャープレシオ（リスクフリーレートは0と仮定）
    sharpe_ratio = port_return / port_risk if port_risk != 0 else 0
    
    # 結果をリストに追加
    portfolio_returns.append(port_return)
    portfolio_risks.append(port_risk)
    portfolio_sharpe.append(sharpe_ratio)
    portfolio_weights.append(weights)

# 結果をDataFrameにまとめる
simulation_df = pd.DataFrame({
    'Return': portfolio_returns,
    'Risk': portfolio_risks,
    'Sharpe': portfolio_sharpe
})
# 各ウェイトも列として追加
for i, ticker in enumerate(tickers):
    simulation_df[ticker + ' Weight'] = [w[i] for w in portfolio_weights]

#%% 結果の可視化と最適点の特定
# 最小リスクポートフォリオと最大シャープレシオポートフォリオを特定
min_volatility_idx = simulation_df['Risk'].idxmin()
max_sharpe_idx = simulation_df['Sharpe'].idxmax()
min_volatility_port = simulation_df.loc[min_volatility_idx]
max_sharpe_port = simulation_df.loc[max_sharpe_idx]

# 散布図の作成
plt.figure(figsize=(16, 8))
sc = plt.scatter(
    simulation_df['Risk'],
    simulation_df['Return'],
    c=simulation_df['Sharpe'],
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(sc, label='シャープレシオ')

# 最小リスクポートフォリオ（赤い星）
plt.scatter(
    min_volatility_port['Risk'],
    min_volatility_port['Return'],
    color='red',
    marker='*',
    s=400,
    label='最小リスクポートフォリオ'
)
# 最大シャープレシオポートフォリオ（緑の星）
plt.scatter(
    max_sharpe_port['Risk'],
    max_sharpe_port['Return'],
    color='green',
    marker='*',
    s=400,
    label='最大シャープレシオポートフォリオ'
)

plt.title('ポートフォリオ・モンテカルロシミュレーション結果')
plt.xlabel('リスク（年率ボラティリティ）')
plt.ylabel('リターン（年率）')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% 最適ポートフォリオの詳細
print("【最大シャープレシオポートフォリオ（最強チーム）】")
print(f"  年率リターン: {max_sharpe_port['Return']:.2%}")
print(f"  年率リスク: {max_sharpe_port['Risk']:.2%}")
print(f"  シャープレシオ: {max_sharpe_port['Sharpe']:.3f}")
print("  銘柄ごとの配分:")
for ticker in tickers:
    weight = max_sharpe_port[ticker + ' Weight']
    print(f"    {ticker}: {weight*100:.2f}%")

print("\n【最小リスクポートフォリオ（最も安全なチーム）】")
print(f"  年率リターン: {min_volatility_port['Return']:.2%}")
print(f"  年率リスク: {min_volatility_port['Risk']:.2%}")
print(f"  シャープレシオ: {min_volatility_port['Sharpe']:.3f}")
print("  銘柄ごとの配分:")
for ticker in tickers:
    weight = min_volatility_port[ticker + ' Weight']
    print(f"    {ticker}: {weight*100:.2f}%")

#%% 数学的最適化（SciPyによる厳密解）

# ポートフォリオのリターン、リスク、シャープレシオを計算する関数

def calc_portfolio_performance(weights, mean_returns, cov_matrix):
    port_return = np.dot(mean_returns, weights)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = port_return / port_risk if port_risk != 0 else 0
    return port_return, port_risk, sharpe

# 最適化の目的関数（負のシャープレシオを最小化）
def neg_sharpe_ratio(weights, mean_returns, cov_matrix):
    return -calc_portfolio_performance(weights, mean_returns, cov_matrix)[2]

# 制約条件: ウェイトの合計が1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# 各ウェイトは0以上1以下
bounds = tuple((0, 1) for _ in range(num_assets))
# 初期値（均等配分）
init_guess = num_assets * [1. / num_assets]

# 最適化実行
opt_result = minimize(
    neg_sharpe_ratio,
    init_guess,
    args=(annual_return.values, returns.cov() * 252),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# 最適ウェイトとパフォーマンス
opt_weights = opt_result.x
opt_return, opt_risk, opt_sharpe = calc_portfolio_performance(opt_weights, annual_return.values, returns.cov() * 252)

print("【SciPyによる厳密最適化】")
print(f"  年率リターン: {opt_return:.2%}")
print(f"  年率リスク: {opt_risk:.2%}")
print(f"  シャープレシオ: {opt_sharpe:.3f}")
print("  銘柄ごとの配分:")
for ticker, weight in zip(tickers, opt_weights):
    print(f"    {ticker}: {weight*100:.2f}%")
