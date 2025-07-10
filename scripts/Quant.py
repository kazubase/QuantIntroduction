#%% [markdown]
# # 【分析レポート】ARM vs NVIDIA パフォーマンス比較分析
#
# ## 1. 分析の目的
#
# 2024年以降、市場の注目を集める半導体銘柄であるARMとNVIDIAについて、どちらがより優れた投資対象であったかを多角的に評価する。
#
# この分析では、以下の3つの主要な観点から両銘柄を比較する。
# 1.  **パフォーマンス（リターン）:** どれだけのリターンを生み出したか？
# 2.  **リスク（ボラティリティ）:** そのリターンを得る過程で、どれだけの価格変動があったか？
# 3.  **相関関係:** 両銘柄の値動きには、どのような関連性があるか？
#
# ## 2. 分析の結論（エグゼクティブ・サマリー）
#
# 分析の結果、この期間においては、**NVIDIAがARMよりも優れた投資対象であった**と結論づける。
#
# NVIDIAは、ARMを上回るリターンを達成しただけでなく、その過程でのリスク（価格変動の激しさ）はARMよりも著しく低かった。つまり、**より低いリスクで、より高いリターンを効率的に達成した**と言える。両銘柄には中程度の正の相関が見られ、同じセクターのテーマで動く傾向があることも確認された。
#%% [markdown]
# ## 3. データ準備
#
# 分析の第一歩として、`yfinance`ライブラリを使用し、指定された期間（2024年1月1日以降）のARMとNVIDIAの公式な日次株価データを取得する。
#
# ここでは、企業の配当や株式分割の影響を考慮しない、純粋な市場価格である「終値（Close）」を使用し、分析の土台となるデータフレームを構築する。
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

#%% [markdown]
# ## 4. パフォーマンス分析：リターン倍率による直接比較
#
# 投資家にとって最も重要なのは「投資した資金が何倍になったか」である。そこで、両銘柄の価格水準の違いを排除し、純粋なパフォーマンスを比較するために**「正規化リターン」**を計算する。
#
# これは、期間の開始日を「1」として、その後の株価が何倍になったかを示す指標である。
#
# **【グラフの読み解き】**
# 以下のグラフは、この期間のパフォーマンス競争の物語を一目で語っている。最終的にNVIDIA（オレンジ線）は初期投資の約3.27倍に達し、ARM（青線）の約2.24倍を大きく引き離した。これは、この期間においてはNVIDIAに投資した方が、より高いリターンを得られたことを明確に示している。
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

#%% [markdown]
# ## 5. リスク分析：リターンの裏側にある変動性
#
# 高いリターンは魅力的だが、そのリターンが安定的なのか、それとも激しい価格変動の結果なのかを評価することは極めて重要である。ここでは、**日次リターンの標準偏差（ボラティリティ）**をリスクの指標として用いる。
#
# **【統計量とヒストグラムの読み解き】**
# - **標準偏差(std):** ARM(`0.052`)はNVIDIA(`0.035`)よりも著しく高い。これは、ARMの日々の価格変動がNVIDIAよりも激しい、つまり**ハイリスクな銘柄**であることを示している。
# - **ヒストグラム:** NVIDIA（オレンジ）の分布が中央に高く集まっているのに対し、ARM（青）の分布は横に広く散らばっている。これも、ARMのリターンがより不安定であることを視覚的に裏付けている。
#
# **結論として、ARMは「ハイリスク・ハイリターン」な暴れ馬、NVIDIAはより安定した「ローリスク・ハイリターン」な優等生、という性格の違いが見て取れる。**
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

#%% [markdown]
# ## 6. 相関分析：二人の関係性
#
# 次に、2つの銘柄がどの程度、同じ方向に動く傾向があるのか（＝相関）を分析する。これは、ポートフォリオを組む際の分散投資効果を考える上で重要な指標となる。
#
# **【相関係数と散布図の読み解き】**
# - **相関係数:** `0.54`という値は、**「中程度の正の相関」**があることを示す。つまり、「ARMが上がる日はNVIDIAも上がる傾向があるが、その関係は完璧ではなく、独立した動きをすることもある」という状態である。
# - **散布図:** 点の集まりが全体として右肩上がりに分布していることが、この正の相関を視覚的に裏付けている。
#
# 両者は同じ半導体・AIセクターのテーマで動く仲間でありながらも、完全に一心同体というわけではない、という複雑な関係性が明らかになった。
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


#%% [markdown]
# ## 7. 総合評価と結論
#
# 最後に、これまでの分析結果を定量的なサマリーとしてまとめる。特に、**リスク１単位あたりどれだけのリターンを得られたかを示す「シャープレシオ」**（年率リターン / 年率ボラティリティ）を計算することで、総合的な投資効率を評価する。
#
# **【結果サマリーの読み解き】**
# - **年率リターン:** NVIDIA(95.22%) > ARM(86.84%)
# - **年率リスク:** NVIDIA(56.00%) < ARM(83.30%)
# - **シャープレシオ（概算）:** NVIDIA(約1.70) > ARM(約1.04)
#
# **最終結論として、NVIDIAはARMを上回るリターンを、より低いリスクで達成しており、投資効率（シャープレシオ）の観点からも、この期間における極めて優れた投資対象であったと判断できる。**
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


# %%
