# Quant

このリポジトリは、金融市場のデータ分析と戦略構築を目的としたPythonベースのプロジェクトです。機械学習を用いた価格予測、ポートフォリオ最適化、テクニカルバックテストなど、様々なクオンツ分析ツールが含まれています。

## ディレクトリ構造

-   `notebooks/`: Jupyter Notebook (.ipynb) ファイルが含まれています。これらはデータ探索、モデル開発、分析結果の可視化に使用されます。
-   `scripts/`: Python (.py) スクリプトが含まれています。これらはクリーンなコード、自動化されたタスク、および本番環境での利用を目的としたユーティリティです。
-   `venv/`: Pythonの仮想環境。プロジェクトの依存関係を管理するために使用されます。
-   `README.md`: プロジェクトの概要と説明です。

## 使い方

プロジェクトをセットアップして実行するには、以下の手順に従ってください。

1.  **リポジトリをクローンする**:
    ```bash
    git clone https://github.com/yourusername/Quant.git
    cd Quant
    ```

2.  **仮想環境をセットアップする**:
    Pythonの仮想環境を作成し、必要な依存関係をインストールすることを強く推奨します。
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # .\venv\Scripts\activate # Windows
    pip install -r requirements.txt # requirements.txtが存在しない場合は作成してください
    ```
    (まだ`requirements.txt`がない場合は、後で作成を提案できます。)

3.  **Jupyter Notebookの実行**:
    `notebooks/`ディレクトリ内のJupyter Notebookを開いて分析を実行できます。
    ```bash
    jupyter notebook notebooks/
    ```

4.  **Pythonスクリプトの実行**:
    `scripts/`ディレクトリ内のPythonスクリプトは直接実行できます。
    ```bash
    python scripts/your_script_name.py
    ```

## 例

-   `notebooks/PriceForcast.ipynb`: 機械学習を用いた価格予測モデルの開発例。
-   `scripts/PortfolioOptimization.py`: ポートフォリオの最適化を行うスクリプトの例。
