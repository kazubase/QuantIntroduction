import matplotlib
import shutil
import os

# matplotlibのキャッシュディレクトリのパスを取得
cache_dir = matplotlib.get_cachedir()

# キャッシュディレクトリが存在する場合
if os.path.exists(cache_dir):
    try:
        # ディレクトリごと削除
        shutil.rmtree(cache_dir)
        print(f"Matplotlibのキャッシュを削除しました: {cache_dir}")
    except Exception as e:
        print(f"キャッシュの削除中にエラーが発生しました: {e}")
else:
    print("キャッシュディレクトリが見つかりませんでした。")

print("キャッシュを削除しました。プログラムを再起動して、再度グラフを描画してください。")