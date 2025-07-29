# -*- coding: utf-8 -*-
"""
統合ラマンスペクトル解析ツール
メインスクリプト

Created on Wed Jun 11 15:56:04 2025
@author: hiroy

Enhanced Integrated Raman Spectrum Analysis Tool with Peak Analysis and Q&A Feature
"""

import streamlit as st

# 各モジュールのインポート
from spectrum_analysis import spectrum_analysis_mode
from peak_analysis_web import peak_analysis_mode
from peak_deconvolution import peak_deconvolution_mode
from multivariate_analysis import multivariate_analysis_mode
from peak_ai_analysis import peak_ai_analysis_mode
from calibration_mode import calibration_mode
from raman_database import database_comparison_mode

def main():
    """
    メイン関数
    """
    # ページ設定
    st.set_page_config(
        page_title="RamanEye Easy Viewer", 
        page_icon="📊", 
        layout="wide"
    )
    
    # タイトル
    st.markdown(
    "<h1>📊 <span style='font-style: italic;'>RamanEye</span> Easy Viewer</h1>",
    unsafe_allow_html=True
)
    # サイドバーにモード選択を配置
    st.sidebar.header("🔧 解析モード選択")
    analysis_mode = st.sidebar.selectbox(
        "解析モードを選択してください:",
        ["スペクトル解析", "ラマンピークファインダー", "ラマンピーク分離", "多変量解析",  "検量線作成", "ピークAI解析", "データベース比較"],
        key="mode_selector"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📋 パラメータ設定")
    
    # 選択されたモードに応じて適切な関数を呼び出す
    if analysis_mode == "スペクトル解析":
        spectrum_analysis_mode()
    elif analysis_mode == "多変量解析":
        multivariate_analysis_mode()
    elif analysis_mode == "ラマンピーク分離":
        peak_deconvolution_mode()
    elif analysis_mode == "ラマンピークファインダー":
        peak_analysis_mode()
    elif analysis_mode == "検量線作成":
        calibration_mode()
    elif analysis_mode == "データベース比較":
        database_comparison_mode()
    else:  # ピークAI解析
        peak_ai_analysis_mode()
    
    # 使用方法の説明
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 使用方法")
    
    if analysis_mode == "スペクトル解析":
        st.sidebar.markdown("""
        **スペクトル解析モード:**
        1. 解析したいCSVファイルをアップロード
        2. パラメータを調整
        3. スペクトルの表示と解析結果を確認
        4. 結果をCSVファイルでダウンロード
        """)
    elif analysis_mode == "多変量解析":
        st.sidebar.markdown("""
        **多変量解析モード:**
        1. 複数のCSVファイルをアップロード
        2. パラメータを調整
        3. 「データプロセス実効」をクリック
        4. 「多変量解析実効」をクリック
        5. 解析結果を確認・ダウンロード
        
        - コンポーネント数: 2-5
        """)
    elif analysis_mode == "ラマンピーク分離":
        st.sidebar.markdown("""
        **ピーク分離モード:**
        1. 解析したいCSVファイルをアップロード
        2. パラメータを調整
        3. フィッテング範囲を設定
        4. ピーク数最適化によりピーク数決定（n=1～6）
        5. 必要であれば波数固定
        6. フィッティングを実効
        
        """)
    elif analysis_mode == "ラマンピークファインダー":
        st.sidebar.markdown("""
        **ラマンピーク解析モード:**
        1. 解析したいCSVファイルをアップロード
        2. パラメータを調整
        3. 「ピーク検出を実行」をクリック
        4. インタラクティブプロットでピークを調整
        5. 手動ピークの追加・除外が可能
        6. グリッドサーチで最適化
        7. 結果をCSVファイルでダウンロード
        
        **インタラクティブ機能:**
        - グラフをクリックして手動ピーク追加
        - 自動検出ピークをクリックして除外
        - グリッドサーチで閾値最適化
        """)
    elif analysis_mode == "検量線作成":
        st.sidebar.markdown("""
        **検量線作成モード:**
        1. **複数ファイルアップロード**: 異なる濃度のスペクトルファイルをアップロード
        2. **濃度データ入力**: 各サンプルの濃度を入力
        3. **検量線タイプ選択**: ピーク面積またはPLS回帰を選択
        4. **波数範囲設定**: 解析に使用する波数範囲を指定
        5. **検量線作成実行**: 統計解析により検量線を作成
        6. **結果確認**: R²、RMSE等の統計指標を確認
        7. **結果エクスポート**: 検量線データをCSVでダウンロード
        
        **ピーク面積モード:**
        - 指定範囲の単一ピークをローレンツ関数でフィッティング
        - ピーク面積と濃度の線形関係を構築
        - ピーク中心波数の固定が可能
        
        **PLS回帰モード:**
        - 指定波数範囲の全スペクトルデータを使用
        - 部分最小二乗回帰による多変量解析
        - クロスバリデーションによる予測精度評価
        - 成分数の最適化が可能
        
        **統計指標:**
        - R²（決定係数）
        - RMSE（平均二乗誤差の平方根）
        - クロスバリデーションR²（PLS回帰のみ）
        """)
    else:  # ピークAI解析
        st.sidebar.markdown("""
        **ピークAI解析モード:**
        1. **LLM設定**: APIキーを入力するかオフラインモデルを起動
        2. **論文アップロード**: RAG機能用の論文PDFをアップロード
        3. **データベース構築**: 論文から検索用データベースを作成
        4. **スペクトルアップロード**: 解析するラマンスペクトルをアップロード
        5. **ピーク検出**: 自動検出 + 手動調整でピークを確定
        6. **AI解析実行**: 確定ピークを基にAIが考察を生成
        7. **質問機能**: 解析結果について追加質問が可能
        
        **サポートファイル形式:**
        - **スペクトル**: CSV, TXT (RamanEye, Wasatch, Eagle対応)
        - **論文**: PDF, DOCX, TXT
        
        **AI解析について:**
        - 論文データベースと組み合わせて高精度な解析を実現
        - 解析後に追加質問で詳細な情報を取得可能
        """)
    
    # フッター情報
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **バージョン情報:**
    - Version: 2.0.0
    - Last Updated: 2025-07-13
    - Author: Hiroyuki Moirmura
    """)

if __name__ == "__main__":
    main()
