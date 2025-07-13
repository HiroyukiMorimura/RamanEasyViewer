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
from peak_analysis import peak_analysis_mode
from peak_deconvolution import peak_deconvolution_mode
from multivariate_analysis import multivariate_analysis_mode
from peak_ai_analysis import peak_ai_analysis_mode

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
        ["スペクトル解析", "ラマンピークファインダー", "ラマンピーク分離", "多変量解析", "ピークAI解析"],
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