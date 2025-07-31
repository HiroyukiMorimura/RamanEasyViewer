# -*- coding: utf-8 -*-
"""
統合ラマンスペクトル解析ツール（メインスクリプト）
RamanEye Easy Viewer (Main Script)

Created on Wed Jun 11 15:56:04 2025
@author: Hiroyuki Morimura
"""

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# 既存の解析モジュールのインポート
try:
    from spectrum_analysis import spectrum_analysis_mode
    from peak_analysis_web import peak_analysis_mode
    from peak_deconvolution import peak_deconvolution_mode
    from multivariate_analysis import multivariate_analysis_mode
    from calibration_mode import calibration_mode
    from raman_database import database_comparison_mode
    from peak_ai_analysis import peak_ai_analysis_mode
    MODULES_AVAILABLE = True
    
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"解析モジュールのインポートエラー: {e}")

class SimpleAuthManager:
    """シンプルな認証マネージャー"""
    
    def __init__(self):
        # デモアカウント
        self.demo_accounts = {
            "admin": {"password": "Admin123!", "role": "管理者"},
            "analyst": {"password": "Analyst123!", "role": "分析者"},
            "viewer": {"password": "Viewer123!", "role": "閲覧者"}
        }
    
    def login(self, username: str, password: str) -> tuple:
        """ログイン処理"""
        if username in self.demo_accounts:
            if self.demo_accounts[username]["password"] == password:
                st.session_state.authenticated = True
                st.session_state.current_user = {
                    "username": username,
                    "role": self.demo_accounts[username]["role"]
                }
                st.session_state.login_time = datetime.now()
                return True, "ログイン成功"
            else:
                return False, "パスワードが間違っています"
        else:
            return False, "ユーザーが見つかりません"
    
    def logout(self):
        """ログアウト処理"""
        st.session_state.authenticated = False
        st.session_state.current_user = None
        st.session_state.login_time = None
    
    def is_authenticated(self) -> bool:
        """認証状態の確認"""
        return st.session_state.get('authenticated', False)

class RamanEyeApp:
    """シンプル版メインアプリケーションクラス（ログイン機能付き）"""
    
    def __init__(self):
        # ページ設定
        st.set_page_config(
            page_title="RamanEye Easy Viewer",
            page_icon="favicon.png",  # 同フォルダ内のPNGをそのまま指定
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 認証マネージャーの初期化
        self.auth_manager = SimpleAuthManager()
        
        # セッション状態の初期化
        self._init_session_state()
    
    def _init_session_state(self):
        """セッション状態の初期化"""
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "current_user" not in st.session_state:
            st.session_state.current_user = None
        if "login_time" not in st.session_state:
            st.session_state.login_time = None
        if "analysis_mode" not in st.session_state:
            st.session_state.analysis_mode = "スペクトル解析"
    
    def run(self):
        """メインアプリケーションの実行"""
        try:
            if not self.auth_manager.is_authenticated():
                self._render_login_page()
            else:
                self._render_main_application()
        except Exception as e:
            st.error(f"アプリケーション実行中にエラーが発生しました: {e}")
    
    def _render_login_page(self):
        """ログインページの表示"""
        # CSSスタイル
        st.markdown(
            """
            <style>
            .login-header {
                color: #1f77b4;
                margin-bottom: 0.5rem !important;
                font-size: 1.8rem !important;
                font-weight: bold;
            }
            .logo-container {
                text-align: center;
                padding: 2rem;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                margin-bottom: 1rem;
            }
            .security-info {
                background-color: #e8f5e8;
                border-left: 4px solid #28a745;
                padding: 1rem;
                margin: 1rem 0;
                border-radius: 5px;
            }
            .feature-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 1.5rem;
                border-radius: 10px;
                text-align: center;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 1.5rem;
                min-height: 180px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            }
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # ロゴとログインフォームを1:1で配置
        col_logo, col_login = st.columns([1, 1])
        
        with col_logo:
            # logo.jpg表示
            self._display_logo_image()
        
        with col_login:
            # ログインフォーム
            
            # ログインフォーム
            with st.form("login_form"):
                st.markdown('<h2 class="login-header"><em>RamanEye</em> Easy Viwer ログイン</h2>', unsafe_allow_html=True)
                username = st.text_input(
                    "ユーザー名", 
                    placeholder="ユーザー名を入力"
                )
                password = st.text_input(
                    "パスワード", 
                    type="password", 
                    placeholder="パスワードを入力"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    login_button = st.form_submit_button("🔐 ログイン", type="primary", use_container_width=True)
                with col2:
                    forgot_password = st.form_submit_button("パスワード忘れ", use_container_width=True)
            
            # ログイン処理
            if login_button:
                self._process_login(username, password)
            
            # パスワードリセット
            if forgot_password:
                st.info("パスワードリセットについては管理者にお問い合わせください")
        
        # セキュリティ機能表示（折りたたみ）
        self._render_security_features_collapsible()
        
        # デモアカウント情報
        self._render_demo_accounts()
        
        # 主要機能の表示
        self._render_features()
        
        # フッター
        self._render_footer()
    
    def _display_logo_image(self):
        try:
            st.image("logo.png", use_container_width = True)
        except Exception as e:
            # logo.jpgが見つからない場合のフォールバック
            st.markdown(
                """
                <div style="
                    text-align: center;
                    padding: 3rem 2rem;
                    background: linear-gradient(135deg, #1f77b4 0%, #17a2b8 100%);
                    color: white;
                    border-radius: 12px;
                    font-size: 1.5rem;
                    font-weight: bold;
                    margin: 1rem 0;
                ">
                    📊 RamanEye<br>
                    <small style="font-size: 0.8rem;">Logo placeholder</small>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.info("logo.jpgファイルが見つからないため、プレースホルダーを表示しています。")

    
    def _render_security_features_collapsible(self):
        """セキュリティ機能の折りたたみ表示"""
        st.markdown("---")
        
        with st.expander("🔒 セキュリティ機能の詳細", expanded=False):
            st.markdown("**基本セキュリティ機能:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("🔐 ログイン認証: ✅")
                st.write("👤 ユーザー管理: ✅")
                st.write("🕐 セッション管理: ✅")
            
            with col2:
                st.write("📝 基本ログ記録: ✅")
                st.write("🔄 自動ログアウト: ✅")
                st.write("🛡️ 基本保護: ✅")
    
    def _process_login(self, username: str, password: str):
        """ログイン処理"""
        if not username or not password:
            st.error("ユーザー名とパスワードを入力してください")
            return
        
        success, message = self.auth_manager.login(username, password)
        
        if success:
            st.success("ログインが完了しました")
            st.rerun()
        else:
            st.error(f"ログインに失敗しました: {message}")
    
    def _render_demo_accounts(self):
        """デモアカウント情報の表示"""
        st.markdown("---")
        
        with st.expander("🔧 デモアカウント情報", expanded=False):
            st.info("学習・評価目的のデモアカウントです")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **👑 管理者**
                - ユーザー名: `admin`
                - パスワード: `Admin123!`
                - 全機能利用可能
                """)
            
            with col2:
                st.markdown("""
                **🔬 分析者**
                - ユーザー名: `analyst`
                - パスワード: `Analyst123!`
                - 分析機能利用可能
                """)
            
            with col3:
                st.markdown("""
                **👁️ 閲覧者**
                - ユーザー名: `viewer`
                - パスワード: `Viewer123!`
                - 基本機能のみ
                """)
    
    def _render_features(self):
        """主要機能の表示"""
        st.markdown("### 🌟 主要機能一覧")
        
        features = [
            ("📊", "スペクトル解析", "ラマンスペクトルの基本解析機能"),
            ("🔍", "ピーク分析", "ピーク検出・解析機能"),
            ("⚗️", "ピーク分離", "重複ピークの分離機能"),
            ("📈", "多変量解析", "統計解析・データマイニング"),
            ("📏", "検量線作成", "定量分析用検量線作成"),
            ("🤖", "AI解析", "AI支援解析・RAG機能"),
            ("🗄️", "DB比較", "データベース照合機能"),
            ("🔧", "ユーザー管理", "アカウント・権限管理")
        ]
        
        # 2行4列のグリッド
        for row in range(2):
            cols = st.columns(4)
            for col_idx in range(4):
                feature_idx = row * 4 + col_idx
                if feature_idx < len(features):
                    icon, title, desc = features[feature_idx]
                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="feature-card">
                                <div class="feature-icon">{icon}</div>
                                <h4 style="margin: 0.5rem 0;">{title}</h4>
                                <p style="font-size: 0.85rem; margin: 0; line-height: 1.3;">{desc}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
    
    def _render_footer(self):
        """フッター"""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>📊 <strong>RamanEye Easy Viewer v2.0.0</strong> - Simple Edition</p>
            <p>© 2025 Hiroyuki Morimura. All rights reserved.</p>
            <p style="font-size: 0.8rem; color: #999;">
                Integrated Raman Spectrum Analysis Tool
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_main_application(self):
        """メインアプリケーションの表示"""
        # 認証後ヘッダー
        self._render_authenticated_header()
        
        # 会社ロゴの表示
        self._display_company_logo()
        
        # メインタイトル
        st.markdown(
            "<h1>📊 <span style='font-style: italic;'>RamanEye</span> Easy Viewer</h1>",
            unsafe_allow_html=True
        )
        
        # サイドバー設定
        self._render_sidebar()
        
        # メインコンテンツエリア
        if not MODULES_AVAILABLE:
            st.error("解析モジュールが利用できません。管理者にお問い合わせください。")
            return
        
        # 解析モード実行
        self._execute_analysis_mode()
    
    def _render_authenticated_header(self):
        """認証後ヘッダー"""
        current_user = st.session_state.get('current_user', {})
        login_time = st.session_state.get('login_time')
        
        with st.sidebar:
            st.markdown("### 👤 ユーザー情報")
            st.info(f"ユーザー: {current_user.get('username', 'Unknown')}")
            st.info(f"役割: {current_user.get('role', 'Unknown')}")
            if login_time:
                st.info(f"ログイン時刻: {login_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if st.button("🚪 ログアウト", use_container_width=True):
                self.auth_manager.logout()
                st.rerun()
    
    def _display_company_logo(self):
        """会社ロゴ表示"""
        st.markdown(
            """
            <div style="text-align: center; margin: 2rem 0;">
                <div style="
                    background: linear-gradient(135deg, #1f77b4 0%, #17a2b8 100%);
                    color: white;
                    padding: 2rem 3rem;
                    border-radius: 15px;
                    font-size: 2.5rem;
                    font-weight: bold;
                    display: inline-block;
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
                    margin-bottom: 1rem;
                ">
                    📊 RamanEye
                </div>
                <div style="font-size: 1.2rem; color: #666; text-align: center; margin: 0;">
                    Easy Viewer
                </div>
                <div style="font-size: 0.9rem; color: #1f77b4; text-align: center; margin-top: 0.5rem;">
                    統合ラマンスペクトル解析ツール
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_sidebar(self):
        """サイドバーの表示"""
        st.sidebar.header("🔧 解析モード選択")
        
        # 利用可能モード
        available_modes = [
            "スペクトル解析",
            "データベース比較",
            "ピークファインダー",
            "ピーク分離",
            "多変量解析",
            "検量線作成",
            "ピークAI解析"
        ]
        
        # モード選択
        analysis_mode = st.sidebar.selectbox(
            "解析モードを選択してください:",
            available_modes,
            index=0,
            key="mode_selector"
        )
        
        st.session_state.analysis_mode = analysis_mode
        
        # 使用方法の説明
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 使用方法")
        self._render_usage_instructions(analysis_mode)
        
        # フッター情報
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **アプリケーション情報:**
        - Version: 2.0.0 Simple
        - Author: Hiroyuki Morimura
        - Last Updated: 2025-07-31
        """)
    
    def _render_usage_instructions(self, analysis_mode):
        """使用方法の説明"""
        instructions = {
            "スペクトル解析": """
            **📊 スペクトル解析:**
            1. CSVファイルをアップロード
            2. 解析パラメータを調整
            3. スペクトル解析を実行
            4. 結果をダウンロード
            """,
            
            "ピークAI解析": """
            **🤖 ピークAI解析:**
            1. API接続を設定
            2. 論文データベースを構築
            3. AI解析を実行
            4. 結果を確認・ダウンロード
            """,
            
            "データベース比較": """
            **🗄️ データベース比較:**
            1. 測定データをアップロード
            2. データベースを選択
            3. 比較解析を実行
            4. マッチング結果を確認
            """,
            
            "多変量解析": """
            **📈 多変量解析:**
            1. 複数のスペクトルデータを準備
            2. 前処理パラメータを設定
            3. 統計解析を実行
            4. 解析結果を可視化
            """,
            
            "ピーク分離": """
            **⚗️ ピーク分離:**
            1. スペクトルデータをアップロード
            2. ピーク分離パラメータを設定
            3. フィッティングを実行
            4. 分離結果を確認
            """,
            
            "ピークファインダー": """
            **🔍 ピークファインダー:**
            1. スペクトルデータをアップロード
            2. ピーク検出パラメータを設定
            3. ピーク検出を実行
            4. 検出結果を確認
            """,
            
            "検量線作成": """
            **📏 検量線作成:**
            1. 標準試料のデータをアップロード
            2. 濃度情報を入力
            3. 検量線を作成
            4. 未知試料を定量分析
            """
        }
        
        instruction = instructions.get(analysis_mode, "使用方法情報なし")
        st.sidebar.markdown(instruction)
    
    def _execute_analysis_mode(self):
        """解析モードの実行"""
        analysis_mode = st.session_state.analysis_mode
        
        try:
            if analysis_mode == "スペクトル解析":
                spectrum_analysis_mode()
            elif analysis_mode == "データベース比較":
                database_comparison_mode()
            elif analysis_mode == "多変量解析":
                multivariate_analysis_mode()
            elif analysis_mode == "ピーク分離":
                peak_deconvolution_mode()
            elif analysis_mode == "ピークファインダー":
                peak_analysis_mode()
            elif analysis_mode == "検量線作成":
                calibration_mode()
            elif analysis_mode == "ピークAI解析":
                peak_ai_analysis_mode()
            else:
                spectrum_analysis_mode()
                
        except Exception as e:
            st.error(f"解析モード '{analysis_mode}' の実行中にエラーが発生しました。")
            st.error(f"エラー詳細: {e}")

def main():
    """メイン関数"""
    try:
        app = RamanEyeApp()
        app.run()
    except Exception as e:
        st.error("アプリケーションの初期化中にエラーが発生しました")
        st.error(f"エラー詳細: {e}")
        
        # 再起動オプション
        st.markdown("---")
        st.info("アプリケーションを再起動しますか？")
        if st.button("🔄 再起動"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
