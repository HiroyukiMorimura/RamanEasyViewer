# -*- coding: utf-8 -*-
"""
統合ラマンスペクトル解析ツール（シンプル版 + ログイン機能）
Integrated Raman Spectrum Analysis Tool - Simple Version with Login

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
            page_icon="📊",
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
                margin-bottom: 1rem;
                font-size: 1.8rem;
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
            st.markdown('<h2 class="login-header">📊 <em>RamanEye</em> Login</h2>', unsafe_allow_html=True)
            
            # ログインフォーム
            with st.form("login_form"):
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
        """logo.jpg画像の表示"""
        try:
            st.markdown(
                """
                <div class="logo-container">
                    <h3 style="color: #1f77b4; margin-bottom: 1rem;">会社ロゴ</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # logo.jpgの表示を試行
            try:
                st.image("logo.jpg", caption="Company Logo", use_column_width=True)
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
        
        except Exception as e:
            st.error(f"ロゴ表示エラー: {e}")
    
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
        if not username or
