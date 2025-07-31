# -*- coding: utf-8 -*-
"""
統合ラマンスペクトル解析ツール（認証機能付き）
メインスクリプト

Created on Wed Jun 11 15:56:04 2025
@author: hiroy

Enhanced Integrated Raman Spectrum Analysis Tool with Authentication System
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# 認証システムのインポート
from auth_system import (
    AuthenticationManager, 
    UserRole, 
    require_auth, 
    require_permission,
    require_role
)
from user_management_ui import (
    LoginUI, 
    UserManagementUI, 
    ProfileUI, 
    render_authenticated_header
)

# 既存の解析モジュールのインポート（権限チェック付きでラップ）
try:
    from spectrum_analysis import spectrum_analysis_mode
    from peak_analysis_web import peak_analysis_mode
    from peak_deconvolution import peak_deconvolution_mode
    from multivariate_analysis import multivariate_analysis_mode
    from peak_ai_analysis import peak_ai_analysis_mode
    from calibration_mode import calibration_mode
    from raman_database import database_comparison_mode
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"解析モジュールのインポートエラー: {e}")

class RamanEyeApp:
    """メインアプリケーションクラス"""
    
    def __init__(self):
        self.auth_manager = AuthenticationManager()
        self.login_ui = LoginUI()
        self.user_management_ui = UserManagementUI()
        self.profile_ui = ProfileUI()
        
        # ページ設定
        st.set_page_config(
            page_title="RamanEye Easy Viewer - Secure", 
            page_icon="🔐", 
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # セッション状態の初期化
        if "show_profile" not in st.session_state:
            st.session_state.show_profile = False
        if "show_user_management" not in st.session_state:
            st.session_state.show_user_management = False
    
    def run(self):
        """メインアプリケーションの実行"""
        # 認証チェック
        if not self.auth_manager.is_authenticated():
            self._render_login_page()
        else:
            # セッションタイムアウトチェック
            if not self.auth_manager.check_session_timeout(timeout_minutes=60):
                st.error("セッションがタイムアウトしました。再度ログインしてください")
                st.stop()
            
            self._render_main_application()
    
    def _display_company_logo(self):
        """会社ロゴを表示"""
        import os
        from PIL import Image
        
        # ロゴファイルのパスを複数チェック
        logo_paths = [
            "logo.jpg",          # 同じフォルダ内
            "logo.png",          # PNG形式も対応
            "assets/logo.jpg",   # assetsフォルダ内
            "assets/logo.png",   # assetsフォルダ内（PNG）
            "images/logo.jpg",   # imagesフォルダ内
            "images/logo.png"    # imagesフォルダ内（PNG）
        ]
        
        logo_displayed = False
        
        # ローカルファイルをチェック
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                try:
                    image = Image.open(logo_path)
                    
                    # ロゴを中央に配置（幅を調整）
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(
                            image, 
                            width=300,  # ロゴの幅を調整
                            caption="",
                            use_column_width=False
                        )
                    
                    logo_displayed = True
                    break
                    
                except Exception as e:
                    st.error(f"ロゴファイルの読み込みエラー ({logo_path}): {str(e)}")
        
        # ローカルファイルが見つからない場合、GitHubからの読み込みを試行
        if not logo_displayed:
            github_logo_urls = [
                "https://raw.githubusercontent.com/yourusername/yourrepository/main/logo.jpg",
                "https://raw.githubusercontent.com/yourusername/yourrepository/main/logo.png",
                "https://raw.githubusercontent.com/yourusername/yourrepository/main/assets/logo.jpg",
                "https://raw.githubusercontent.com/yourusername/yourrepository/main/assets/logo.png"
            ]
            
            for url in github_logo_urls:
                try:
                    # GitHubからの画像読み込み
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(
                            url,
                            width=300,
                            caption="",
                            use_column_width=False
                        )
                    
                    logo_displayed = True
                    break
                    
                except Exception:
                    continue
        
        # ロゴが見つからない場合のフォールバック
        if not logo_displayed:
            # テキストベースのロゴを表示
            st.markdown(
                """
                <div style="text-align: center; margin: 1rem 0;">
                    <div style="
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 1rem 2rem;
                        border-radius: 10px;
                        font-size: 1.5rem;
                        font-weight: bold;
                        display: inline-block;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        🏢 Your Company Name
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # ロゴファイルの配置に関する情報を表示（開発用）
            with st.expander("ℹ️ ロゴファイルの配置について"):
                st.info("""
                **ロゴを表示するには、以下のいずれかの場所にlogo.jpgまたはlogo.pngを配置してください:**
                
                📁 **同じフォルダ内**:
                - `logo.jpg` または `logo.png`
                
                📁 **サブフォルダ内**:
                - `assets/logo.jpg` または `assets/logo.png`
                - `images/logo.jpg` または `images/logo.png`
                
                🌐 **GitHub Repository**:
                - GitHubのraw URLを使用する場合は、`_display_company_logo()`メソッド内のURLを実際のリポジトリURLに変更してください
                
                **サポート形式**: JPG, PNG
                **推奨サイズ**: 300px幅程度
                """)
    
    def _render_login_page(self):
        """ログインページの表示"""
        # カスタムCSS
        st.markdown(
            """
            <style>
            .main-header {
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
                font-size: 3rem;
                font-weight: bold;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 3rem;
                font-size: 1.2rem;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
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
        
        # 会社ロゴの表示
        self._display_company_logo()
        
        # ヘッダー
        st.markdown(
            '<h1 class="main-header">🔐 RamanEye Easy Viewer</h1>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p class="subtitle">Secure Raman Spectrum Analysis Platform</p>',
            unsafe_allow_html=True
        )
        
        # 機能紹介
        st.markdown("### 🌟 主要機能")
        
        features = [
            ("📊", "スペクトル解析", "ラマンスペクトルの基本解析・可視化"),
            ("🔍", "ピーク分析", "自動ピーク検出・解析・最適化"),
            ("⚗️", "ピーク分離", "複雑なピークの分離・フィッティング"),
            ("📈", "多変量解析", "PCA・クラスター分析等の統計解析"),
            ("📏", "検量線作成", "定量分析用検量線の作成・評価"),
            ("🤖", "AI解析", "機械学習によるスペクトル解釈"),
            ("🗄️", "データベース比較", "スペクトルライブラリとの照合"),
            ("🔒", "セキュリティ", "ユーザー管理・権限制御・監査機能")
        ]
        
        # 2行4列のグリッドで機能を表示（重なりを防ぐ）
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
        
        st.markdown("---")
        
        # ログインフォーム
        self.login_ui.render_login_page()
        
        # フッター
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>🔬 <strong>RamanEye Easy Viewer v2.0.0</strong> - Secure Edition</p>
            <p>Advanced Raman Spectrum Analysis with Enterprise Security</p>
            <p>© 2025 Hiroyuki Morimura. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_main_application(self):
        """メインアプリケーションの表示"""
        # 認証後ヘッダー
        render_authenticated_header()
        
        # 会社ロゴの表示
        self._display_company_logo()
        
        # プロファイル表示チェック
        if st.session_state.get("show_profile", False):
            self.profile_ui.render_profile_page()
            if st.button("⬅️ メインメニューに戻る"):
                st.session_state.show_profile = False
                st.rerun()
            return
        
        # ユーザー管理表示チェック
        if st.session_state.get("show_user_management", False):
            self.user_management_ui.render_user_management_page()
            if st.button("⬅️ メインメニューに戻る"):
                st.session_state.show_user_management = False
                st.rerun()
            return
        
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
        
        # 選択されたモードに応じて適切な関数を呼び出す
        analysis_mode = st.session_state.get("mode_selector", "スペクトル解析")
        
        try:
            if analysis_mode == "スペクトル解析":
                self._render_spectrum_analysis()
            elif analysis_mode == "データベース比較":
                self._render_database_comparison()
            elif analysis_mode == "多変量解析":
                self._render_multivariate_analysis()
            elif analysis_mode == "ラマンピーク分離":
                self._render_peak_deconvolution()
            elif analysis_mode == "ラマンピークファインダー":
                self._render_peak_analysis()
            elif analysis_mode == "検量線作成":
                self._render_calibration()
            elif analysis_mode == "ピークAI解析":
                self._render_peak_ai_analysis()
            elif analysis_mode == "ユーザー管理":
                st.session_state.show_user_management = True
                st.rerun()
            else:
                # デフォルトはスペクトル解析
                self._render_spectrum_analysis()
        except Exception as e:
            st.error(f"機能の実行中にエラーが発生しました: {e}")
            st.error("管理者にお問い合わせください。")
    
    def _render_sidebar(self):
        """サイドバーの設定"""
        st.sidebar.header("🔧 解析モード選択")
        
        # 現在のユーザーの権限を取得
        current_role = self.auth_manager.get_current_role()
        permissions = UserRole.get_role_permissions(current_role)
        
        # 利用可能なモードを権限に基づいて決定（スペクトル解析を最初に配置）
        available_modes = []
        mode_permissions = {
            "スペクトル解析": "spectrum_analysis",           # 全ユーザー利用可能
            "データベース比較": "database_comparison",       # 全ユーザー利用可能  
            "ラマンピークファインダー": "peak_analysis", 
            "ラマンピーク分離": "peak_deconvolution",
            "多変量解析": "multivariate_analysis",
            "検量線作成": "calibration",
            "ピークAI解析": "peak_ai_analysis"
        }
        
        # 全ユーザーが使用可能な機能を最初に追加
        for mode, permission in mode_permissions.items():
            if permissions.get(permission, False):
                available_modes.append(mode)
        
        # 管理者はユーザー管理も利用可能（最後に追加）
        if permissions.get("user_management", False):
            available_modes.append("ユーザー管理")
        
        # モード選択
        analysis_mode = st.sidebar.selectbox(
            "解析モードを選択してください:",
            available_modes,
            index=0,  # 常に最初の利用可能なモード（スペクトル解析）をデフォルトに
            key="mode_selector"
        )
        
        # 権限情報表示
        st.sidebar.markdown("---")
        st.sidebar.header("👤 アクセス権限")
        
        role_descriptions = {
            UserRole.ADMIN: "🔧 すべての機能にアクセス可能",
            UserRole.ANALYST: "📊 分析機能にフルアクセス可能", 
            UserRole.VIEWER: "👁️ 閲覧・基本分析のみ可能"
        }
        
        st.sidebar.info(role_descriptions.get(current_role, "権限情報なし"))
        
        # 使用方法の説明
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 使用方法")
        
        self._render_usage_instructions(analysis_mode)
        
        # フッター情報
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
        **バージョン情報:**
        - Version: 2.0.0 Secure Edition
        - Last Updated: 2025-07-31
        - Author: Hiroyuki Morimura
        - Security: Enterprise Grade
        """)
    
    def _render_usage_instructions(self, analysis_mode):
        """使用方法の説明"""
        instructions = {
            "スペクトル解析": """
            **スペクトル解析モード:**
            1. 解析したいCSVファイルをアップロード
            2. パラメータを調整
            3. スペクトルの表示と解析結果を確認
            4. 結果をCSVファイルでダウンロード
            """,
            
            "多変量解析": """
            **多変量解析モード:**
            1. 複数のCSVファイルをアップロード
            2. パラメータを調整
            3. 「データプロセス実効」をクリック
            4. 「多変量解析実効」をクリック
            5. 解析結果を確認・ダウンロード
            
            - コンポーネント数: 2-5
            """,
            
            "ラマンピーク分離": """
            **ピーク分離モード:**
            1. 解析したいCSVファイルをアップロード
            2. パラメータを調整
            3. フィッティング範囲を設定
            4. ピーク数最適化によりピーク数決定（n=1～6）
            5. 必要であれば波数固定
            6. フィッティングを実効
            """,
            
            "ラマンピークファインダー": """
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
            """,
            
            "検量線作成": """
            **検量線作成モード:**
            1. **複数ファイルアップロード**: 異なる濃度のスペクトルファイルをアップロード
            2. **濃度データ入力**: 各サンプルの濃度を入力
            3. **検量線タイプ選択**: ピーク面積またはPLS回帰を選択
            4. **波数範囲設定**: 解析に使用する波数範囲を指定
            5. **検量線作成実行**: 統計解析により検量線を作成
            6. **結果確認**: R²、RMSE等の統計指標を確認
            7. **結果エクスポート**: 検量線データをCSVでダウンロード
            """,
            
            "データベース比較": """
            **データベース比較モード:**
            1. **ファイルアップロード**: 複数のスペクトルファイル（CSV/TXT）をアップロード
            2. **前処理パラメータ設定**: ベースライン補正や波数範囲を設定
            3. **スペクトル処理**: 全ファイルを一括処理してデータベース化
            4. **比較計算**: 比較マトリックスを計算
            5. **効率化機能**: プーリングと上位N個選択で高速化
            6. **結果確認**: 統計サマリーと比較マトリックスを表示
            7. **最高一致ペア**: 最も一致したスペクトルペアを自動検出・表示
            8. **エクスポート**: 結果をCSV形式でダウンロード
            """,
            
            "ピークAI解析": """
            **ピークAI解析モード:**
            1. **LLM設定**: APIキーを入力するかオフラインモデルを起動
            2. **論文アップロード**: RAG機能用の論文PDFをアップロード
            3. **データベース構築**: 論文から検索用データベースを作成
            4. **スペクトルアップロード**: 解析するラマンスペクトルをアップロード
            5. **ピーク検出**: 自動検出 + 手動調整でピークを確定
            6. **AI解析実行**: 確定ピークを基にAIが考察を生成
            7. **質問機能**: 解析結果について追加質問が可能
            """,
            
            "ユーザー管理": """
            **ユーザー管理モード:**
            1. **ユーザー一覧**: 全ユーザーの状態確認
            2. **新規作成**: 新しいユーザーアカウントの作成
            3. **権限管理**: ロール変更・アクセス制御
            4. **アカウント管理**: ロック・解除・削除
            5. **パスワード管理**: 強制リセット・ポリシー設定
            6. **監査機能**: ログイン履歴・活動記録の確認
            
            **⚠️ 管理者専用機能**
            """
        }
        
        instruction = instructions.get(analysis_mode, "使用方法情報なし")
        st.sidebar.markdown(instruction)
    
    # 各解析モードのラッパー関数（権限チェック付き）
    @require_permission("spectrum_analysis")
    def _render_spectrum_analysis(self):
        """スペクトル解析モード（権限チェック付き）"""
        spectrum_analysis_mode()
    
    @require_permission("multivariate_analysis")
    def _render_multivariate_analysis(self):
        """多変量解析モード（権限チェック付き）"""
        multivariate_analysis_mode()
    
    @require_permission("peak_deconvolution")
    def _render_peak_deconvolution(self):
        """ピーク分離モード（権限チェック付き）"""
        peak_deconvolution_mode()
    
    @require_permission("peak_analysis")
    def _render_peak_analysis(self):
        """ピーク解析モード（権限チェック付き）"""
        peak_analysis_mode()
    
    @require_permission("calibration")
    def _render_calibration(self):
        """検量線作成モード（権限チェック付き）"""
        calibration_mode()
    
    @require_permission("database_comparison")
    def _render_database_comparison(self):
        """データベース比較モード（権限チェック付き）"""
        database_comparison_mode()
    
    @require_permission("peak_ai_analysis")
    def _render_peak_ai_analysis(self):
        """AI解析モード（権限チェック付き）"""
        peak_ai_analysis_mode()

def main():
    """メイン関数"""
    app = RamanEyeApp()
    app.run()

if __name__ == "__main__":
    main()
