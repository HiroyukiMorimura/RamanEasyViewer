# -*- coding: utf-8 -*-
"""
統合ラマンスペクトル解析ツール
メインスクリプト

Created on Wed Jun 11 15:56:04 2025
@author: Hiroyuki Morimura

"""

import streamlit as st
import pandas as pd
from datetime import datetime

# 循環インポートを回避するため、必要な時にインポートする関数を定義
def get_auth_system():
    """認証システムを遅延インポート"""
    from auth_system import (
        AuthenticationManager, 
        UserRole, 
        require_auth, 
        require_permission,
        require_role
    )
    return {
        'AuthenticationManager': AuthenticationManager,
        'UserRole': UserRole,
        'require_auth': require_auth,
        'require_permission': require_permission,
        'require_role': require_role
    }

def get_ui_components():
    """UIコンポーネントを遅延インポート"""
    from user_management_ui import (
        LoginUI, 
        UserManagementUI, 
        ProfileUI, 
        render_authenticated_header
    )
    return {
        'LoginUI': LoginUI,
        'UserManagementUI': UserManagementUI,
        'ProfileUI': ProfileUI,
        'render_authenticated_header': render_authenticated_header
    }

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
        # 遅延初期化用の変数
        self._auth_system = None
        self._ui_components = None
        
        # ページ設定
        st.set_page_config(
            page_title="RamanEye Easy Viewer", 
            page_icon="🏢",  # ロゴのマークに変更（実際のロゴがある場合は画像パスを指定可能）
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # セッション状態の初期化
        if "show_profile" not in st.session_state:
            st.session_state.show_profile = False
        if "show_user_management" not in st.session_state:
            st.session_state.show_user_management = False
    
    def _get_auth_system(self):
        """認証システムの遅延取得"""
        if self._auth_system is None:
            self._auth_system = get_auth_system()
        return self._auth_system
    
    def _get_ui_components(self):
        """UIコンポーネントの遅延取得"""
        if self._ui_components is None:
            self._ui_components = get_ui_components()
        return self._ui_components
    
    def run(self):
        """メインアプリケーションの実行"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        # 認証チェック
        if not auth_manager.is_authenticated():
            self._render_login_page()
        else:
            # セッションタイムアウトチェック
            if not auth_manager.check_session_timeout(timeout_minutes=60):
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
                    st.image(
                        image, 
                        width=300,  # ロゴの幅を調整
                        caption="",
                        use_container_width=True
                    )
                    
                    logo_displayed = True
                    break
                    
                except Exception as e:
                    st.error(f"ロゴファイルの読み込みエラー ({logo_path}): {str(e)}")
        
    
    def _render_login_page(self):
        """ログインページの表示"""
        # カスタムCSS
        st.markdown(
            """
            <style>
            .login-header {
                color: #1f77b4;
                margin-bottom: 0.5rem;
                font-size: 0.5rem !important;
                font-weight: bold;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 2rem 0;
            }
            .feature-icon {
                font-size: 2rem;
                margin-bottom: 0.5rem;
            }
            
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # 上部レイアウト：ロゴ（1/2）+ ログイン（1/2）
        col_logo, col_login = st.columns([1, 1])  # 面積を半分半分に変更
        
        with col_logo:
            # ロゴ表示（左側、1/2サイズ、中央配置）
            self._display_company_logo()
        
        with col_login:
            # ログインフォーム
            with st.form("login_form"):
                st.markdown(
                    '<h2 class="login-header"><em>RamanEye</em> Easy Viewer ログイン</h2>',
                    unsafe_allow_html=True
                )
                
                username = st.text_input("ユーザー名", placeholder="ユーザー名を入力")
                password = st.text_input("パスワード", type="password", placeholder="パスワードを入力")
                
                col1, col2 = st.columns(2)
                with col1:
                    login_button = st.form_submit_button("ログイン", type="primary", use_container_width=True)
                with col2:
                    forgot_password = st.form_submit_button("パスワード忘れ", use_container_width=True)
            
            # ログイン処理
            if login_button:
                if username and password:
                    ui_components = self._get_ui_components()
                    login_ui = ui_components['LoginUI']()
                    success, message = login_ui.auth_manager.login(username, password)
                    if success:
                        st.success("ログインしました")
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("ユーザー名とパスワードを入力してください")
            
            # パスワードリセット（デモ用）
            if forgot_password:
                st.info("管理者にお問い合わせください")
        
        # デモアカウント情報をロゴとログインフォームの下に配置
        st.markdown("---")
        
        # 展開可能なデモアカウント情報
        with st.expander("🔧 デモアカウント情報", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **👑 管理者**
                - ユーザー名: `admin`
                - パスワード: `Admin123!`
                - 権限: 全機能アクセス可能
                """)
            
            with col2:
                st.markdown("""
                **🔬 分析者**
                - ユーザー名: `analyst`
                - パスワード: `Analyst123!`
                - 権限: 分析機能フルアクセス
                """)
            
            with col3:
                st.markdown("""
                **👁️ 閲覧者**
                - ユーザー名: `viewer`
                - パスワード: `Viewer123!`
                - 権限: 基本機能のみ
                """)
            
            st.info("💡 上記のアカウント情報をコピーしてログインフォームに入力してください")
        
        st.markdown("---")
        
        # 主要機能のアイコン群を表示
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
        
        # 2行4列のグリッドで機能を表示
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
        ui_components = self._get_ui_components()
        
        # 認証後ヘッダー
        ui_components['render_authenticated_header']()
        
        # プロファイル表示チェック
        if st.session_state.get("show_profile", False):
            profile_ui = ui_components['ProfileUI']()
            profile_ui.render_profile_page()
            if st.button("⬅️ メインメニューに戻る"):
                st.session_state.show_profile = False
                st.rerun()
            return
        
        # ユーザー管理表示チェック
        if st.session_state.get("show_user_management", False):
            user_management_ui = ui_components['UserManagementUI']()
            user_management_ui.render_user_management_page()
            if st.button("⬅️ メインメニューに戻る"):
                st.session_state.show_user_management = False
                st.rerun()
            return
        
        # メインタイトル
        st.markdown(
            "<h1>📊 <span style='font-style: italic;'>RamanEye</span> Easy Viewer</h1>",
            unsafe_allow_html=True
        )
        
        # サイドバー設定（メインアプリケーションでも表示）
        self._render_mode_sidebar()
        
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
            elif analysis_mode == "電子署名管理":
                self._render_signature_management()
            elif analysis_mode == "電子署名統合デモ":
                self._render_signature_integration_demo()
            elif analysis_mode == "ユーザー管理":
                st.session_state.show_user_management = True
                st.rerun()
            else:
                # デフォルトはスペクトル解析
                self._render_spectrum_analysis()
            self._render_secure_sidebar()
        
        except Exception as e:
            st.error(f"機能の実行中にエラーが発生しました: {e}")
            st.error("管理者にお問い合わせください。")
    
    def _render_mode_sidebar(self):
        """サイドバーの設定"""
        auth_system = self._get_auth_system()
        AuthenticationManager = auth_system['AuthenticationManager']
        UserRole = auth_system['UserRole']
        
        auth_manager = AuthenticationManager()
        
        st.sidebar.header("🔧 解析モード選択")
        
        # 現在のユーザーの権限を取得
        current_role = auth_manager.get_current_role()
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
        
        # 管理者・分析者は電子署名管理も利用可能
        if permissions.get("user_management", False) or current_role == "analyst":
            available_modes.append("電子署名管理")
        
        # 管理者は電子署名統合デモも利用可能
        if permissions.get("user_management", False):
            available_modes.append("電子署名統合デモ")
        
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
    def _render_secure_sidebar(self)
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
        - Version: 1.0.0 Secure Edition
        - Last Updated: 2025-07-31
        - Author: METASENSING
        """)
    
    def _render_mode_parameters(self, analysis_mode):
        """選択されたモードに応じたパラメータ設定を表示"""
        if analysis_mode == "スペクトル解析":
            st.sidebar.number_input("波数範囲 開始", value=200, min_value=0, max_value=4000, key="start_wavenum")
            st.sidebar.number_input("波数範囲 終了", value=2000, min_value=0, max_value=4000, key="end_wavenum")
            st.sidebar.slider("ベースライン閾値", 0.001, 0.1, 0.01, key="dssn_th")
            st.sidebar.selectbox("平滑化窓サイズ", [3, 5, 7, 9, 11], index=2, key="savgol_wsize")
            
        elif analysis_mode == "ラマンピークファインダー":
            st.sidebar.slider("ピーク検出閾値", 0.01, 1.0, 0.1, key="peak_threshold")
            st.sidebar.number_input("最小ピーク高さ", value=0.05, min_value=0.01, max_value=1.0, key="min_height")
            st.sidebar.number_input("最小ピーク距離", value=10, min_value=1, max_value=100, key="min_distance")
            
        elif analysis_mode == "ラマンピーク分離":
            st.sidebar.number_input("フィッティング開始", value=800, min_value=0, max_value=4000, key="fit_start")
            st.sidebar.number_input("フィッティング終了", value=1200, min_value=0, max_value=4000, key="fit_end")
            st.sidebar.selectbox("最大ピーク数", [1, 2, 3, 4, 5, 6], index=2, key="max_peaks")
            
        elif analysis_mode == "多変量解析":
            st.sidebar.selectbox("コンポーネント数", [2, 3, 4, 5], index=1, key="n_components")
            st.sidebar.selectbox("解析手法", ["PCA", "K-means", "階層クラスター"], key="analysis_method")
            st.sidebar.checkbox("標準化", value=True, key="normalize")
            
        elif analysis_mode == "検量線作成":
            st.sidebar.selectbox("検量線タイプ", ["ピーク面積", "PLS回帰"], key="calibration_type")
            st.sidebar.number_input("解析波数範囲 開始", value=800, min_value=0, max_value=4000, key="cal_start")
            st.sidebar.number_input("解析波数範囲 終了", value=1200, min_value=0, max_value=4000, key="cal_end")
            
        elif analysis_mode == "ピークAI解析":
            st.sidebar.selectbox("AI モデル", ["GPT-4", "Claude", "ローカルモデル"], key="ai_model")
            st.sidebar.checkbox("RAG機能を使用", value=True, key="use_rag")
            st.sidebar.slider("応答の詳細度", 1, 5, 3, key="detail_level")
            
        elif analysis_mode == "データベース比較":
            st.sidebar.selectbox("比較手法", ["相関係数", "コサイン類似度", "ユークリッド距離"], key="comparison_method")
            st.sidebar.number_input("上位N個表示", value=10, min_value=1, max_value=50, key="top_n")
            st.sidebar.checkbox("プーリング計算", value=True, key="use_pooling")
            
        elif analysis_mode == "電子署名管理":
            st.sidebar.selectbox("表示フィルター", ["すべて", "完了", "待機中", "拒否"], key="signature_filter")
            st.sidebar.number_input("表示件数", value=50, min_value=10, max_value=500, key="signature_limit")
            
        elif analysis_mode == "電子署名統合デモ":
            st.sidebar.info("デモ用パラメータ")
            st.sidebar.checkbox("デバッグモード", value=False, key="debug_mode")
            st.sidebar.selectbox("デモレベル", ["基本", "詳細"], key="demo_level")
            
        elif analysis_mode == "ユーザー管理":
            st.sidebar.selectbox("表示フィルター", ["すべて", "アクティブ", "ロック中"], key="user_filter")
            st.sidebar.checkbox("詳細表示", value=True, key="detailed_view")
        
        else:
            st.sidebar.info("パラメータはありません")
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
            
            "電子署名管理": """
            **電子署名管理モード:**
            1. **ペンディング署名確認**: 署名待ちの操作を確認・実行
            2. **署名実行**: パスワード再入力＋署名理由入力で電子署名
            3. **署名履歴確認**: 過去の署名記録を確認・監査
            4. **署名統計**: 署名の完了率・拒否率などの統計情報
            5. **署名設定**: 署名ポリシー・セキュリティ設定の管理
            
            **署名レベル:**
            - **一段階署名**: 一人の承認で完了
            - **二段階署名**: 二人の承認が必要（重要な操作）
            
            **署名情報記録:**
            - 署名者氏名（印字名）・日時・理由・UserID
            - タイムスタンプ付きで改ざん防止
            - 完全な監査証跡を提供
            
            **⚠️ 管理者・分析者が利用可能**
            """,
            
            "電子署名統合デモ": """
            **電子署名統合デモモード:**
            1. **セキュア操作デモ**: 署名が必要な操作の実例
            2. **データエクスポート**: 一段階署名が必要なデータ出力
            3. **レポート確定**: 二段階署名が必要な重要操作
            4. **データベース更新**: セキュリティ機能付きDB操作
            5. **システム設定変更**: 高セキュリティ設定操作
            6. **統合ガイド**: 既存機能への署名統合方法
            
            **デモ機能:**
            - **一段階署名**: パスワード再入力＋理由記録
            - **二段階署名**: 二人の承認が必要な重要操作
            - **署名記録**: 完全な監査証跡の提供
            - **統合例**: 実際の機能への適用方法
            
            **学習内容:**
            - 電子署名の実装方法
            - セキュリティポリシーの設定
            - コンプライアンス対応
            - ベストプラクティス
            
            **⚠️ 管理者専用デモ機能**
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
            
            "電子署名管理": """
            **電子署名管理モード:**
            1. **ペンディング署名確認**: 署名待ちの操作を確認・実行
            2. **署名実行**: パスワード再入力＋署名理由入力で電子署名
            3. **署名履歴確認**: 過去の署名記録を確認・監査
            4. **署名統計**: 署名の完了率・拒否率などの統計情報
            5. **署名設定**: 署名ポリシー・セキュリティ設定の管理
            
            **署名レベル:**
            - **一段階署名**: 一人の承認で完了
            - **二段階署名**: 二人の承認が必要（重要な操作）
            
            **署名情報記録:**
            - 署名者氏名（印字名）・日時・理由・UserID
            - タイムスタンプ付きで改ざん防止
            - 完全な監査証跡を提供
            
            **⚠️ 管理者・分析者が利用可能**
            """,
            
            "電子署名統合デモ": """
            **電子署名統合デモモード:**
            1. **セキュア操作デモ**: 署名が必要な操作の実例
            2. **データエクスポート**: 一段階署名が必要なデータ出力
            3. **レポート確定**: 二段階署名が必要な重要操作
            4. **データベース更新**: セキュリティ機能付きDB操作
            5. **システム設定変更**: 高セキュリティ設定操作
            6. **統合ガイド**: 既存機能への署名統合方法
            
            **デモ機能:**
            - **一段階署名**: パスワード再入力＋理由記録
            - **二段階署名**: 二人の承認が必要な重要操作
            - **署名記録**: 完全な監査証跡の提供
            - **統合例**: 実際の機能への適用方法
            
            **学習内容:**
            - 電子署名の実装方法
            - セキュリティポリシーの設定
            - コンプライアンス対応
            - ベストプラクティス
            
            **⚠️ 管理者専用デモ機能**
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
    def _render_spectrum_analysis(self):
        """スペクトル解析モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("spectrum_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        spectrum_analysis_mode()
    
    def _render_multivariate_analysis(self):
        """多変量解析モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("multivariate_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        multivariate_analysis_mode()
    
    def _render_peak_deconvolution(self):
        """ピーク分離モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_deconvolution"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        peak_deconvolution_mode()
    
    def _render_peak_analysis(self):
        """ピーク解析モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        peak_analysis_mode()
    
    def _render_calibration(self):
        """検量線作成モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("calibration"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        calibration_mode()
    
    def _render_database_comparison(self):
        """データベース比較モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("database_comparison"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        database_comparison_mode()
    
    def _render_peak_ai_analysis(self):
        """AI解析モード（権限チェック付き）"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_ai_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        peak_ai_analysis_mode()
    
    def _render_signature_management(self):
        """電子署名管理モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        # 管理者または分析者のみアクセス可能
        current_role = auth_manager.get_current_role()
        if current_role not in ["admin", "analyst"]:
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        try:
            from signature_management_ui import render_signature_demo_page
            render_signature_demo_page()
        except ImportError:
            st.error("電子署名管理機能がインストールされていません")
            st.info("電子署名機能を使用するには、追加のモジュールをインストールしてください")
    
    def _render_signature_integration_demo(self):
        """電子署名統合デモモード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        # 管理者のみアクセス可能
        current_role = auth_manager.get_current_role()
        if current_role != "admin":
            st.error("この機能を使用する権限がありません")
            st.info("電子署名統合デモは管理者専用機能です")
            st.stop()
        
        try:
            from signature_integration_example import demo_secure_operations, signature_integration_guide
            
            st.header("🔐 電子署名統合デモ")
            
            st.markdown("""
            このページでは、電子署名システムの実装例と統合方法をデモンストレーションします。
            管理者として、重要な操作に電子署名を統合する方法を学習できます。
            """)
            
            # 電子署名システムの状態表示
            try:
                from electronic_signature import SignatureLevel
                st.success("✅ 電子署名システムが正常に動作しています")
            except ImportError:
                st.warning("⚠️ 電子署名システムが利用できません（デモモードで動作）")
                st.info("electronic_signature.py モジュールをインストールすると完全な機能が利用できます")
            
            # タブで機能を分離
            tab1, tab2 = st.tabs(["セキュア操作デモ", "統合ガイド"])
            
            with tab1:
                st.markdown("### 🎯 署名が必要な操作の実例")
                st.info("以下の操作を実行すると、電子署名のプロセスを体験できます")
                demo_secure_operations()
            
            with tab2:
                st.markdown("### 📚 電子署名統合ガイド")
                signature_integration_guide()
                
        except ImportError as e:
            st.error("電子署名統合デモ機能がインストールされていません")
            st.error(f"エラー詳細: {e}")
            st.info("signature_integration_example.py ファイルが必要です")
            
            # フォールバック：基本的な説明を表示
            st.markdown("---")
            st.subheader("📋 電子署名統合について")
            st.markdown("""
            電子署名統合デモでは、以下の機能を提供します：
            
            **🔐 セキュア操作例**:
            - データエクスポート（一段階署名）
            - レポート確定（二段階署名）
            - データベース更新（一段階署名）
            - システム設定変更（二段階署名）
            
            **📚 統合ガイド**:
            - デコレータベースの実装方法
            - 署名レベルの選択基準
            - セキュリティ考慮事項
            - コンプライアンス対応
            
            **実装方法**:
            ```python
            @require_signature(
                operation_type="重要操作",
                signature_level=SignatureLevel.DUAL
            )
            def secure_operation():
                # 実際の処理
            ```
            """)
        
        except Exception as e:
            st.error(f"電子署名統合デモの実行中にエラーが発生しました: {e}")
            st.info("管理者にお問い合わせください")

def main():
    """メイン関数"""
    app = RamanEyeApp()
    app.run()

if __name__ == "__main__":
    main()
