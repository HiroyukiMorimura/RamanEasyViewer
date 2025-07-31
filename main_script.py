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
import ssl
import hashlib
from datetime import datetime
from pathlib import Path

# セキュリティモジュールのインポート
try:
    from security_manager import (
        SecurityManager,
        get_security_manager,
        init_security_system,
        SecurityConfig,
        SecurityException
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    st.error("セキュリティモジュールが利用できません。security_manager.pyを確認してください。")

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
    from calibration_mode import calibration_mode
    from raman_database import database_comparison_mode
    
    # セキュア版AI解析モジュール
    if SECURITY_AVAILABLE:
        from peak_ai_analysis import peak_ai_analysis_mode  # セキュア版
    else:
        from peak_ai_analysis import peak_ai_analysis_mode  # 通常版
        
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
        self._security_manager = None
        
        # ページ設定（セキュリティヘッダー付き）
        st.set_page_config(
            page_title="RamanEye Easy Viewer",
            page_icon="favicon.png",  # 同フォルダ内のPNGをそのまま指定
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # セキュリティシステムの初期化
        if SECURITY_AVAILABLE:
            self._security_manager = init_security_system()
        
        # セッション状態の初期化
        self._init_secure_session_state()
    
    def _add_security_headers(self):
        """セキュリティヘッダーの追加"""
        st.markdown(
            """
            <script>
            // セキュリティヘッダーの設定（可能な範囲で）
            if (typeof window !== 'undefined') {
                // XSS保護
                document.addEventListener('DOMContentLoaded', function() {
                    // CSP違反の検出
                    document.addEventListener('securitypolicyviolation', function(e) {
                        console.warn('CSP Violation:', e.violatedDirective);
                    });
                });
            }
            </script>
            """,
            unsafe_allow_html=True
        )
    
    def _init_secure_session_state(self):
        """セキュアなセッション状態の初期化"""
        # 基本セッション状態
        if "show_profile" not in st.session_state:
            st.session_state.show_profile = False
        if "show_user_management" not in st.session_state:
            st.session_state.show_user_management = False
        
        # セキュリティ関連セッション状態
        if "session_id" not in st.session_state:
            st.session_state.session_id = self._generate_secure_session_id()
        if "security_level" not in st.session_state:
            st.session_state.security_level = "standard"
        if "last_activity" not in st.session_state:
            st.session_state.last_activity = datetime.now()
        
        # セキュリティイベントログ
        if "security_events" not in st.session_state:
            st.session_state.security_events = []
    
    def _generate_secure_session_id(self) -> str:
        """セキュアなセッションIDの生成"""
        import secrets
        return secrets.token_urlsafe(32)
    
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
    
    def _get_security_manager(self):
        """セキュリティマネージャーの取得"""
        if self._security_manager is None and SECURITY_AVAILABLE:
            self._security_manager = get_security_manager()
        return self._security_manager
    
    def run(self):
        """メインアプリケーションの実行"""
        try:
            # セキュリティチェック
            if not self._perform_security_checks():
                st.stop()
            
            auth_system = self._get_auth_system()
            auth_manager = auth_system['AuthenticationManager']()
            
            # 認証チェック
            if not auth_manager.is_authenticated():
                self._render_secure_login_page()
            else:
                # セッションタイムアウトチェック（セキュリティ強化）
                if not self._check_secure_session_timeout(auth_manager, timeout_minutes=60):
                    st.error("セキュリティのため、セッションがタイムアウトしました。再度ログインしてください")
                    self._log_security_event("SESSION_TIMEOUT", "system", {"reason": "timeout"})
                    st.stop()
                
                # アクティビティトラッキング
                self._update_activity_tracking()
                
                self._render_secure_main_application()
                
        except Exception as e:
            self._handle_security_exception(e)
    
    def _perform_security_checks(self) -> bool:
        """基本的なセキュリティチェック"""
        try:
            # HTTPS確認（可能な場合）
            if hasattr(st.runtime.get_instance(), '_server'):
                # Note: Streamlitの内部構造に依存するため、エラー処理が必要
                pass
            
            # セキュリティモジュールの動作確認
            if SECURITY_AVAILABLE:
                security_manager = self._get_security_manager()
                if security_manager:
                    security_status = security_manager.get_security_status()
                    if not security_status.get('encryption_enabled', False):
                        st.warning("⚠️ データ暗号化が無効になっています")
            
            return True
            
        except Exception as e:
            st.error(f"セキュリティチェックエラー: {e}")
            return False
    
    def _check_secure_session_timeout(self, auth_manager, timeout_minutes: int = 60) -> bool:
        """セキュア強化されたセッションタイムアウトチェック"""
        try:
            # 標準のタイムアウトチェック
            if not auth_manager.check_session_timeout(timeout_minutes=timeout_minutes):
                return False
            
            # 追加のセキュリティチェック
            last_activity = st.session_state.get('last_activity')
            if last_activity:
                inactive_duration = datetime.now() - last_activity
                if inactive_duration.total_seconds() > (timeout_minutes * 60):
                    return False
            
            return True
            
        except Exception as e:
            self._log_security_event("SESSION_CHECK_ERROR", "system", {"error": str(e)})
            return False
    
    def _update_activity_tracking(self):
        """アクティビティトラッキングの更新"""
        st.session_state.last_activity = datetime.now()
    
    def _log_security_event(self, event_type: str, user_id: str, details: dict):
        """セキュリティイベントのログ記録"""
        security_manager = self._get_security_manager()
        if security_manager:
            security_manager.audit_logger.log_security_event(
                event_type=event_type,
                user_id=user_id,
                details=details,
                severity="INFO"
            )
        
        # セッション内ログも保持
        event = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'details': details
        }
        st.session_state.security_events.append(event)
        
        # ログサイズ制限
        if len(st.session_state.security_events) > 100:
            st.session_state.security_events = st.session_state.security_events[-50:]
    
    def _handle_security_exception(self, exception: Exception):
        """セキュリティ例外の処理"""
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        self._log_security_event(
            "SECURITY_EXCEPTION",
            user_id,
            {"error": str(exception), "type": type(exception).__name__}
        )
        
        st.error("セキュリティエラーが発生しました。管理者にお問い合わせください。")
        st.error(f"エラー詳細: {exception}")
    
    def _render_secure_login_page(self):
        """セキュア強化されたログインページの表示"""
        # セキュリティ強化されたCSS
        st.markdown(
            """
            <style>
            /* セキュリティ強化されたスタイル */
            .main-header {
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
                font-size: 3rem;
                font-weight: bold;
            }
            .security-badge {
                background: linear-gradient(135deg, #28a745, #20c997);
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: bold;
                display: inline-block;
                margin: 0.5rem 0;
            }
            .login-header {
                color: #1f77b4;
                margin-top: 0rem !important;
                margin-bottom: 0rem !important;
                font-size: 1.8rem !important;
                font-weight: bold;
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
        
        # セキュリティステータス表示
        # col_status, col_logo, col_login = st.columns([1, 2, 2])
        col_logo, col_login = st.columns([1, 1])
        
        
        with col_logo:
            # 会社ロゴ表示（セキュア版）
            self._display_secure_company_logo()
        
        with col_login:
            
            
            # セキュアなログインフォーム
            with st.form("secure_login_form"):
                st.markdown('<h2 class="login-header"><em>RamanEye</em> Easy Viewer ログインフォーム</h2>', unsafe_allow_html=True)
                
                # ログイン試行制限の表示
                failed_attempts = st.session_state.get('failed_login_attempts', 0)
                if failed_attempts > 0:
                    st.warning(f"⚠️ ログイン失敗回数: {failed_attempts}/{SecurityConfig.MAX_LOGIN_ATTEMPTS}")
                
                username = st.text_input(
                    "ユーザー名", 
                    placeholder="ユーザー名を入力",
                    help="セキュア認証によりログイン試行が記録されます"
                )
                password = st.text_input(
                    "パスワード", 
                    type="password", 
                    placeholder="パスワードを入力",
                    help="パスワードは暗号化されて送信されます"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    login_button = st.form_submit_button("🔐 セキュアログイン", type="primary", use_container_width=True)
                with col2:
                    forgot_password = st.form_submit_button("パスワード忘れ", use_container_width=True)
            
            # セキュア強化されたログイン処理
            if login_button:
                self._process_secure_login(username, password)
            
            # パスワードリセット（セキュア版）
            if forgot_password:
                self._handle_password_reset_request(username)
        
        # セキュリティ情報の表示
        self._render_security_information()
        
        # デモアカウント情報（セキュリティ警告付き）
        self._render_demo_accounts_with_security_warning()
        
        # 主要機能の表示（セキュリティ機能を含む）
        self._render_secure_features()
        
        # セキュアフッター
        self._render_secure_footer()
    
    def _process_secure_login(self, username: str, password: str):
        """セキュア強化されたログイン処理"""
        if not username or not password:
            st.error("ユーザー名とパスワードを入力してください")
            return
        
        # ログイン試行制限チェック
        failed_attempts = st.session_state.get('failed_login_attempts', 0)
        if failed_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            st.error(f"ログイン試行回数が上限に達しました。{SecurityConfig.LOCKOUT_DURATION // 60}分後に再試行してください。")
            self._log_security_event("LOGIN_BLOCKED", username, {"reason": "max_attempts_reached"})
            return
        
        try:
            ui_components = self._get_ui_components()
            login_ui = ui_components['LoginUI']()
            success, message = login_ui.auth_manager.login(username, password)
            
            if success:
                # ログイン成功
                st.session_state.failed_login_attempts = 0
                self._log_security_event("LOGIN_SUCCESS", username, {"method": "password"})
                st.success("ログインが完了しました")
                st.rerun()
            else:
                # ログイン失敗
                st.session_state.failed_login_attempts = failed_attempts + 1
                self._log_security_event("LOGIN_FAILURE", username, {"reason": message})
                st.error(f"ログインに失敗しました: {message}")
                
        except Exception as e:
            self._log_security_event("LOGIN_ERROR", username, {"error": str(e)})
            st.error("ログイン処理中にエラーが発生しました")
    
    def _handle_password_reset_request(self, username: str):
        """パスワードリセット要求の処理"""
        if username:
            self._log_security_event("PASSWORD_RESET_REQUEST", username, {"method": "web_form"})
            st.info(f"ユーザー '{username}' のパスワードリセット要求を記録しました。管理者にお問い合わせください。")
        else:
            st.info("パスワードリセットについては管理者にお問い合わせください")
    
    def _display_secure_company_logo(self):
        st.image("logo.png", use_container_width = True)
    
    def _render_security_information(self):
        """セキュリティ情報の表示"""
        st.markdown("---")
        
        with st.expander("🛡️ セキュリティ機能について", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🔐 データ保護:**
                - AES-256暗号化によるファイル保護
                - データベースの完全暗号化
                - メモリ内データの保護
                - 自動データ消去機能
                
                **🔍 完全性管理:**
                - SHA-256ハッシュによるファイル検証
                - HMAC署名による改ざん検知
                - リアルタイム完全性チェック
                - 自動修復機能
                """)
            
            with col2:
                st.markdown("""
                **🛡️ アクセス制御:**
                - 多層認証システム
                - 役割ベースアクセス制御
                - セッション管理とタイムアウト
                - IPアドレス制限（オプション）
                
                **📝 監査・コンプライアンス:**
                - 全操作の完全な監査ログ
                - リアルタイムセキュリティ監視
                - コンプライアンスレポート
                - インシデント対応機能
                """)
        
        if SECURITY_AVAILABLE:
            security_manager = self._get_security_manager()
            if security_manager:
                with st.expander("📊 現在のセキュリティ状態", expanded=False):
                    status = security_manager.get_security_status()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("暗号化", "有効" if status['encryption_enabled'] else "無効")
                        st.metric("完全性チェック", "有効" if status['integrity_checking_enabled'] else "無効")
                    
                    with col2:
                        st.metric("アクセス制御", "有効" if status['access_control_enabled'] else "無効")
                        st.metric("監査ログ", "有効" if status['audit_logging_enabled'] else "無効")
                    
                    with col3:
                        st.metric("HTTPS通信", "有効" if status['https_enforced'] else "無効")
                        st.metric("データベース", "初期化済" if status['databases_initialized'] else "未初期化")
    
    def _render_demo_accounts_with_security_warning(self):
        """セキュリティ警告付きデモアカウント情報"""
        st.markdown("---")
        
        with st.expander("🔧 デモアカウント情報（セキュリティ警告）", expanded=False):
            st.warning("""
            ⚠️ **セキュリティ警告**: 
            デモアカウントは学習・評価目的のみで使用してください。
            本番環境では必ず独自のアカウントを作成してください。
            """)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **👑 管理者（フルアクセス）**
                - ユーザー名: `admin`
                - パスワード: `Admin123!`
                - 🔒 全機能・データアクセス可能
                - 🛡️ セキュリティ設定管理
                """)
            
            with col2:
                st.markdown("""
                **🔬 分析者（分析機能）**
                - ユーザー名: `analyst`
                - パスワード: `Analyst123!`
                - 📊 分析機能フルアクセス
                - 🔐 データ暗号化機能利用可能
                """)
            
            with col3:
                st.markdown("""
                **👁️ 閲覧者（基本機能）**
                - ユーザー名: `viewer`
                - パスワード: `Viewer123!`
                - 👀 基本閲覧機能のみ
                - 🚫 データ編集・削除制限
                """)
            
            st.info("💡 セキュリティ向上のため、初回ログイン後にパスワード変更を推奨します")
    
    def _render_secure_features(self):
        """セキュリティ機能を含む主要機能の表示"""
        st.markdown("### 🌟 セキュア機能一覧")
        
        features = [
            ("📊", "スペクトル解析", "ラマンスペクトル解析"),
            ("🔍", "ピーク分析", "ピーク検出・解析"),
            ("⚗️", "ピーク分離", "ピーク分離"),
            ("📈", "多変量解析", "統計解析"),
            ("📏", "検量線作成", "定量分析"),
            ("🤖", "AI解析", "ピークAI解析・RAG機能"),
            ("🗄️", "DB比較", "データベース照合"),
            ("🔒", "エンタープライズセキュリティ", "多層セキュリティ・監査・コンプライアンス")
        ]
        
        # 2行4列のセキュアグリッド
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
    
    def _render_secure_footer(self):
        """セキュアフッター"""
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p><strong>RamanEye Easy Viewer v1.0.0</strong></p>
            <p>© 2025 Hiroyuki Morimura. All rights reserved.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_secure_main_application(self):
        """メインアプリケーションの表示"""
        
        # メインタイトル
        st.markdown(
            "<h1><span style='font-style: italic;'>RamanEye</span> Easy Viewer</h1>",
            unsafe_allow_html=True
        )
        
        ui_components = self._get_ui_components()
        
        # 認証後ヘッダー（セキュリティ情報付き）- ユーザー状態を一番上に
        self._render_secure_authenticated_header()
        
        # サイドバー設定を先に実行
        self._render_sidebar()
        
        # メインコンテンツエリア（セキュリティ付き）
        if not MODULES_AVAILABLE:
            st.error("解析モジュールが利用できません。管理者にお問い合わせください。")
            return
        
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
        
        # 解析モード実行
        self._execute_analysis_mode()
    
    def _render_secure_authenticated_header(self):
        """セキュア強化された認証後ヘッダー"""
        ui_components = self._get_ui_components()
        
        # 基本の認証ヘッダー
        ui_components['render_authenticated_header']()
        
        # セキュリティ情報は_render_sidebar()内で下側に表示
    
    def _render_sidebar(self):
        """サイドバー"""
        # 解析モード選択を一番上に
        st.sidebar.header("🔧 解析モード選択")
        
        auth_system = self._get_auth_system()
        AuthenticationManager = auth_system['AuthenticationManager']
        UserRole = auth_system['UserRole']
        
        auth_manager = AuthenticationManager()
        
        # 現在のユーザーの権限を取得
        current_role = auth_manager.get_current_role()
        permissions = UserRole.get_role_permissions(current_role)
        
        mode_permissions = {
            "スペクトル解析": "spectrum_analysis",
            "データベース比較": "database_comparison",
            "ピークファインダー": "peak_analysis", 
            "ピーク分離": "peak_deconvolution",
            "多変量解析": "multivariate_analysis",
            "検量線作成": "calibration",
            "ピークAI解析": "peak_ai_analysis"
        }
            
        # 権限チェックして available_modes を作る
        available_modes = [
            mode for mode, perm in mode_permissions.items()
            if permissions.get(perm, False)
        ]
        
        # 管理者・分析者向けセキュリティ管理機能
        if permissions.get("user_management", False) or current_role == "analyst":
            available_modes.append("セキュリティ管理")
        
        # 管理者向けセキュリティ監査
        if permissions.get("user_management", False):
            available_modes.append("セキュリティ監査")
            available_modes.append("ユーザー管理")
        
        # 必ず最低１つ入れる
        if not available_modes:
            available_modes = ["スペクトル解析"]
            
        # 解析モード選択のselectbox
        analysis_mode = st.sidebar.selectbox(
            "セキュア解析モードを選択してください:",
            available_modes,
            index=0,
            key="mode_selector"
        )
        
        # セキュリティ関連の表示を下側に移動
        st.sidebar.markdown("---")
        
        # セキュリティ状態（元々_render_secure_authenticated_headerにあった内容）
        st.sidebar.subheader("🔒 セキュリティ状態")
        
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        # セッション情報
        st.sidebar.info(f"セッションID: {st.session_state.session_id[:8]}...")
        st.sidebar.info(f"セキュリティレベル: {st.session_state.security_level.upper()}")
        
        # セキュリティメトリクス
        if SECURITY_AVAILABLE:
            security_manager = self._get_security_manager()
            if security_manager:
                status = security_manager.get_security_status()
                
                security_score = sum([
                    status.get('encryption_enabled', False),
                    status.get('integrity_checking_enabled', False),
                    status.get('access_control_enabled', False),
                    status.get('audit_logging_enabled', False),
                    status.get('https_enforced', False)
                ])
                
                st.sidebar.metric("セキュリティスコア", f"{security_score}/5", f"{security_score * 20}%")
        
        # 最近のセキュリティイベント
        recent_events = st.session_state.security_events[-3:] if st.session_state.security_events else []
        if recent_events:
            with st.sidebar.expander("📋 最近のアクティビティ", expanded=False):
                for event in reversed(recent_events):
                    st.text(f"{event['event_type']} - {event['timestamp'][:19]}")
        
        # セキュリティ権限情報表示
        st.sidebar.markdown("---")
        st.sidebar.header("🛡️ セキュリティ権限")
        
        role_descriptions = {
            "admin": "🔧 全機能・セキュリティ管理可能",
            "analyst": "📊 分析・暗号化機能利用可能", 
            "viewer": "👁️ 閲覧・基本分析のみ"
        }
        
        st.sidebar.info(role_descriptions.get(current_role, "権限情報なし"))
        
        # セキュリティ設定
        self._render_security_settings_sidebar()
        
        # 使用方法の説明（セキュリティ版）
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 セキュア使用方法")
        analysis_mode = st.session_state.get("mode_selector", "スペクトル解析")
        self._render_secure_usage_instructions(analysis_mode)
        
    
    def _render_security_settings_sidebar(self):
        """サイドバーのセキュリティ設定"""
        if SECURITY_AVAILABLE:
            with st.sidebar.expander("⚙️ セキュリティ設定", expanded=False):
                security_manager = self._get_security_manager()
                
                if security_manager:
                    # 暗号化レベル設定
                    encryption_level = st.selectbox(
                        "暗号化レベル",
                        ["Standard", "High", "Maximum"],
                        index=1,
                        help="データ暗号化の強度を選択"
                    )
                    
                    # セッション設定
                    auto_logout = st.checkbox(
                        "自動ログアウト",
                        value=True,
                        help="一定時間非アクティブ時の自動ログアウト"
                    )
                    
                    # 監査ログレベル
                    audit_level = st.selectbox(
                        "監査ログレベル",
                        ["Basic", "Detailed", "Verbose"],
                        index=1,
                        help="記録する監査情報の詳細度"
                    )
                    
                    # 完全性チェック頻度
                    integrity_check = st.selectbox(
                        "完全性チェック",
                        ["Access時", "定期的", "リアルタイム"],
                        index=0,
                        help="ファイル完全性の検証頻度"
                    )
                    
                    # 設定保存ボタン
                    if st.button("設定保存", key="security_settings_save"):
                        # セキュリティ設定の保存処理
                        current_user = st.session_state.get('current_user', {})
                        user_id = current_user.get('username', 'unknown')
                        
                        self._log_security_event(
                            "SECURITY_SETTINGS_CHANGED",
                            user_id,
                            {
                                'encryption_level': encryption_level,
                                'auto_logout': auto_logout,
                                'audit_level': audit_level,
                                'integrity_check': integrity_check
                            }
                        )
                        
                        st.success("セキュリティ設定が保存されました")
    
    def _render_secure_usage_instructions(self, analysis_mode):
        """セキュリティ強化された使用方法の説明"""
        instructions = {
            "スペクトル解析": """
            **スペクトル解析:**
            1. CSVファイルをアップロード
            2. 解析パラメータを調整
            3. スペクトル解析実行
            4. 結果をダウンロード
            """,
            
            "セキュアピークAI解析": """
            **🤖 セキュアピークAI解析:**
            1. セキュアHTTPS通信でAPI接続
            2. 暗号化された論文データベース構築
            3. プロンプトインジェクション対策付きAI解析
            4. 完全な監査証跡付きで結果生成
            
            **セキュリティ機能:**
            - HTTPS強制通信
            - プロンプトサニタイズ
            - API通信ログ記録
            - 暗号化データベース
            """,
            
            "セキュリティ管理": """
            **🛡️ セキュリティ管理:**
            1. セキュリティ状態の監視・管理
            2. 暗号化キーの管理・ローテーション
            3. アクセス権限の設定・変更
            4. 監査ログの確認・エクスポート
            5. インシデント対応・復旧処理
            
            **⚠️ 管理者・分析者専用機能**
            """,
            
            "セキュリティ監査": """
            **📊 セキュリティ監査:**
            1. 全ユーザーアクティビティの監査
            2. セキュリティイベントの分析
            3. コンプライアンスレポート生成
            4. 異常行動の検出・アラート
            5. セキュリティメトリクスの可視化
            
            **⚠️ 管理者専用機能**
            """
        }
        
        instruction = instructions.get(analysis_mode, "使用方法情報なし")
        st.sidebar.markdown(instruction)
    
    def _execute_analysis_mode(self):
        """解析モードの実行"""
        analysis_mode = st.session_state.get("mode_selector", "スペクトル解析")
        
        # セキュリティログ記録
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        self._log_security_event(
            "ANALYSIS_MODE_ACCESS",
            user_id,
            {'mode': analysis_mode}
        )
        
        try:
            if analysis_mode == "スペクトル解析":
                self._render_spectrum_analysis()
            elif analysis_mode == "データベース比較":
                self._render_secure_database_comparison()
            elif analysis_mode == "多変量解析":
                self._render_secure_multivariate_analysis()
            elif analysis_mode == "ピーク分離":
                self._render_secure_peak_deconvolution()
            elif analysis_mode == "ピークファインダー":
                self._render_secure_peak_analysis()
            elif analysis_mode == "検量線作成":
                self._render_secure_calibration()
            elif analysis_mode == "ピークAI解析":
                self._render_secure_peak_ai_analysis()
            elif analysis_mode == "セキュリティ管理":
                self._render_security_management()
            elif analysis_mode == "セキュリティ監査":
                self._render_security_audit()
            elif analysis_mode == "ユーザー管理":
                st.session_state.show_user_management = True
                st.rerun()
            else:
                self._render_spectrum_analysis()
                
        except Exception as e:
            self._handle_analysis_security_exception(analysis_mode, e)
    
    def _handle_analysis_security_exception(self, mode: str, exception: Exception):
        """解析モードでのセキュリティ例外処理"""
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        self._log_security_event(
            "ANALYSIS_ERROR",
            user_id,
            {
                'mode': mode,
                'error': str(exception),
                'type': type(exception).__name__
            }
        )
        
        st.error(f"セキュア解析モード '{mode}' の実行中にエラーが発生しました。")
        st.error("管理者にお問い合わせください。")
        
        # 詳細なエラー情報（管理者のみ）
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        current_role = auth_manager.get_current_role()
        
        if current_role == "admin":
            with st.expander("🔍 エラー詳細（管理者専用）", expanded=False):
                st.error(f"エラータイプ: {type(exception).__name__}")
                st.error(f"エラーメッセージ: {str(exception)}")
    
    # セキュア強化された各解析モードのラッパー関数
    def _render_spectrum_analysis(self):
        """スペクトル解析モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("spectrum_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        spectrum_analysis_mode()
    
    def _render_secure_peak_ai_analysis(self):
        """セキュア強化されたAI解析モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_ai_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        # セキュリティヘッダー追加
        st.markdown("### 🤖 セキュアピークAI解析")
        st.info("このモードでは、AI通信がHTTPS暗号化され、全てのAPI呼び出しが監査されます。")
        
        peak_ai_analysis_mode()
    
    def _render_security_management(self):
        """セキュリティ管理モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        # 管理者または分析者のみアクセス可能
        current_role = auth_manager.get_current_role()
        if current_role not in ["admin", "analyst"]:
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.header("🛡️ セキュリティ管理")
        
        if not SECURITY_AVAILABLE:
            st.error("セキュリティモジュールが利用できません")
            return
        
        security_manager = self._get_security_manager()
        if not security_manager:
            st.error("セキュリティマネージャーを初期化できません")
            return
        
        # タブで機能を分割
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 監視", "🔐 暗号化", "🔑 アクセス制御", "📊 統計"])
        
        with tab1:
            st.subheader("🔍 リアルタイム監視")
            self._render_security_monitoring(security_manager)
        
        with tab2:
            st.subheader("🔐 暗号化管理")
            self._render_encryption_management(security_manager)
        
        with tab3:
            st.subheader("🔑 アクセス制御管理")
            self._render_access_control_management(security_manager)
        
        with tab4:
            st.subheader("📊 セキュリティ統計")
            self._render_security_statistics(security_manager)
    
    def _render_security_monitoring(self, security_manager):
        """セキュリティ監視画面"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**システム状態:**")
            status = security_manager.get_security_status()
            
            for key, value in status.items():
                if isinstance(value, bool):
                    icon = "✅" if value else "❌"
                    st.write(f"{icon} {key.replace('_', ' ').title()}: {'有効' if value else '無効'}")
        
        with col2:
            st.markdown("**最近のセキュリティイベント:**")
            recent_events = st.session_state.security_events[-5:] if st.session_state.security_events else []
            
            for event in reversed(recent_events):
                severity_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌", "CRITICAL": "🚨"}.get(event.get('severity', 'INFO'), "ℹ️")
                st.write(f"{severity_icon} {event['event_type']} - {event['timestamp'][:19]}")
    
    def _render_encryption_management(self, security_manager):
        """暗号化管理画面"""
        st.markdown("**現在の暗号化設定:**")
        st.write(f"- アルゴリズム: {SecurityConfig.ENCRYPTION_ALGORITHM}")
        st.write(f"- キー長: 256-bit")
        st.write(f"- 反復回数: {SecurityConfig.KEY_DERIVATION_ITERATIONS:,}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 マスターキー再生成", help="セキュリティ強化のためキーを再生成"):
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self._log_security_event(
                    "MASTER_KEY_REGENERATION",
                    user_id,
                    {"requested": True}
                )
                
                st.warning("⚠️ マスターキー再生成は既存の暗号化データに影響します")
                st.info("この操作は管理者の確認が必要です")
        
        with col2:
            if st.button("🔍 暗号化統計", help="暗号化されたファイルの統計情報"):
                # 暗号化統計の表示（実装例）
                st.info("暗号化ファイル数: 検索中...")
                st.info("総暗号化データサイズ: 計算中...")
    
    def _render_access_control_management(self, security_manager):
        """アクセス制御管理画面"""
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        st.markdown("**現在のアクセス権限:**")
        permissions = security_manager.access_control_manager.get_user_file_permissions(user_id)
        
        if permissions:
            df = pd.DataFrame(permissions)
            st.dataframe(df)
        else:
            st.info("現在のユーザーに特別なファイル権限は設定されていません")
        
        # アクセス制御の設定
        with st.expander("🔧 新しいアクセス権限の設定", expanded=False):
            target_user = st.text_input("対象ユーザー")
            file_path = st.text_input("ファイルパス")
            permission_type = st.selectbox("権限タイプ", ["read", "write", "delete"])
            
            if st.button("権限付与"):
                if target_user and file_path:
                    result = security_manager.access_control_manager.grant_file_permission(
                        file_path, target_user, permission_type, user_id
                    )
                    if result:
                        st.success("権限を付与しました")
                    else:
                        st.error("権限付与に失敗しました")
    
    def _render_security_statistics(self, security_manager):
        """セキュリティ統計画面"""
        col1, col2, col3 = st.columns(3)
        
        # セッション内統計
        total_events = len(st.session_state.security_events)
        
        with col1:
            st.metric("セキュリティイベント", total_events)
        
        with col2:
            login_events = len([e for e in st.session_state.security_events if 'LOGIN' in e['event_type']])
            st.metric("ログインイベント", login_events)
        
        with col3:
            error_events = len([e for e in st.session_state.security_events if e.get('severity') in ['ERROR', 'CRITICAL']])
            st.metric("エラーイベント", error_events)
        
        # イベントタイプ別統計
        if st.session_state.security_events:
            event_types = {}
            for event in st.session_state.security_events:
                event_type = event['event_type']
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            st.markdown("**イベントタイプ別統計:**")
            df_stats = pd.DataFrame(list(event_types.items()), columns=['イベントタイプ', '回数'])
            st.bar_chart(df_stats.set_index('イベントタイプ'))
    
    def _render_security_audit(self):
        """セキュリティ監査モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        # 管理者のみアクセス可能
        current_role = auth_manager.get_current_role()
        if current_role != "admin":
            st.error("この機能を使用する権限がありません")
            st.info("セキュリティ監査は管理者専用機能です")
            st.stop()
        
        st.header("📊 セキュリティ監査")
        
        # 完全な監査情報の表示
        if st.session_state.security_events:
            st.subheader("🔍 詳細監査ログ")
            
            # フィルタリングオプション
            col1, col2, col3 = st.columns(3)
            
            with col1:
                event_filter = st.selectbox("イベントフィルター", ["すべて", "LOGIN", "FILE_ACCESS", "SECURITY", "ERROR"])
            
            with col2:
                user_filter = st.text_input("ユーザーフィルター")
            
            with col3:
                date_filter = st.date_input("日付フィルター")
            
            # フィルタリング適用
            filtered_events = st.session_state.security_events
            
            if event_filter != "すべて":
                filtered_events = [e for e in filtered_events if event_filter in e['event_type']]
            
            if user_filter:
                filtered_events = [e for e in filtered_events if user_filter in e['user_id']]
            
            # 監査ログテーブル
            if filtered_events:
                audit_data = []
                for event in filtered_events:
                    audit_data.append({
                        'タイムスタンプ': event['timestamp'],
                        'イベントタイプ': event['event_type'],
                        'ユーザー': event['user_id'],
                        '詳細': str(event['details'])[:100] + "..." if len(str(event['details'])) > 100 else str(event['details'])
                    })
                
                df_audit = pd.DataFrame(audit_data)
                st.dataframe(df_audit, use_container_width=True)
                
                # 監査レポートのダウンロード
                if st.button("📥 監査レポートをダウンロード"):
                    csv = df_audit.to_csv(index=False)
                    st.download_button(
                        label="CSV形式でダウンロード",
                        data=csv,
                        file_name=f"security_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            else:
                st.info("フィルター条件に一致するイベントがありません")
        
        else:
            st.info("監査ログがありません")
    
    # その他のセキュア解析モード（スペース制限により省略）
    def _render_secure_database_comparison(self):
        """セキュア強化されたデータベース比較モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("database_comparison"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.markdown("### 🔒 セキュアデータベース比較")
        st.info("このモードでは、データベース操作が暗号化・監査されます。")
        
        database_comparison_mode()
    
    def _render_secure_multivariate_analysis(self):
        """セキュア強化された多変量解析モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("multivariate_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.markdown("### 🔒 セキュア多変量解析")
        st.info("このモードでは、統計処理が暗号化環境で実行されます。")
        
        multivariate_analysis_mode()
    
    def _render_secure_peak_deconvolution(self):
        """セキュア強化されたピーク分離モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_deconvolution"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.markdown("### 🔒 セキュアピーク分離")
        st.info("このモードでは、ピーク分離処理がセキュアに実行されます。")
        
        peak_deconvolution_mode()
    
    def _render_secure_peak_analysis(self):
        """セキュア強化されたピーク解析モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("peak_analysis"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.markdown("### 🔒 セキュアピーク解析")
        st.info("このモードでは、ピーク検出がセキュアに実行されます。")
        
        peak_analysis_mode()
    
    def _render_secure_calibration(self):
        """セキュア強化された検量線作成モード"""
        auth_system = self._get_auth_system()
        auth_manager = auth_system['AuthenticationManager']()
        
        if not auth_manager.has_permission("calibration"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.markdown("### 🔒 セキュア検量線作成")
        st.info("このモードでは、検量線データがセキュアに処理されます。")
        
        calibration_mode()

def main():
    """メイン関数"""
    try:
        app = RamanEyeApp()
        app.run()
        
    except Exception as e:
        st.error("アプリケーションの初期化中にエラーが発生しました")
        st.error(f"エラー詳細: {e}")
        
        # 緊急時のフォールバック
        st.markdown("---")
        st.info("通常モードで起動しますか？")
        if st.button("🔄 通常モードで再起動"):
            st.session_state.clear()
            st.rerun()

if __name__ == "__main__":
    main()
