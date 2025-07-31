# -*- coding: utf-8 -*-
"""
ユーザー管理UI - 完全修正版
循環インポートを完全に排除したバージョン

Created for RamanEye Easy Viewer
@author: User Management UI System
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import re

def safe_datetime_format(date_value, format_str="%Y-%m-%d %H:%M", default="不明"):
    """安全な日時フォーマット変換"""
    if not date_value:
        return default
    
    try:
        if isinstance(date_value, str):
            dt = datetime.fromisoformat(date_value)
            return dt.strftime(format_str)
        elif isinstance(date_value, datetime):
            return date_value.strftime(format_str)
        else:
            return default
    except (ValueError, TypeError, AttributeError):
        return default

class LoginUI:
    """ログインUIクラス"""
    
    def __init__(self):
        # 遅延インポートで認証システムを取得
        from auth_system import AuthenticationManager
        self.auth_manager = AuthenticationManager()
    
    def render_login_page(self):
        """ログインページをレンダリング"""
        # カスタムCSSでログインページのスタイリング
        st.markdown(
            """
            <style>
            .login-container {
                max-width: 400px;
                margin: 0 auto;
                padding: 2rem;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                background-color: white;
            }
            .login-header {
                text-align: center;
                color: #1f77b4;
                margin-bottom: 2rem;
            }
            .demo-accounts {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 5px;
                margin-top: 1rem;
                border-left: 4px solid #17a2b8;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        # ヘッダー
        st.markdown(
            '<h1 class="login-header">🔐 RamanEye Login</h1>',
            unsafe_allow_html=True
        )
        
        # ログインフォーム
        with st.form("login_form"):
            username = st.text_input("ユーザー名", placeholder="ユーザー名を入力してください")
            password = st.text_input("パスワード", type="password", placeholder="パスワードを入力してください")
            
            col1, col2 = st.columns(2)
            with col1:
                login_button = st.form_submit_button("ログイン", type="primary", use_container_width=True)
            with col2:
                forgot_password = st.form_submit_button("パスワードを忘れた方", use_container_width=True)
        
        # ログイン処理
        if login_button:
            if username and password:
                success, message = self.auth_manager.login(username, password)
                if success:
                    st.success("ログインしました")
                    st.rerun()
                else:
                    st.error(message)
            else:
                st.error("ユーザー名とパスワードを入力してください")
        
        # パスワードリセット（デモ用）
        if forgot_password:
            st.info("デモ版では、管理者に直接お問い合わせください")
        
        # デモアカウント情報
        st.markdown(
            """
            <div class="demo-accounts">
            <h4>🔧 デモアカウント</h4>
            <p><strong>管理者:</strong> admin / Admin123!</p>
            <p><strong>分析者:</strong> analyst / Analyst123!</p>
            <p><strong>閲覧者:</strong> viewer / Viewer123!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

class UserManagementUI:
    """ユーザー管理UIクラス"""
    
    def __init__(self):
        # 遅延インポートで認証システムを取得
        from auth_system import AuthenticationManager, UserDatabase, UserRole, PasswordPolicy
        self.auth_manager = AuthenticationManager()
        self.db = UserDatabase()
        self.UserRole = UserRole
        self.PasswordPolicy = PasswordPolicy
    
    def render_user_management_page(self):
        """ユーザー管理ページをレンダリング"""
        # 権限チェックを手動で実行
        if not self.auth_manager.has_permission("user_management"):
            st.error("この機能を使用する権限がありません")
            st.stop()
        
        st.header("👥 ユーザー管理")
        
        # タブで機能を分割
        tab1, tab2, tab3 = st.tabs([
            "ユーザー一覧", 
            "新規ユーザー作成", 
            "一括操作"
        ])
        
        with tab1:
            self._render_user_list()
        
        with tab2:
            self._render_create_user()
        
        with tab3:
            self._render_bulk_operations()
    
    def _render_user_list(self):
        """ユーザー一覧表示"""
        st.subheader("📋 ユーザー一覧")
        
        users = self.db.list_users()
        
        if users:
            # データフレーム作成
            user_data = []
            for username, user_info in users.items():
                locked_status = "🔒 ロック中" if user_info.get("locked_until") else "✅ アクティブ"
                
                # 最終ログイン時刻の安全な処理
                last_login = safe_datetime_format(
                    user_info.get("last_login"), 
                    "%Y-%m-%d %H:%M", 
                    "未ログイン"
                )
                
                # 作成日の安全な処理
                created_at = safe_datetime_format(
                    user_info.get("created_at"), 
                    "%Y-%m-%d", 
                    "不明"
                )
                
                user_data.append({
                    "ユーザー名": username,
                    "フルネーム": user_info.get("full_name", ""),
                    "メールアドレス": user_info.get("email", ""),
                    "ロール": user_info["role"],
                    "ステータス": locked_status,
                    "最終ログイン": last_login,
                    "失敗回数": user_info.get("failed_attempts", 0),
                    "作成日": created_at
                })
            
            df = pd.DataFrame(user_data)
            st.dataframe(df, use_container_width=True)
            
            # ユーザー個別操作
            st.subheader("🔧 ユーザー個別操作")
            selected_user = st.selectbox("操作対象ユーザー", list(users.keys()))
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔓 ロック解除", key="unlock_user"):
                    if self.db.update_user(selected_user, {"locked_until": None, "failed_attempts": 0}):
                        st.success(f"{selected_user}のロックを解除しました")
                        st.rerun()
            
            with col2:
                if st.button("🔒 アカウントロック", key="lock_user"):
                    lock_time = (datetime.now() + pd.Timedelta(hours=24)).isoformat()
                    if self.db.update_user(selected_user, {"locked_until": lock_time}):
                        st.success(f"{selected_user}を24時間ロックしました")
                        st.rerun()
            
            with col3:
                if st.button("🔄 パスワードリセット", key="reset_password"):
                    # デモ用の簡易リセット
                    new_password = f"Reset123!{datetime.now().strftime('%m%d')}"
                    hashed = self.db._hash_password(new_password)
                    if self.db.update_user(selected_user, {"password_hash": hashed, "failed_attempts": 0}):
                        st.success(f"{selected_user}のパスワードをリセットしました")
                        st.info(f"新しいパスワード: {new_password}")
            
            with col4:
                if st.button("🗑️ ユーザー削除", key="delete_user"):
                    if selected_user != self.auth_manager.get_current_user():
                        if st.button("⚠️ 削除を確認", key="confirm_delete"):
                            if self.db.delete_user(selected_user):
                                st.success(f"{selected_user}を削除しました")
                                st.rerun()
                    else:
                        st.error("自分自身は削除できません")
        else:
            st.info("登録されているユーザーがありません")
    
    def _render_create_user(self):
        """新規ユーザー作成フォーム"""
        st.subheader("➕ 新規ユーザー作成")
        
        with st.form("create_user_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                username = st.text_input("ユーザー名 *", placeholder="半角英数字（3-20文字）")
                password = st.text_input("パスワード *", type="password", placeholder="パスワードを入力")
                password_confirm = st.text_input("パスワード確認 *", type="password", placeholder="パスワードを再入力")
            
            with col2:
                full_name = st.text_input("フルネーム *", placeholder="氏名を入力")
                email = st.text_input("メールアドレス *", placeholder="example@company.com")
                role = st.selectbox("ロール *", self.UserRole.get_all_roles())
            
            # パスワード強度表示
            if password:
                is_valid, errors = self.PasswordPolicy.validate_password(password)
                if is_valid:
                    st.success("✅ パスワード強度: 良好")
                else:
                    st.error("❌ パスワード強度の問題:")
                    for error in errors:
                        st.error(f"• {error}")
            
            create_button = st.form_submit_button("ユーザー作成", type="primary")
            
            if create_button:
                # バリデーション
                if not all([username, password, password_confirm, full_name, email]):
                    st.error("すべての必須項目を入力してください")
                elif password != password_confirm:
                    st.error("パスワードが一致しません")
                elif not re.match(r"^[a-zA-Z0-9_]{3,20}$", username):
                    st.error("ユーザー名は3-20文字の半角英数字と_のみ使用可能です")
                elif not re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", email):
                    st.error("有効なメールアドレスを入力してください")
                else:
                    success, message = self.db.create_user(username, password, role, email, full_name)
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
    
    def _render_bulk_operations(self):
        """一括操作"""
        st.subheader("📦 一括操作")
        
        users = self.db.list_users()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔓 一括ロック解除**")
            if st.button("全ユーザーのロックを解除"):
                count = 0
                for username in users:
                    if users[username].get("locked_until"):
                        self.db.update_user(username, {"locked_until": None, "failed_attempts": 0})
                        count += 1
                st.success(f"{count}人のユーザーのロックを解除しました")
                st.rerun()
        
        with col2:
            st.markdown("**📊 統計情報**")
            total_users = len(users)
            locked_users = sum(1 for u in users.values() if u.get("locked_until"))
            active_users = total_users - locked_users
            
            st.metric("総ユーザー数", total_users)
            st.metric("アクティブユーザー", active_users)
            st.metric("ロック中ユーザー", locked_users)

class ProfileUI:
    """プロファイルUIクラス"""
    
    def __init__(self):
        # 遅延インポートで認証システムを取得
        from auth_system import AuthenticationManager, UserDatabase, PasswordPolicy
        self.auth_manager = AuthenticationManager()
        self.db = UserDatabase()
        self.PasswordPolicy = PasswordPolicy
    
    def render_profile_page(self):
        """プロファイルページをレンダリング"""
        # 権限チェックを手動で実行
        if not self.auth_manager.is_authenticated():
            st.error("この機能を使用するにはログインが必要です")
            st.stop()
        
        st.header("👤 プロファイル管理")
        
        current_user = self.auth_manager.get_current_user()
        user_info = self.db.get_user(current_user)
        
        if not user_info:
            st.error("ユーザー情報が見つかりません")
            return
        
        # タブで機能を分割
        tab1, tab2 = st.tabs(["基本情報", "パスワード変更"])
        
        with tab1:
            self._render_basic_info(current_user, user_info)
        
        with tab2:
            self._render_password_change(current_user)
    
    def _render_basic_info(self, username, user_info):
        """基本情報表示・編集"""
        st.subheader("📋 基本情報")
        
        # 読み取り専用情報
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("ユーザー名", value=username, disabled=True)
            st.text_input("ロール", value=user_info["role"], disabled=True)
            
        with col2:
            # 作成日の安全な処理
            created_at = safe_datetime_format(user_info.get("created_at"), "%Y-%m-%d %H:%M", "不明")
            st.text_input("作成日", value=created_at, disabled=True)
            
            # 最終ログインの安全な処理
            last_login = safe_datetime_format(user_info.get("last_login"), "%Y-%m-%d %H:%M", "未ログイン")
            st.text_input("最終ログイン", value=last_login, disabled=True)
        
        # 編集可能情報
        st.subheader("✏️ 編集可能情報")
        
        with st.form("update_profile_form"):
            new_full_name = st.text_input("フルネーム", value=user_info.get("full_name", ""))
            new_email = st.text_input("メールアドレス", value=user_info.get("email", ""))
            
            if st.form_submit_button("情報を更新"):
                updates = {
                    "full_name": new_full_name,
                    "email": new_email
                }
                
                if self.db.update_user(username, updates):
                    st.success("プロファイルを更新しました")
                    st.rerun()
                else:
                    st.error("更新に失敗しました")
    
    def _render_password_change(self, username):
        """パスワード変更"""
        st.subheader("🔑 パスワード変更")
        
        with st.form("change_password_form"):
            current_password = st.text_input("現在のパスワード", type="password")
            new_password = st.text_input("新しいパスワード", type="password")
            confirm_password = st.text_input("新しいパスワード（確認）", type="password")
            
            # パスワード強度表示
            if new_password:
                is_valid, errors = self.PasswordPolicy.validate_password(new_password)
                if is_valid:
                    st.success("✅ パスワード強度: 良好")
                else:
                    st.error("❌ パスワード強度の問題:")
                    for error in errors:
                        st.error(f"• {error}")
            
            if st.form_submit_button("パスワードを変更"):
                if not all([current_password, new_password, confirm_password]):
                    st.error("すべての項目を入力してください")
                elif new_password != confirm_password:
                    st.error("新しいパスワードが一致しません")
                else:
                    success, message = self.db.change_password(username, current_password, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

# ヘッダーコンポーネント
def render_authenticated_header():
    """認証後のヘッダー"""
    # 遅延インポートで認証システムを取得
    from auth_system import AuthenticationManager
    auth_manager = AuthenticationManager()
    
    if auth_manager.is_authenticated():
        current_user = auth_manager.get_current_user()
        current_role = auth_manager.get_current_role()
        
        # ヘッダーバー
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            role_emoji = {"admin": "👑", "analyst": "🔬", "viewer": "👁️"}
            st.write(f"**{role_emoji.get(current_role, '👤')} {current_user}** ({current_role})")
        
        with col2:
            if st.button("👤 プロファイル"):
                st.session_state.show_profile = True
        
        with col3:
            if st.button("🚪 ログアウト"):
                auth_manager.logout()
                st.rerun()
        
        st.divider()
