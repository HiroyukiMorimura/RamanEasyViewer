# -*- coding: utf-8 -*-
"""
認証・認可システム
ユーザー管理、認証、権限管理を統合的に管理

Created for RamanEye Easy Viewer
@author: Enhanced Authentication System
"""

import streamlit as st
import hashlib
import re
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Optional, Any
import time

# ユーザーロールの定義
class UserRole:
    ADMIN = "admin"
    ANALYST = "analyst" 
    VIEWER = "viewer"
    
    @classmethod
    def get_all_roles(cls):
        return [cls.ADMIN, cls.ANALYST, cls.VIEWER]
    
    @classmethod
    def get_role_permissions(cls, role: str) -> Dict[str, bool]:
        """ロール別権限マトリックス"""
        permissions = {
            cls.ADMIN: {
                "user_management": True,
                "spectrum_analysis": True,
                "peak_analysis": True,
                "peak_deconvolution": True,
                "multivariate_analysis": True,
                "calibration": True,
                "database_comparison": True,
                "peak_ai_analysis": True,
                "download_results": True,
                "upload_files": True,
                "system_settings": True
            },
            cls.ANALYST: {
                "user_management": False,
                "spectrum_analysis": True,
                "peak_analysis": True,
                "peak_deconvolution": True,
                "multivariate_analysis": True,
                "calibration": True,
                "database_comparison": True,
                "peak_ai_analysis": True,
                "download_results": True,
                "upload_files": True,
                "system_settings": False
            },
            cls.VIEWER: {
                "user_management": False,
                "spectrum_analysis": True,
                "peak_analysis": False,
                "peak_deconvolution": False,
                "multivariate_analysis": False,
                "calibration": False,
                "database_comparison": True,
                "peak_ai_analysis": False,
                "download_results": False,
                "upload_files": False,
                "system_settings": False
            }
        }
        return permissions.get(role, {})

class PasswordPolicy:
    """パスワード強度ポリシー"""
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL = False
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def validate_password(cls, password: str) -> tuple[bool, List[str]]:
        """パスワード強度を検証"""
        errors = []
        
        if len(password) < cls.MIN_LENGTH:
            errors.append(f"パスワードは{cls.MIN_LENGTH}文字以上である必要があります")
        
        if len(password) > cls.MAX_LENGTH:
            errors.append(f"パスワードは{cls.MAX_LENGTH}文字以下である必要があります")
        
        if cls.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            errors.append("パスワードには大文字を含める必要があります")
        
        if cls.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            errors.append("パスワードには小文字を含める必要があります")
        
        if cls.REQUIRE_DIGITS and not re.search(r'\d', password):
            errors.append("パスワードには数字を含める必要があります")
        
        if cls.REQUIRE_SPECIAL and not re.search(f'[{re.escape(cls.SPECIAL_CHARS)}]', password):
            errors.append(f"パスワードには特殊文字({cls.SPECIAL_CHARS})を含める必要があります")
        
        return len(errors) == 0, errors

class UserDatabase:
    """ユーザーデータベース（簡易版）"""
    
    def __init__(self):
        if "user_db" not in st.session_state:
            # デフォルトアカウントの作成
            st.session_state.user_db = {
                "admin": {
                    "password_hash": self._hash_password("Admin123!"),
                    "role": UserRole.ADMIN,
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "failed_attempts": 0,
                    "locked_until": None,
                    "email": "admin@ramaneye.com",
                    "full_name": "System Administrator"
                },
                "analyst": {
                    "password_hash": self._hash_password("Analyst123!"),
                    "role": UserRole.ANALYST,
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "failed_attempts": 0,
                    "locked_until": None,
                    "email": "analyst@ramaneye.com",
                    "full_name": "Data Analyst"
                },
                "viewer": {
                    "password_hash": self._hash_password("Viewer123!"),
                    "role": UserRole.VIEWER,
                    "created_at": datetime.now().isoformat(),
                    "last_login": None,
                    "failed_attempts": 0,
                    "locked_until": None,
                    "email": "viewer@ramaneye.com",
                    "full_name": "Data Viewer"
                }
            }
    
    def _hash_password(self, password: str) -> str:
        """パスワードのハッシュ化"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username: str, password: str, role: str, email: str, full_name: str) -> tuple[bool, str]:
        """新規ユーザー作成"""
        if username in st.session_state.user_db:
            return False, "ユーザー名が既に存在します"
        
        # パスワード強度チェック
        is_valid, errors = PasswordPolicy.validate_password(password)
        if not is_valid:
            return False, "\n".join(errors)
        
        # ユーザー作成
        st.session_state.user_db[username] = {
            "password_hash": self._hash_password(password),
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "failed_attempts": 0,
            "locked_until": None,
            "email": email,
            "full_name": full_name
        }
        
        return True, "ユーザーが正常に作成されました"
    
    def authenticate_user(self, username: str, password: str) -> tuple[bool, str]:
        """ユーザー認証"""
        if username not in st.session_state.user_db:
            return False, "無効なユーザー名またはパスワードです"
        
        user = st.session_state.user_db[username]
        
        # アカウントロック確認
        if user.get("locked_until"):
            lock_time = datetime.fromisoformat(user["locked_until"])
            if datetime.now() < lock_time:
                remaining = (lock_time - datetime.now()).seconds // 60
                return False, f"アカウントがロックされています。{remaining}分後に再試行してください"
            else:
                # ロック期間終了
                user["locked_until"] = None
                user["failed_attempts"] = 0
        
        # パスワード確認
        if user["password_hash"] == self._hash_password(password):
            # 認証成功
            user["last_login"] = datetime.now().isoformat()
            user["failed_attempts"] = 0
            user["locked_until"] = None
            return True, "認証成功"
        else:
            # 認証失敗
            user["failed_attempts"] += 1
            
            # アカウントロック（5回失敗で30分ロック）
            if user["failed_attempts"] >= 5:
                user["locked_until"] = (datetime.now() + timedelta(minutes=30)).isoformat()
                return False, "認証に5回失敗しました。アカウントが30分間ロックされます"
            
            remaining_attempts = 5 - user["failed_attempts"]
            return False, f"無効なパスワードです。残り{remaining_attempts}回の試行が可能です"
    
    def get_user(self, username: str) -> Optional[Dict]:
        """ユーザー情報取得"""
        return st.session_state.user_db.get(username)
    
    def update_user(self, username: str, updates: Dict) -> bool:
        """ユーザー情報更新"""
        if username not in st.session_state.user_db:
            return False
        st.session_state.user_db[username].update(updates)
        return True
    
    def delete_user(self, username: str) -> bool:
        """ユーザー削除"""
        if username not in st.session_state.user_db:
            return False
        del st.session_state.user_db[username]
        return True
    
    def list_users(self) -> Dict:
        """全ユーザー一覧取得"""
        return st.session_state.user_db
    
    def change_password(self, username: str, old_password: str, new_password: str) -> tuple[bool, str]:
        """パスワード変更"""
        if username not in st.session_state.user_db:
            return False, "ユーザーが存在しません"
        
        user = st.session_state.user_db[username]
        
        # 現在のパスワード確認
        if user["password_hash"] != self._hash_password(old_password):
            return False, "現在のパスワードが正しくありません"
        
        # 新しいパスワードの強度チェック
        is_valid, errors = PasswordPolicy.validate_password(new_password)
        if not is_valid:
            return False, "\n".join(errors)
        
        # パスワード更新
        user["password_hash"] = self._hash_password(new_password)
        return True, "パスワードが正常に変更されました"

class AuthenticationManager:
    """認証管理クラス"""
    
    def __init__(self):
        self.db = UserDatabase()
        
        # セッション状態の初期化
        if "authenticated" not in st.session_state:
            st.session_state.authenticated = False
        if "username" not in st.session_state:
            st.session_state.username = None
        if "user_role" not in st.session_state:
            st.session_state.user_role = None
        if "session_start" not in st.session_state:
            st.session_state.session_start = None
    
    def login(self, username: str, password: str) -> tuple[bool, str]:
        """ログイン処理"""
        success, message = self.db.authenticate_user(username, password)
        
        if success:
            user = self.db.get_user(username)
            st.session_state.authenticated = True
            st.session_state.username = username
            st.session_state.user_role = user["role"]
            st.session_state.session_start = datetime.now()
            
        return success, message
    
    def logout(self):
        """ログアウト処理"""
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.session_start = None
    
    def is_authenticated(self) -> bool:
        """認証状態確認"""
        return st.session_state.get("authenticated", False)
    
    def get_current_user(self) -> Optional[str]:
        """現在のユーザー名取得"""
        return st.session_state.get("username")
    
    def get_current_role(self) -> Optional[str]:
        """現在のユーザーロール取得"""
        return st.session_state.get("user_role")
    
    def has_permission(self, permission: str) -> bool:
        """権限確認"""
        if not self.is_authenticated():
            return False
        
        role = self.get_current_role()
        permissions = UserRole.get_role_permissions(role)
        return permissions.get(permission, False)
    
    def check_session_timeout(self, timeout_minutes: int = 60) -> bool:
        """セッションタイムアウト確認"""
        if not self.is_authenticated():
            return False
        
        session_start = st.session_state.get("session_start")
        if session_start:
            if datetime.now() - session_start > timedelta(minutes=timeout_minutes):
                self.logout()
                return False
        
        return True

# デコレータベースの認可システム
def require_auth(func):
    """認証が必要な機能に使用するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        auth_manager = AuthenticationManager()
        
        if not auth_manager.is_authenticated():
            st.error("この機能を使用するにはログインが必要です")
            st.stop()
        
        # セッションタイムアウト確認
        if not auth_manager.check_session_timeout():
            st.error("セッションがタイムアウトしました。再度ログインしてください")
            st.stop()
        
        return func(*args, **kwargs)
    return wrapper

def require_permission(permission: str):
    """特定の権限が必要な機能に使用するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_manager = AuthenticationManager()
            
            if not auth_manager.is_authenticated():
                st.error("この機能を使用するにはログインが必要です")
                st.stop()
            
            if not auth_manager.has_permission(permission):
                st.error("この機能を使用する権限がありません")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_role(required_role: str):
    """特定のロールが必要な機能に使用するデコレータ"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            auth_manager = AuthenticationManager()
            
            if not auth_manager.is_authenticated():
                st.error("この機能を使用するにはログインが必要です")
                st.stop()
            
            current_role = auth_manager.get_current_role()
            if current_role != required_role:
                st.error(f"この機能は{required_role}ロールのみ使用可能です")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator
