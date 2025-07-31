# =============================================================================
# config.py - Production Configuration
# =============================================================================

import os
from datetime import timedelta

class SecurityConfig:
    """セキュリティ設定"""
    
    # パスワードポリシー
    PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', '8'))
    PASSWORD_MAX_LENGTH = int(os.getenv('PASSWORD_MAX_LENGTH', '128'))
    PASSWORD_REQUIRE_UPPERCASE = os.getenv('PASSWORD_REQUIRE_UPPERCASE', 'True').lower() == 'true'
    PASSWORD_REQUIRE_LOWERCASE = os.getenv('PASSWORD_REQUIRE_LOWERCASE', 'True').lower() == 'true'
    PASSWORD_REQUIRE_DIGITS = os.getenv('PASSWORD_REQUIRE_DIGITS', 'True').lower() == 'true'
    PASSWORD_REQUIRE_SPECIAL = os.getenv('PASSWORD_REQUIRE_SPECIAL', 'False').lower() == 'true'
    
    # アカウントロックアウト
    MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', '5'))
    LOCKOUT_DURATION_MINUTES = int(os.getenv('LOCKOUT_DURATION_MINUTES', '30'))
    
    # セッション管理
    SESSION_TIMEOUT_MINUTES = int(os.getenv('SESSION_TIMEOUT_MINUTES', '60'))
    SESSION_REFRESH_MINUTES = int(os.getenv('SESSION_REFRESH_MINUTES', '15'))
    
    # 暗号化
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    HASH_ALGORITHM = os.getenv('HASH_ALGORITHM', 'sha256')

class DatabaseConfig:
    """データベース設定"""
    
    # データベースタイプ
    DB_TYPE = os.getenv('DB_TYPE', 'session')  # session, postgresql, mysql
    
    # PostgreSQL設定
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = int(os.getenv('POSTGRES_PORT', '5432'))
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'ramaneye')
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', '')
    
    # MySQL設定
    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_DB = os.getenv('MYSQL_DB', 'ramaneye')
    MYSQL_USER = os.getenv('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')

class ApplicationConfig:
    """アプリケーション設定"""
    
    # アプリケーション基本設定
    APP_NAME = os.getenv('APP_NAME', 'RamanEye Easy Viewer')
    APP_VERSION = os.getenv('APP_VERSION', '2.0.0')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # ログ設定
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'ramaneye.log')
    
    # ファイルアップロード制限
    MAX_UPLOAD_SIZE_MB = int(os.getenv('MAX_UPLOAD_SIZE_MB', '100'))
    ALLOWED_FILE_EXTENSIONS = os.getenv('ALLOWED_FILE_EXTENSIONS', 'csv,txt,xlsx,pdf,docx').split(',')
    
    # UI設定
    THEME = os.getenv('THEME', 'light')
    SIDEBAR_STATE = os.getenv('SIDEBAR_STATE', 'expanded')

class LDAPConfig:
    """LDAP統合設定（企業向け）"""
    
    LDAP_ENABLED = os.getenv('LDAP_ENABLED', 'False').lower() == 'true'
    LDAP_SERVER = os.getenv('LDAP_SERVER', 'ldap://localhost:389')
    LDAP_BASE_DN = os.getenv('LDAP_BASE_DN', 'dc=company,dc=com')
    LDAP_USER_DN = os.getenv('LDAP_USER_DN', 'cn=admin,dc=company,dc=com')
    LDAP_PASSWORD = os.getenv('LDAP_PASSWORD', '')
    LDAP_USER_SEARCH = os.getenv('LDAP_USER_SEARCH', '(uid={username})')

# =============================================================================
# production_database.py - Production Database Integration
# =============================================================================

import psycopg2
import mysql.connector
from typing import Dict, List, Optional, Any
import json
import hashlib
from datetime import datetime

class ProductionUserDatabase:
    """本番環境向けデータベース"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self.connection = None
        self._connect()
    
    def _connect(self):
        """データベース接続"""
        if self.config.DB_TYPE == 'postgresql':
            self.connection = psycopg2.connect(
                host=self.config.POSTGRES_HOST,
                port=self.config.POSTGRES_PORT,
                database=self.config.POSTGRES_DB,
                user=self.config.POSTGRES_USER,
                password=self.config.POSTGRES_PASSWORD
            )
        elif self.config.DB_TYPE == 'mysql':
            self.connection = mysql.connector.connect(
                host=self.config.MYSQL_HOST,
                port=self.config.MYSQL_PORT,
                database=self.config.MYSQL_DB,
                user=self.config.MYSQL_USER,
                password=self.config.MYSQL_PASSWORD
            )
    
    def create_tables(self):
        """テーブル作成"""
        cursor = self.connection.cursor()
        
        # ユーザーテーブル
        user_table_sql = """
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            role VARCHAR(20) NOT NULL,
            email VARCHAR(255),
            full_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            failed_attempts INTEGER DEFAULT 0,
            locked_until TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
        """
        
        # セッションテーブル
        session_table_sql = """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50) NOT NULL,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ip_address VARCHAR(45),
            user_agent TEXT,
            is_active BOOLEAN DEFAULT TRUE
        )
        """
        
        # 監査ログテーブル
        audit_table_sql = """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id SERIAL PRIMARY KEY,
            username VARCHAR(50),
            action VARCHAR(100) NOT NULL,
            resource VARCHAR(100),
            ip_address VARCHAR(45),
            user_agent TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN DEFAULT TRUE,
            details TEXT
        )
        """
        
        cursor.execute(user_table_sql)
        cursor.execute(session_table_sql)
        cursor.execute(audit_table_sql)
        
        self.connection.commit()
        cursor.close()
    
    def create_user(self, username: str, password: str, role: str, email: str, full_name: str) -> tuple[bool, str]:
        """ユーザー作成"""
        cursor = self.connection.cursor()
        
        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            
            insert_sql = """
            INSERT INTO users (username, password_hash, role, email, full_name)
            VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_sql, (username, password_hash, role, email, full_name))
            self.connection.commit()
            
            # 監査ログ
            self._log_audit(username, 'USER_CREATED', f'user:{username}')
            
            return True, "ユーザーが正常に作成されました"
            
        except Exception as e:
            self.connection.rollback()
            return False, f"ユーザー作成エラー: {str(e)}"
        finally:
            cursor.close()
    
    def authenticate_user(self, username: str, password: str, ip_address: str = None) -> tuple[bool, str]:
        """ユーザー認証"""
        cursor = self.connection.cursor()
        
        try:
            # ユーザー情報取得
            select_sql = """
            SELECT password_hash, failed_attempts, locked_until, is_active
            FROM users WHERE username = %s
            """
            
            cursor.execute(select_sql, (username,))
            result = cursor.fetchone()
            
            if not result:
                self._log_audit(username, 'LOGIN_FAILED', 'authentication', ip_address, False, 'User not found')
                return False, "無効なユーザー名またはパスワードです"
            
            password_hash, failed_attempts, locked_until, is_active = result
            
            # アカウント状態チェック
            if not is_active:
                self._log_audit(username, 'LOGIN_FAILED', 'authentication', ip_address, False, 'Account disabled')
                return False, "このアカウントは無効化されています"
            
            # ロック状態チェック
            if locked_until and datetime.now() < locked_until:
                remaining = (locked_until - datetime.now()).seconds // 60
                self._log_audit(username, 'LOGIN_FAILED', 'authentication', ip_address, False, 'Account locked')
                return False, f"アカウントがロックされています。{remaining}分後に再試行してください"
            
            # パスワード確認
            input_hash = hashlib.sha256(password.encode()).hexdigest()
            
            if password_hash == input_hash:
                # 認証成功
                update_sql = """
                UPDATE users 
                SET last_login = CURRENT_TIMESTAMP, failed_attempts = 0, locked_until = NULL
                WHERE username = %s
                """
                cursor.execute(update_sql, (username,))
                self.connection.commit()
                
                self._log_audit(username, 'LOGIN_SUCCESS', 'authentication', ip_address, True)
                return True, "認証成功"
            
            else:
                # 認証失敗
                failed_attempts += 1
                
                if failed_attempts >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
                    # アカウントロック
                    lock_time = datetime.now() + timedelta(minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES)
                    update_sql = """
                    UPDATE users 
                    SET failed_attempts = %s, locked_until = %s
                    WHERE username = %s
                    """
                    cursor.execute(update_sql, (failed_attempts, lock_time, username))
                    self._log_audit(username, 'ACCOUNT_LOCKED', 'authentication', ip_address, False, f'Failed attempts: {failed_attempts}')
                    message = f"認証に{SecurityConfig.MAX_LOGIN_ATTEMPTS}回失敗しました。アカウントが{SecurityConfig.LOCKOUT_DURATION_MINUTES}分間ロックされます"
                else:
                    update_sql = "UPDATE users SET failed_attempts = %s WHERE username = %s"
                    cursor.execute(update_sql, (failed_attempts, username))
                    remaining = SecurityConfig.MAX_LOGIN_ATTEMPTS - failed_attempts
                    message = f"無効なパスワードです。残り{remaining}回の試行が可能です"
                
                self.connection.commit()
                self._log_audit(username, 'LOGIN_FAILED', 'authentication', ip_address, False, f'Invalid password, attempts: {failed_attempts}')
                return False, message
                
        except Exception as e:
            self.connection.rollback()
            self._log_audit(username, 'LOGIN_ERROR', 'authentication', ip_address, False, str(e))
            return False, f"認証エラー: {str(e)}"
        finally:
            cursor.close()
    
    def _log_audit(self, username: str, action: str, resource: str = None, ip_address: str = None, success: bool = True, details: str = None):
        """監査ログ記録"""
        cursor = self.connection.cursor()
        
        try:
            insert_sql = """
            INSERT INTO audit_logs (username, action, resource, ip_address, success, details)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_sql, (username, action, resource, ip_address, success, details))
            self.connection.commit()
            
        except Exception as e:
            print(f"Audit log error: {e}")
        finally:
            cursor.close()
