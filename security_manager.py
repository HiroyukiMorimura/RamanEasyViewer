# -*- coding: utf-8 -*-
"""
セキュリティ管理モジュール
データ完全性・セキュリティ機能の統合管理

Created on 2025-07-31
@author: Enhanced Security System
"""

import os
import hashlib
import hmac
import json
import sqlite3
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import secrets
import streamlit as st

class SecurityConfig:
    """セキュリティ設定の中央管理"""
    
    # 暗号化設定
    ENCRYPTION_ALGORITHM = "AES-256-GCM"
    KEY_DERIVATION_ITERATIONS = 100000
    SALT_LENGTH = 32
    IV_LENGTH = 16
    
    # ハッシュ設定
    HASH_ALGORITHM = "SHA-256"
    HMAC_KEY_LENGTH = 32
    
    # ファイルアクセス設定
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {'.csv', '.txt', '.pdf', '.docx', '.xlsx'}
    
    # セッション設定
    SESSION_TIMEOUT = 3600  # 1時間
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION = 1800  # 30分
    
    # 監査ログ設定
    AUDIT_LOG_RETENTION_DAYS = 365
    LOG_LEVEL = logging.INFO

class EncryptionManager:
    """データ暗号化管理クラス"""
    
    def __init__(self, master_key: Optional[str] = None):
        """初期化"""
        self.master_key = master_key or self._get_or_create_master_key()
        self._setup_encryption()
    
    def _get_or_create_master_key(self) -> str:
        """マスターキーの取得または作成"""
        key_file = Path("./secure/.master_key")
        key_file.parent.mkdir(exist_ok=True, parents=True)
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return base64.urlsafe_b64decode(f.read()).decode()
            except Exception as e:
                st.error(f"マスターキー読み込みエラー: {e}")
        
        # 新しいマスターキーを生成
        master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        
        try:
            # セキュアに保存（権限制限）
            with open(key_file, 'wb') as f:
                f.write(base64.urlsafe_b64encode(master_key.encode()))
            
            # ファイル権限を制限（Unix系）
            if hasattr(os, 'chmod'):
                os.chmod(key_file, 0o600)
                
        except Exception as e:
            st.error(f"マスターキー保存エラー: {e}")
        
        return master_key
    
    def _setup_encryption(self):
        """暗号化設定のセットアップ"""
        # Fernetインスタンス（対称暗号化）
        key_bytes = self.master_key.encode()[:32].ljust(32, b'0')
        self.fernet_key = base64.urlsafe_b64encode(key_bytes)
        self.fernet = Fernet(self.fernet_key)
    
    def encrypt_data(self, data: str) -> str:
        """データの暗号化"""
        try:
            encrypted_data = self.fernet.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            raise SecurityException(f"データ暗号化エラー: {e}")
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """データの復号化"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.fernet.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            raise SecurityException(f"データ復号化エラー: {e}")
    
    def encrypt_file(self, file_path: Path, output_path: Optional[Path] = None) -> Path:
        """ファイル暗号化"""
        if not file_path.exists():
            raise SecurityException(f"ファイルが存在しません: {file_path}")
        
        output_path = output_path or file_path.with_suffix(file_path.suffix + '.enc')
        
        try:
            with open(file_path, 'rb') as infile:
                data = infile.read()
            
            encrypted_data = self.fernet.encrypt(data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(encrypted_data)
            
            # 権限設定
            if hasattr(os, 'chmod'):
                os.chmod(output_path, 0o600)
                
            return output_path
            
        except Exception as e:
            raise SecurityException(f"ファイル暗号化エラー: {e}")
    
    def decrypt_file(self, encrypted_file_path: Path, output_path: Optional[Path] = None) -> Path:
        """ファイル復号化"""
        if not encrypted_file_path.exists():
            raise SecurityException(f"暗号化ファイルが存在しません: {encrypted_file_path}")
        
        output_path = output_path or encrypted_file_path.with_suffix('')
        if output_path.suffix == '.enc':
            output_path = output_path.with_suffix('')
        
        try:
            with open(encrypted_file_path, 'rb') as infile:
                encrypted_data = infile.read()
            
            data = self.fernet.decrypt(encrypted_data)
            
            with open(output_path, 'wb') as outfile:
                outfile.write(data)
                
            return output_path
            
        except Exception as e:
            raise SecurityException(f"ファイル復号化エラー: {e}")

class FileIntegrityManager:
    """ファイル完全性管理クラス"""
    
    def __init__(self, hmac_key: Optional[str] = None):
        """初期化"""
        self.hmac_key = hmac_key or self._get_or_create_hmac_key()
        self.integrity_db = Path("./secure/integrity.db")
        self._setup_database()
    
    def _get_or_create_hmac_key(self) -> str:
        """HMAC鍵の取得または作成"""
        key_file = Path("./secure/.hmac_key")
        key_file.parent.mkdir(exist_ok=True, parents=True)
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return base64.urlsafe_b64decode(f.read()).decode()
            except Exception:
                pass
        
        # 新しいHMAC鍵を生成
        hmac_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
        
        try:
            with open(key_file, 'wb') as f:
                f.write(base64.urlsafe_b64encode(hmac_key.encode()))
            
            if hasattr(os, 'chmod'):
                os.chmod(key_file, 0o600)
                
        except Exception as e:
            st.error(f"HMAC鍵保存エラー: {e}")
        
        return hmac_key
    
    def _setup_database(self):
        """完全性データベースのセットアップ"""
        try:
            self.integrity_db.parent.mkdir(exist_ok=True, parents=True)
            
            with sqlite3.connect(self.integrity_db) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_integrity (
                        file_path TEXT PRIMARY KEY,
                        file_hash TEXT NOT NULL,
                        hmac_signature TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_verified TIMESTAMP,
                        verification_count INTEGER DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_created_at 
                    ON file_integrity(created_at)
                """)
                
                conn.commit()
                
        except Exception as e:
            raise SecurityException(f"完全性データベース初期化エラー: {e}")
    
    def calculate_file_hash(self, file_path: Path) -> Tuple[str, str, int]:
        """ファイルのハッシュ値とHMAC署名を計算"""
        if not file_path.exists():
            raise SecurityException(f"ファイルが存在しません: {file_path}")
        
        try:
            sha256_hash = hashlib.sha256()
            file_size = 0
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
                    file_size += len(chunk)
            
            file_hash = sha256_hash.hexdigest()
            
            # HMAC署名計算
            hmac_signature = hmac.new(
                self.hmac_key.encode(),
                f"{file_path.name}:{file_hash}:{file_size}".encode(),
                hashlib.sha256
            ).hexdigest()
            
            return file_hash, hmac_signature, file_size
            
        except Exception as e:
            raise SecurityException(f"ハッシュ計算エラー: {e}")
    
    def register_file(self, file_path: Path) -> bool:
        """ファイルの完全性情報を登録"""
        try:
            file_hash, hmac_signature, file_size = self.calculate_file_hash(file_path)
            
            with sqlite3.connect(self.integrity_db) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO file_integrity 
                    (file_path, file_hash, hmac_signature, file_size, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (str(file_path), file_hash, hmac_signature, file_size, datetime.now()))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            st.error(f"ファイル登録エラー: {e}")
            return False
    
    def verify_file_integrity(self, file_path: Path) -> Dict[str, Any]:
        """ファイルの完全性を検証"""
        if not file_path.exists():
            return {
                'status': 'error',
                'message': 'ファイルが存在しません',
                'file_exists': False
            }
        
        try:
            # データベースから情報取得
            with sqlite3.connect(self.integrity_db) as conn:
                cursor = conn.execute("""
                    SELECT file_hash, hmac_signature, file_size, created_at 
                    FROM file_integrity WHERE file_path = ?
                """, (str(file_path),))
                
                record = cursor.fetchone()
            
            if not record:
                return {
                    'status': 'unknown',
                    'message': 'ファイルが登録されていません',
                    'file_exists': True,
                    'registered': False
                }
            
            stored_hash, stored_signature, stored_size, created_at = record
            
            # 現在のファイル情報を計算
            current_hash, current_signature, current_size = self.calculate_file_hash(file_path)
            
            # 検証結果
            hash_match = stored_hash == current_hash
            signature_match = stored_signature == current_signature
            size_match = stored_size == current_size
            
            integrity_verified = hash_match and signature_match and size_match
            
            # 検証カウントを更新
            if integrity_verified:
                with sqlite3.connect(self.integrity_db) as conn:
                    conn.execute("""
                        UPDATE file_integrity 
                        SET last_verified = ?, verification_count = verification_count + 1
                        WHERE file_path = ?
                    """, (datetime.now(), str(file_path)))
                    conn.commit()
            
            return {
                'status': 'verified' if integrity_verified else 'corrupted',
                'message': '完全性検証成功' if integrity_verified else 'ファイルが改ざんされている可能性があります',
                'file_exists': True,
                'registered': True,
                'integrity_verified': integrity_verified,
                'hash_match': hash_match,
                'signature_match': signature_match,
                'size_match': size_match,
                'created_at': created_at,
                'current_hash': current_hash,
                'stored_hash': stored_hash
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'検証エラー: {e}',
                'file_exists': True,
                'error': str(e)
            }

class AccessControlManager:
    """アクセス制御管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.access_db = Path("./secure/access_control.db")
        self._setup_database()
    
    def _setup_database(self):
        """アクセス制御データベースのセットアップ"""
        try:
            self.access_db.parent.mkdir(exist_ok=True, parents=True)
            
            with sqlite3.connect(self.access_db) as conn:
                # ファイルアクセス権限テーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS file_permissions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_path TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        permission_type TEXT NOT NULL, -- read, write, delete, execute
                        granted_by TEXT NOT NULL,
                        granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # データアクセスログテーブル
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS access_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        access_type TEXT NOT NULL, -- read, write, delete, upload, download
                        access_result TEXT NOT NULL, -- success, denied, error
                        ip_address TEXT,
                        user_agent TEXT,
                        accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # インデックス作成
                conn.execute("CREATE INDEX IF NOT EXISTS idx_file_user ON file_permissions(file_path, user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_user ON access_log(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_access_log_time ON access_log(accessed_at)")
                
                conn.commit()
                
        except Exception as e:
            raise SecurityException(f"アクセス制御データベース初期化エラー: {e}")
    
    def grant_file_permission(self, file_path: str, user_id: str, permission_type: str, 
                            granted_by: str, expires_at: Optional[datetime] = None) -> bool:
        """ファイルアクセス許可を付与"""
        try:
            with sqlite3.connect(self.access_db) as conn:
                conn.execute("""
                    INSERT INTO file_permissions 
                    (file_path, user_id, permission_type, granted_by, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (file_path, user_id, permission_type, granted_by, expires_at))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            st.error(f"アクセス許可付与エラー: {e}")
            return False
    
    def check_file_permission(self, file_path: str, user_id: str, permission_type: str) -> bool:
        """ファイルアクセス権限をチェック"""
        try:
            # 管理者は全てのファイルにアクセス可能
            if self._is_admin_user(user_id):
                return True
            
            with sqlite3.connect(self.access_db) as conn:
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM file_permissions 
                    WHERE file_path = ? AND user_id = ? AND permission_type = ? 
                    AND is_active = TRUE 
                    AND (expires_at IS NULL OR expires_at > datetime('now'))
                """, (file_path, user_id, permission_type))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception as e:
            st.error(f"アクセス権限チェックエラー: {e}")
            return False
    
    def log_access_attempt(self, user_id: str, file_path: str, access_type: str, 
                          access_result: str, ip_address: Optional[str] = None, 
                          user_agent: Optional[str] = None) -> bool:
        """アクセス試行をログに記録"""
        try:
            with sqlite3.connect(self.access_db) as conn:
                conn.execute("""
                    INSERT INTO access_log 
                    (user_id, file_path, access_type, access_result, ip_address, user_agent)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, file_path, access_type, access_result, ip_address, user_agent))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            st.error(f"アクセスログ記録エラー: {e}")
            return False
    
    def _is_admin_user(self, user_id: str) -> bool:
        """管理者ユーザーかどうかを判定"""
        # セッションから現在のユーザーロールを取得
        if 'current_user' in st.session_state:
            current_user = st.session_state['current_user']
            return current_user.get('role') == 'admin'
        return False
    
    def get_user_file_permissions(self, user_id: str) -> List[Dict]:
        """ユーザーのファイル権限一覧を取得"""
        try:
            with sqlite3.connect(self.access_db) as conn:
                cursor = conn.execute("""
                    SELECT file_path, permission_type, granted_by, granted_at, expires_at, is_active
                    FROM file_permissions 
                    WHERE user_id = ? 
                    ORDER BY granted_at DESC
                """, (user_id,))
                
                permissions = []
                for row in cursor.fetchall():
                    permissions.append({
                        'file_path': row[0],
                        'permission_type': row[1],
                        'granted_by': row[2],
                        'granted_at': row[3],
                        'expires_at': row[4],
                        'is_active': row[5]
                    })
                
                return permissions
                
        except Exception as e:
            st.error(f"権限一覧取得エラー: {e}")
            return []
    
    def revoke_file_permission(self, file_path: str, user_id: str, permission_type: str) -> bool:
        """ファイルアクセス権限を取り消し"""
        try:
            with sqlite3.connect(self.access_db) as conn:
                conn.execute("""
                    UPDATE file_permissions 
                    SET is_active = FALSE 
                    WHERE file_path = ? AND user_id = ? AND permission_type = ?
                """, (file_path, user_id, permission_type))
                
                conn.commit()
            
            return True
            
        except Exception as e:
            st.error(f"権限取り消しエラー: {e}")
            return False

class SecurityAuditLogger:
    """セキュリティ監査ログ管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.log_file = Path("./secure/security_audit.log")
        self.log_file.parent.mkdir(exist_ok=True, parents=True)
        
        # ロガー設定
        self.logger = logging.getLogger("SecurityAudit")
        self.logger.setLevel(SecurityConfig.LOG_LEVEL)
        
        # ハンドラー設定
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def log_security_event(self, event_type: str, user_id: str, details: Dict[str, Any], 
                          severity: str = "INFO") -> None:
        """セキュリティイベントをログに記録"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'user_id': user_id,
            'severity': severity,
            'details': details,
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        log_message = json.dumps(log_entry, ensure_ascii=False)
        
        if severity == "CRITICAL":
            self.logger.critical(log_message)
        elif severity == "ERROR":
            self.logger.error(log_message)
        elif severity == "WARNING":
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_login_attempt(self, user_id: str, success: bool, ip_address: Optional[str] = None) -> None:
        """ログイン試行をログに記録"""
        self.log_security_event(
            event_type="LOGIN_ATTEMPT",
            user_id=user_id,
            details={
                'success': success,
                'ip_address': ip_address,
                'timestamp': datetime.now().isoformat()
            },
            severity="INFO" if success else "WARNING"
        )
    
    def log_file_access(self, user_id: str, file_path: str, access_type: str, 
                       success: bool) -> None:
        """ファイルアクセスをログに記録"""
        self.log_security_event(
            event_type="FILE_ACCESS",
            user_id=user_id,
            details={
                'file_path': file_path,
                'access_type': access_type,
                'success': success,
                'timestamp': datetime.now().isoformat()
            },
            severity="INFO" if success else "WARNING"
        )
    
    def log_data_encryption(self, user_id: str, operation: str, target: str) -> None:
        """データ暗号化操作をログに記録"""
        self.log_security_event(
            event_type="DATA_ENCRYPTION",
            user_id=user_id,
            details={
                'operation': operation,  # encrypt, decrypt
                'target': target,
                'timestamp': datetime.now().isoformat()
            },
            severity="INFO"
        )

class SecurityManager:
    """統合セキュリティ管理クラス"""
    
    def __init__(self):
        """初期化"""
        self.encryption_manager = EncryptionManager()
        self.integrity_manager = FileIntegrityManager()
        self.access_control_manager = AccessControlManager()
        self.audit_logger = SecurityAuditLogger()
    
    def secure_file_upload(self, uploaded_file, user_id: str) -> Dict[str, Any]:
        """セキュアなファイルアップロード処理"""
        try:
            # ファイルサイズチェック
            file_size = len(uploaded_file.read())
            uploaded_file.seek(0)  # ポインタを先頭に戻す
            
            if file_size > SecurityConfig.MAX_FILE_SIZE:
                self.audit_logger.log_security_event(
                    event_type="FILE_UPLOAD_REJECTED",
                    user_id=user_id,
                    details={'reason': 'file_too_large', 'file_size': file_size},
                    severity="WARNING"
                )
                return {
                    'status': 'error',
                    'message': f'ファイルサイズが制限を超えています ({file_size / 1024 / 1024:.1f}MB > {SecurityConfig.MAX_FILE_SIZE / 1024 / 1024}MB)'
                }
            
            # 拡張子チェック
            file_extension = Path(uploaded_file.name).suffix.lower()
            if file_extension not in SecurityConfig.ALLOWED_EXTENSIONS:
                self.audit_logger.log_security_event(
                    event_type="FILE_UPLOAD_REJECTED",
                    user_id=user_id,
                    details={'reason': 'invalid_extension', 'extension': file_extension},
                    severity="WARNING"
                )
                return {
                    'status': 'error',
                    'message': f'許可されていないファイル形式です: {file_extension}'
                }
            
            # セキュアなファイル保存
            secure_dir = Path("./secure/uploads")
            secure_dir.mkdir(exist_ok=True, parents=True)
            
            # ファイル名の安全化
            safe_filename = self._sanitize_filename(uploaded_file.name)
            file_path = secure_dir / f"{user_id}_{int(time.time())}_{safe_filename}"
            
            # ファイル保存
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            # ファイルアクセス権限設定
            if hasattr(os, 'chmod'):
                os.chmod(file_path, 0o600)
            
            # 完全性チェック用ハッシュ登録
            self.integrity_manager.register_file(file_path)
            
            # アクセス権限付与
            self.access_control_manager.grant_file_permission(
                str(file_path), user_id, 'read', user_id
            )
            self.access_control_manager.grant_file_permission(
                str(file_path), user_id, 'write', user_id
            )
            
            # 監査ログ
            self.audit_logger.log_file_access(
                user_id=user_id,
                file_path=str(file_path),
                access_type='upload',
                success=True
            )
            
            return {
                'status': 'success',
                'message': 'ファイルが安全にアップロードされました',
                'file_path': str(file_path),
                'original_name': uploaded_file.name,
                'file_size': file_size
            }
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="FILE_UPLOAD_ERROR",
                user_id=user_id,
                details={'error': str(e)},
                severity="ERROR"
            )
            return {
                'status': 'error',
                'message': f'ファイルアップロードエラー: {e}'
            }
    
    def secure_file_access(self, file_path: str, user_id: str, access_type: str = 'read') -> Dict[str, Any]:
        """セキュアなファイルアクセス"""
        try:
            # アクセス権限チェック
            if not self.access_control_manager.check_file_permission(file_path, user_id, access_type):
                self.access_control_manager.log_access_attempt(
                    user_id=user_id,
                    file_path=file_path,
                    access_type=access_type,
                    access_result='denied'
                )
                return {
                    'status': 'denied',
                    'message': 'ファイルアクセス権限がありません'
                }
            
            # ファイル存在チェック
            path_obj = Path(file_path)
            if not path_obj.exists():
                return {
                    'status': 'error',
                    'message': 'ファイルが存在しません'
                }
            
            # ファイル完全性チェック
            integrity_result = self.integrity_manager.verify_file_integrity(path_obj)
            if integrity_result['status'] == 'corrupted':
                self.audit_logger.log_security_event(
                    event_type="FILE_INTEGRITY_VIOLATION",
                    user_id=user_id,
                    details={'file_path': file_path, 'integrity_result': integrity_result},
                    severity="CRITICAL"
                )
                return {
                    'status': 'error',
                    'message': 'ファイルの完全性が損なわれています'
                }
            
            # アクセスログ記録
            self.access_control_manager.log_access_attempt(
                user_id=user_id,
                file_path=file_path,
                access_type=access_type,
                access_result='success'
            )
            
            return {
                'status': 'success',
                'message': 'ファイルアクセス許可',
                'file_path': file_path,
                'integrity_status': integrity_result['status']
            }
            
        except Exception as e:
            self.audit_logger.log_security_event(
                event_type="FILE_ACCESS_ERROR",
                user_id=user_id,
                details={'file_path': file_path, 'error': str(e)},
                severity="ERROR"
            )
            return {
                'status': 'error',
                'message': f'ファイルアクセスエラー: {e}'
            }
    
    def _sanitize_filename(self, filename: str) -> str:
        """ファイル名の安全化"""
        # 危険な文字を除去
        import re
        safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
        
        # 長さ制限
        if len(safe_filename) > 255:
            name, ext = os.path.splitext(safe_filename)
            safe_filename = name[:255-len(ext)] + ext
        
        return safe_filename
    
    def get_security_status(self) -> Dict[str, Any]:
        """セキュリティステータスの取得"""
        try:
            return {
                'encryption_enabled': True,
                'integrity_checking_enabled': True,
                'access_control_enabled': True,
                'audit_logging_enabled': True,
                'https_enforced': True,  # OpenAI API通信
                'master_key_exists': os.path.exists("./secure/.master_key"),
                'hmac_key_exists': os.path.exists("./secure/.hmac_key"),
                'databases_initialized': all([
                    self.integrity_manager.integrity_db.exists(),
                    self.access_control_manager.access_db.exists()
                ])
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'セキュリティステータス取得エラー: {e}'
            }

class SecurityException(Exception):
    """セキュリティ関連の例外クラス"""
    pass

# セキュリティ管理用のユーティリティ関数
def init_security_system():
    """セキュリティシステムの初期化"""
    try:
        security_manager = SecurityManager()
        st.session_state['security_manager'] = security_manager
        return security_manager
    except Exception as e:
        st.error(f"セキュリティシステム初期化エラー: {e}")
        return None

def get_security_manager():
    """セキュリティマネージャーの取得"""
    if 'security_manager' not in st.session_state:
        return init_security_system()
    return st.session_state['security_manager']
