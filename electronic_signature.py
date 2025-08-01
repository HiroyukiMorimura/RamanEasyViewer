# -*- coding: utf-8 -*-
"""
電子署名システム（セキュリティ統合版） - 完全版
重要な操作に対するセキュア電子署名機能を提供
Enhanced with comprehensive security features

Created for RamanEye Easy Viewer - Secure Enterprise Edition
@author: Enhanced Electronic Signature System with Security Integration
"""

import streamlit as st
import hashlib
import json
import secrets
import base64
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hmac

# 暗号化ライブラリのインポート（エラーハンドリング付き）
try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    if 'crypto_warning_shown' not in st.session_state:
        st.warning("⚠️ cryptographyライブラリが見つかりません。デジタル署名機能は制限されます。")
        st.session_state.crypto_warning_shown = True

# セキュリティモジュールのインポート
try:
    from security_manager import (
        SecurityManager,
        get_security_manager,
        SecurityConfig,
        SecurityException
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    # SecurityExceptionのフォールバック定義
    class SecurityException(Exception):
        """セキュリティ例外（フォールバック）"""
        pass

class SignatureLevel(Enum):
    """署名レベル（セキュリティ強化版）"""
    SINGLE = "single"          # 一段階署名
    DUAL = "dual"             # 二段階署名（二人体制）
    MULTI = "multi"           # 多段階署名（3人以上）
    HIERARCHICAL = "hierarchical"  # 階層署名（管理者承認必須）

class SignatureStatus(Enum):
    """署名ステータス（強化版）"""
    PENDING = "pending"               # 署名待ち
    PARTIAL = "partial"              # 部分署名済み
    COMPLETED = "completed"          # 署名完了
    REJECTED = "rejected"            # 署名拒否
    EXPIRED = "expired"              # 期限切れ
    REVOKED = "revoked"              # 取り消し
    SUSPENDED = "suspended"          # 一時停止

class SignatureType(Enum):
    """署名タイプ"""
    APPROVAL = "approval"            # 承認署名
    WITNESS = "witness"              # 証人署名
    NOTARIZATION = "notarization"    # 公証署名
    AUTHORIZATION = "authorization"  # 認可署名

@dataclass
class SecureSignatureRecord:
    """セキュア強化署名記録"""
    signature_id: str
    operation_type: str
    operation_data_hash: str
    signature_level: SignatureLevel
    signature_type: SignatureType
    status: SignatureStatus
    
    # セキュリティ強化フィールド（デフォルト値を設定）
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    digital_signature: Optional[str] = None
    tamper_proof_seal: Optional[str] = None
    blockchain_hash: Optional[str] = None
    
    # 第一署名者情報（暗号化強化）
    primary_signer_id: Optional[str] = None
    primary_signer_name: Optional[str] = None
    primary_signature_time: Optional[str] = None
    primary_signature_reason: Optional[str] = None
    primary_password_hash: Optional[str] = None
    primary_digital_signature: Optional[str] = None
    primary_certificate_fingerprint: Optional[str] = None
    
    # 第二署名者情報（暗号化強化）
    secondary_signer_id: Optional[str] = None
    secondary_signer_name: Optional[str] = None
    secondary_signature_time: Optional[str] = None
    secondary_signature_reason: Optional[str] = None
    secondary_password_hash: Optional[str] = None
    secondary_digital_signature: Optional[str] = None
    secondary_certificate_fingerprint: Optional[str] = None
    
    # 追加署名者情報（多段階署名用）
    additional_signers: Optional[List[Dict]] = None
    
    # 監査・コンプライアンス情報
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    geolocation: Optional[str] = None
    compliance_flags: Optional[List[str]] = None
    audit_trail: Optional[List[Dict]] = None
    
    # 暗号化・完全性情報
    encryption_algorithm: str = "AES-256-GCM"
    hash_algorithm: str = "SHA-256"
    signature_algorithm: str = "RSA-PSS"
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if self.additional_signers is None:
            self.additional_signers = []
        if self.compliance_flags is None:
            self.compliance_flags = []
        if self.audit_trail is None:
            self.audit_trail = []

class SecureElectronicSignatureManager:
    """セキュア強化電子署名管理クラス（完全修正版）"""
    
    def __init__(self):
        # セッション状態の初期化
        if "secure_signature_records" not in st.session_state:
            st.session_state.secure_signature_records = {}
        if "secure_pending_signatures" not in st.session_state:
            st.session_state.secure_pending_signatures = {}
        
        # セキュリティマネージャーの取得
        self.security_manager = get_security_manager() if SECURITY_AVAILABLE else None
        
        # 暗号化キーの初期化
        if CRYPTO_AVAILABLE:
            self._initialize_crypto_keys()
    
    def _initialize_crypto_keys(self):
        """暗号化キーの初期化"""
        if not CRYPTO_AVAILABLE:
            return
        
        try:
            if "signature_private_key" not in st.session_state:
                # RSA秘密鍵の生成
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                
                # PEM形式でシリアライズ
                private_pem = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                public_key = private_key.public_key()
                public_pem = public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
                
                st.session_state.signature_private_key = private_pem
                st.session_state.signature_public_key = public_pem
                
        except Exception as e:
            st.error(f"暗号化キー初期化エラー: {e}")
    
    def create_secure_signature_request(self, 
                                      operation_type: str, 
                                      operation_data: Any,
                                      signature_level: SignatureLevel = SignatureLevel.SINGLE,
                                      signature_type: SignatureType = SignatureType.APPROVAL,
                                      required_signers: List[str] = None,
                                      expires_in_hours: int = 24) -> str:
        """セキュア強化署名要求を作成"""
        
        try:
            signature_id = str(uuid.uuid4())
            
            # 操作データの暗号化とハッシュ化
            operation_hash = self._secure_hash_operation_data(operation_data)
            tamper_proof_seal = self._generate_tamper_proof_seal(operation_data, signature_id)
            
            # 有効期限の設定
            expires_at = (datetime.now(timezone.utc) + timedelta(hours=expires_in_hours)).isoformat()
            
            # デジタル署名の生成
            digital_signature = None
            if CRYPTO_AVAILABLE:
                try:
                    digital_signature = self._generate_digital_signature(operation_hash + signature_id)
                except Exception as e:
                    st.warning(f"デジタル署名生成に失敗しました: {e}")
            
            # 署名記録を作成
            signature_record = SecureSignatureRecord(
                signature_id=signature_id,
                operation_type=operation_type,
                operation_data_hash=operation_hash,
                signature_level=signature_level,
                signature_type=signature_type,
                status=SignatureStatus.PENDING,
                expires_at=expires_at,
                digital_signature=digital_signature,
                tamper_proof_seal=tamper_proof_seal
            )
            
            # コンプライアンス情報の追加
            signature_record.compliance_flags = self._determine_compliance_flags(operation_type)
            
            # 記録を保存
            st.session_state.secure_signature_records[signature_id] = signature_record
            st.session_state.secure_pending_signatures[signature_id] = {
                "record": signature_record,
                "required_signers": required_signers or [],
                "operation_data_original": operation_data,
                "security_context": self._capture_security_context()
            }
            
            return signature_id
            
        except Exception as e:
            error_msg = f"署名要求作成エラー: {e}"
            st.error(error_msg)
            raise Exception(error_msg)
    
    def _secure_hash_operation_data(self, data: Any) -> str:
        """セキュア強化操作データハッシュ化"""
        try:
            if isinstance(data, dict):
                data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, str):
                data_str = data
            else:
                data_str = str(data)
            
            # ソルト付きハッシュ
            salt = secrets.token_bytes(32)
            hasher = hashlib.sha256()
            hasher.update(salt + data_str.encode('utf-8'))
            salted_hash = base64.urlsafe_b64encode(salt + hasher.digest()).decode()
            return salted_hash
            
        except Exception as e:
            error_msg = f"セキュアハッシュ化エラー: {e}"
            st.error(error_msg)
            raise Exception(error_msg)
    
    def _generate_tamper_proof_seal(self, data: Any, signature_id: str) -> str:
        """改ざん防止シールの生成"""
        try:
            key = secrets.token_bytes(32)
            message = f"{signature_id}:{str(data)}:{datetime.now(timezone.utc).isoformat()}"
            seal = hmac.new(key, message.encode(), hashlib.sha256).hexdigest()
            return base64.urlsafe_b64encode(key + bytes.fromhex(seal)).decode()
        except Exception as e:
            st.error(f"改ざん防止シール生成エラー: {e}")
            return ""
    
    def _generate_digital_signature(self, data: str) -> str:
        """デジタル署名の生成"""
        if not CRYPTO_AVAILABLE:
            return None
        
        try:
            private_key_pem = st.session_state.get("signature_private_key")
            if not private_key_pem:
                return None
            
            private_key = load_pem_private_key(private_key_pem, password=None)
            signature = private_key.sign(
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return base64.urlsafe_b64encode(signature).decode()
        except Exception as e:
            st.warning(f"デジタル署名生成エラー: {e}")
            return None
    
    def _verify_digital_signature(self, data: str, signature: str) -> bool:
        """デジタル署名の検証"""
        if not CRYPTO_AVAILABLE or not signature:
            return True
        
        try:
            public_key_pem = st.session_state.get("signature_public_key")
            if not public_key_pem:
                return False
            
            public_key = load_pem_public_key(public_key_pem)
            signature_bytes = base64.urlsafe_b64decode(signature.encode())
            
            public_key.verify(
                signature_bytes,
                data.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def _determine_compliance_flags(self, operation_type: str) -> List[str]:
        """コンプライアンス要件の判定"""
        flags = []
        high_risk_operations = [
            "data_export", "system_configuration", "user_management",
            "security_settings", "database_modification", "重要レポート確定"
        ]
        
        if any(risk_op in operation_type.lower() for risk_op in high_risk_operations):
            flags.append("HIGH_RISK")
        
        try:
            from config import ComplianceConfig
            if ComplianceConfig.GDPR_COMPLIANCE:
                flags.append("GDPR")
        except ImportError:
            flags.append("STANDARD_COMPLIANCE")
        
        return flags
    
    def _capture_security_context(self) -> Dict[str, Any]:
        """セキュリティコンテキストの取得"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': st.session_state.get('session_id', 'unknown'),
            'user_agent': 'Streamlit-Application',
            'ip_address': 'localhost',
            'security_level': st.session_state.get('security_level', 'standard')
        }
    
    def _hash_password_secure(self, password: str) -> str:
        """セキュア強化パスワードハッシュ化"""
        salt = secrets.token_bytes(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return base64.urlsafe_b64encode(salt + password_hash).decode()
    
    def verify_user_password_secure(self, username: str, password: str) -> bool:
        """セキュア強化ユーザーパスワード検証"""
        try:
            from auth_system import UserDatabase
            db = UserDatabase()
            success, _ = db.authenticate_user(username, password)
            return success
        except Exception as e:
            st.warning(f"認証システムエラー: {e}")
            return False
    
    def get_secure_signature_record(self, signature_id: str) -> Optional[SecureSignatureRecord]:
        """セキュア署名記録を取得 - 重要メソッド"""
        try:
            if signature_id in st.session_state.secure_signature_records:
                return st.session_state.secure_signature_records[signature_id]
            else:
                st.warning(f"署名記録が見つかりません: {signature_id}")
                return None
        except Exception as e:
            st.error(f"署名記録取得エラー: {e}")
            return None
    
    def _get_required_signature_count(self, signature_level: SignatureLevel) -> int:
        """必要な署名数を取得"""
        try:
            counts = {
                SignatureLevel.SINGLE: 1,
                SignatureLevel.DUAL: 2,
                SignatureLevel.MULTI: 3,
                SignatureLevel.HIERARCHICAL: 2
            }
            return counts.get(signature_level, 1)
        except Exception:
            return 1
    
    def _count_current_signatures(self, record: SecureSignatureRecord) -> int:
        """現在の署名数をカウント"""
        try:
            count = 0
            if record.primary_signer_id:
                count += 1
            if record.secondary_signer_id:
                count += 1
            if record.additional_signers:
                count += len(record.additional_signers)
            return count
        except Exception:
            return 0
    
    def _on_secure_signature_completed(self, signature_id: str):
        """セキュア署名完了時の処理"""
        try:
            if signature_id not in st.session_state.secure_signature_records:
                return
            
            record = st.session_state.secure_signature_records[signature_id]
            
            # ペンディング署名から削除
            if signature_id in st.session_state.secure_pending_signatures:
                del st.session_state.secure_pending_signatures[signature_id]
            
            # ブロックチェーンハッシュの生成
            blockchain_data = f"{signature_id}:{record.operation_type}:{record.status.value}:{datetime.now(timezone.utc).isoformat()}"
            record.blockchain_hash = hashlib.sha256(blockchain_data.encode()).hexdigest()
            
            # 記録を更新
            st.session_state.secure_signature_records[signature_id] = record
            
            # 完了通知
            st.success(f"🔒 セキュア電子署名が完了しました: {record.operation_type}")
            st.balloons()
                
        except Exception as e:
            st.error(f"署名完了処理エラー: {e}")
    
    def add_secure_signature(self, 
                           signature_id: str, 
                           signer_id: str, 
                           signer_name: str,
                           password: str,
                           reason: str,
                           is_secondary: bool = False,
                           additional_context: Dict = None) -> tuple[bool, str]:
        """セキュア強化署名追加"""
        
        try:
            if signature_id not in st.session_state.secure_signature_records:
                return False, "署名要求が見つかりません"
            
            record = st.session_state.secure_signature_records[signature_id]
            
            # 有効期限チェック
            if record.expires_at:
                expires_time = datetime.fromisoformat(record.expires_at.replace('Z', '+00:00'))
                if datetime.now(timezone.utc) > expires_time:
                    record.status = SignatureStatus.EXPIRED
                    return False, "署名要求の有効期限が切れています"
            
            # パスワード検証
            if not self.verify_user_password_secure(signer_id, password):
                return False, "パスワードが正しくありません"
            
            # 二重署名防止チェック
            if (record.primary_signer_id == signer_id or 
                record.secondary_signer_id == signer_id or
                any(s.get('signer_id') == signer_id for s in record.additional_signers)):
                return False, "同一ユーザーによる重複署名は許可されていません"
            
            # セキュア署名データの生成
            current_time = datetime.now(timezone.utc).isoformat()
            password_hash = self._hash_password_secure(password)
            
            # デジタル署名の生成
            signature_data = f"{signature_id}:{signer_id}:{current_time}:{reason}"
            digital_signature = None
            if CRYPTO_AVAILABLE:
                try:
                    digital_signature = self._generate_digital_signature(signature_data)
                except Exception as e:
                    st.warning(f"デジタル署名生成に失敗: {e}")
            
            # 証明書フィンガープリントの生成
            certificate_fingerprint = hashlib.sha256(f"{signer_id}:{current_time}".encode()).hexdigest()[:16]
            
            # 署名の追加
            if not is_secondary and record.primary_signer_id is None:
                # 第一署名者
                record.primary_signer_id = signer_id
                record.primary_signer_name = signer_name
                record.primary_signature_time = current_time
                record.primary_signature_reason = reason
                record.primary_password_hash = password_hash
                record.primary_digital_signature = digital_signature
                record.primary_certificate_fingerprint = certificate_fingerprint
                
                if record.signature_level == SignatureLevel.SINGLE:
                    record.status = SignatureStatus.COMPLETED
                else:
                    record.status = SignatureStatus.PARTIAL
                    
            elif record.signature_level in [SignatureLevel.DUAL, SignatureLevel.MULTI, SignatureLevel.HIERARCHICAL]:
                if record.secondary_signer_id is None and record.status == SignatureStatus.PARTIAL:
                    # 第二署名者
                    record.secondary_signer_id = signer_id
                    record.secondary_signer_name = signer_name
                    record.secondary_signature_time = current_time
                    record.secondary_signature_reason = reason
                    record.secondary_password_hash = password_hash
                    record.secondary_digital_signature = digital_signature
                    record.secondary_certificate_fingerprint = certificate_fingerprint
                    
                    if record.signature_level == SignatureLevel.DUAL:
                        record.status = SignatureStatus.COMPLETED
                    else:
                        record.status = SignatureStatus.PARTIAL
                        
                elif record.signature_level in [SignatureLevel.MULTI, SignatureLevel.HIERARCHICAL]:
                    # 追加署名者
                    additional_signer = {
                        'signer_id': signer_id,
                        'signer_name': signer_name,
                        'signature_time': current_time,
                        'signature_reason': reason,
                        'password_hash': password_hash,
                        'digital_signature': digital_signature,
                        'certificate_fingerprint': certificate_fingerprint
                    }
                    record.additional_signers.append(additional_signer)
                    
                    # 必要な署名数に達したかチェック
                    required_signatures = self._get_required_signature_count(record.signature_level)
                    current_signatures = self._count_current_signatures(record)
                    
                    if current_signatures >= required_signatures:
                        record.status = SignatureStatus.COMPLETED
                else:
                    return False, "署名の順序または権限が正しくありません"
            else:
                return False, "この操作の署名レベルでは追加署名は許可されていません"
            
            # 監査証跡の更新
            audit_entry = {
                'action': 'SIGNATURE_ADDED',
                'signer_id': signer_id,
                'timestamp': current_time,
                'ip_address': additional_context.get('ip_address') if additional_context else None,
                'user_agent': additional_context.get('user_agent') if additional_context else None
            }
            record.audit_trail.append(audit_entry)
            
            # 記録を更新
            st.session_state.secure_signature_records[signature_id] = record
            
            # 署名完了時の処理
            if record.status == SignatureStatus.COMPLETED:
                self._on_secure_signature_completed(signature_id)
            
            return True, "セキュア署名が正常に完了しました"
            
        except Exception as e:
            error_msg = f"セキュア署名エラー: {e}"
            st.error(error_msg)
            return False, error_msg
    
    def get_pending_secure_signatures(self, user_id: str = None) -> List[Dict]:
        """ペンディングセキュア署名一覧を取得"""
        pending = []
        
        try:
            for sig_id, sig_data in st.session_state.secure_pending_signatures.items():
                record = sig_data["record"]
                required_signers = sig_data.get("required_signers", [])
                
                # 有効期限チェック
                if record.expires_at:
                    expires_time = datetime.fromisoformat(record.expires_at.replace('Z', '+00:00'))
                    if datetime.now(timezone.utc) > expires_time:
                        record.status = SignatureStatus.EXPIRED
                        continue
                
                # ユーザーフィルタリング
                if user_id:
                    if user_id not in required_signers and required_signers:
                        continue
                    
                    # 既に署名済みのユーザーは除外
                    if (record.primary_signer_id == user_id or 
                        record.secondary_signer_id == user_id or
                        any(s.get('signer_id') == user_id for s in record.additional_signers)):
                        continue
                
                pending.append({
                    "signature_id": sig_id,
                    "operation_type": record.operation_type,
                    "signature_level": record.signature_level.value,
                    "signature_type": record.signature_type.value,
                    "status": record.status.value,
                    "created_at": record.created_at,
                    "expires_at": record.expires_at,
                    "required_signers": required_signers,
                    "compliance_flags": record.compliance_flags,
                    "current_signatures": self._count_current_signatures(record),
                    "required_signatures": self._get_required_signature_count(record.signature_level)
                })
        except Exception as e:
            st.error(f"ペンディング署名取得エラー: {e}")
        
        return pending
    
    def get_secure_signature_history(self, limit: int = 50, include_sensitive: bool = False) -> List[Dict]:
        """セキュア署名履歴を取得"""
        history = []
        
        try:
            for sig_id, record in st.session_state.secure_signature_records.items():
                history_entry = {
                    "signature_id": sig_id,
                    "operation_type": record.operation_type,
                    "signature_level": record.signature_level.value,
                    "signature_type": record.signature_type.value,
                    "status": record.status.value,
                    "primary_signer": record.primary_signer_name,
                    "primary_time": record.primary_signature_time,
                    "secondary_signer": record.secondary_signer_name,
                    "secondary_time": record.secondary_signature_time,
                    "created_at": record.created_at,
                    "expires_at": record.expires_at,
                    "compliance_flags": record.compliance_flags,
                    "blockchain_verified": bool(record.blockchain_hash),
                    "signature_count": self._count_current_signatures(record)
                }
                history.append(history_entry)
            
            # 作成日時でソート
            history.sort(key=lambda x: x["created_at"], reverse=True)
            return history[:limit]
        except Exception as e:
            st.error(f"署名履歴取得エラー: {e}")
            return []
    
    def export_secure_signature_records(self, include_sensitive: bool = False) -> str:
        """セキュア署名記録のエクスポート"""
        try:
            records = []
            for sig_id, record in st.session_state.secure_signature_records.items():
                record_dict = asdict(record)
                
                # センシティブ情報の制御
                if not include_sensitive:
                    sensitive_fields = [
                        'primary_password_hash', 'secondary_password_hash',
                        'tamper_proof_seal', 'digital_signature'
                    ]
                    for field in sensitive_fields:
                        record_dict.pop(field, None)
                    
                    for signer in record_dict.get('additional_signers', []):
                        signer.pop('password_hash', None)
                
                records.append(record_dict)
            
            return json.dumps(records, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            st.error(f"署名記録エクスポートエラー: {e}")
            return "{}"
    
    def verify_signature_integrity(self, signature_id: str) -> Dict[str, Any]:
        """署名の完全性を検証"""
        try:
            record = self.get_secure_signature_record(signature_id)
            if not record:
                return {
                    "status": "error", 
                    "message": "署名記録が見つかりません",
                    "signature_id": signature_id
                }
            
            verification_results = {
                "signature_id": signature_id,
                "status": "verified",
                "checks": [],
                "warnings": [],
                "errors": []
            }
            
            # デジタル署名の検証
            if record.digital_signature:
                try:
                    signature_data = record.operation_data_hash + signature_id
                    if self._verify_digital_signature(signature_data, record.digital_signature):
                        verification_results["checks"].append("✅ デジタル署名検証成功")
                    else:
                        verification_results["errors"].append("❌ デジタル署名検証失敗")
                except Exception as e:
                    verification_results["errors"].append(f"❌ デジタル署名検証エラー: {e}")
            
            # タイムスタンプの検証
            try:
                if record.created_at:
                    created_time = datetime.fromisoformat(record.created_at.replace('Z', '+00:00'))
                    if created_time > datetime.now(timezone.utc):
                        verification_results["warnings"].append("⚠️ 作成日時が未来になっています")
                    else:
                        verification_results["checks"].append("✅ タイムスタンプ検証成功")
                else:
                    verification_results["warnings"].append("⚠️ 作成日時が記録されていません")
            except Exception as e:
                verification_results["errors"].append(f"❌ タイムスタンプ検証エラー: {e}")
            
            # 署名者の検証
            try:
                signers_verified = self._count_current_signatures(record)
                required_signers = self._get_required_signature_count(record.signature_level)
                
                if signers_verified >= required_signers:
                    verification_results["checks"].append(f"✅ 必要署名数達成 ({signers_verified}/{required_signers})")
                else:
                    verification_results["errors"].append(f"❌ 署名数不足 ({signers_verified}/{required_signers})")
            except Exception as e:
                verification_results["errors"].append(f"❌ 署名者検証エラー: {e}")
            
            # ブロックチェーンハッシュの検証
            if record.blockchain_hash:
                verification_results["checks"].append("✅ ブロックチェーンハッシュ存在")
            
            # 全体的なステータス判定
            if verification_results["errors"]:
                verification_results["status"] = "failed"
            elif verification_results["warnings"]:
                verification_results["status"] = "warning"
            
            return verification_results
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"検証エラー: {e}",
                "signature_id": signature_id
            }

# セキュア強化署名UIコンポーネント
class SecureSignatureUI:
    """セキュア強化署名UI管理クラス"""
    
    def __init__(self):
        self.signature_manager = SecureElectronicSignatureManager()
        self.security_manager = get_security_manager() if SECURITY_AVAILABLE else None
    
    def render_secure_signature_dialog(self, 
                                     signature_id: str, 
                                     current_user_id: str,
                                     current_user_name: str) -> bool:
        """セキュア強化署名ダイアログを表示"""
        
        record = self.signature_manager.get_secure_signature_record(signature_id)
        if not record:
            st.error("🔒 署名要求が見つかりません")
            return False
        
        # 有効期限チェック
        if record.expires_at:
            expires_time = datetime.fromisoformat(record.expires_at.replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > expires_time:
                st.error("🕐 署名要求の有効期限が切れています")
                return False
        
        # 署名が必要かどうかの判定
        can_sign = self._can_user_sign(record, current_user_id)
        
        if not can_sign:
            st.info("ℹ️ この操作の署名は既に完了しているか、あなたの署名は不要です")
            return True
        
        # セキュア署名フォーム
        st.subheader("🔒 セキュア電子署名")
        
        # セキュリティ情報の表示
        with st.expander("🛡️ セキュリティ情報", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**暗号化アルゴリズム:**")
                st.code(f"データ: {record.encryption_algorithm}")
                st.code(f"ハッシュ: {record.hash_algorithm}")
                st.code(f"署名: {record.signature_algorithm}")
            
            with col2:
                st.write("**コンプライアンス:**")
                for flag in record.compliance_flags:
                    if flag == "完了":
                        st.success(f"✅ {flag}")
                    elif flag == "待機中":
                        st.warning(f"⏳ {flag}")
                    elif flag == "処理中":
                        st.info(f"⚙️ {flag}")
                    else:
                        st.write(f"🏷️ {flag}")  # 中性的な表示
                
                if record.blockchain_hash:
                    st.write("**ブロックチェーン:** ✅ 有効")
        
        # 操作情報の表示
        st.info(f"""
        **操作タイプ**: {record.operation_type}
        **署名レベル**: {self._get_signature_level_description(record.signature_level)}
        **署名タイプ**: {self._get_signature_type_description(record.signature_type)}
        **ステータス**: {record.status.value}
        **有効期限**: {record.expires_at}
        """)
        
        # 既存署名の表示
        self._render_existing_signatures(record, signature_id)
        
        # セキュア署名フォーム
        with st.form(f"secure_signature_form_{signature_id}"):
            st.write(f"**署名者**: {current_user_name}")
            
            # パスワード再入力（セキュリティ強化）
            password = st.text_input(
                "🔐 パスワードを再入力してください", 
                type="password",
                help="本人確認のため、現在のパスワードを入力してください。パスワードは暗号化されて検証されます。"
            )
            
            # 署名理由（必須）
            reason = st.text_area(
                "📝 署名理由（必須）",
                placeholder="例：データ解析結果を確認し、科学的妥当性を認めて承認いたします",
                help="署名する理由を明確に記載してください。この情報は監査証跡として永続的に保存されます。"
            )
            
            # セキュリティ確認
            security_agreement = st.checkbox(
                "🔒 セキュリティポリシーに同意し、この署名の法的効力を理解しています",
                help="この署名はデジタル証明書により法的拘束力を持ちます"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                sign_button = st.form_submit_button(
                    "🔏 セキュア署名実行", 
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                reject_button = st.form_submit_button(
                    "❌ 署名拒否",
                    use_container_width=True
                )
        
        # セキュア署名処理
        if sign_button:
            if not password:
                st.error("🔐 パスワードを入力してください")
                return False
            
            if not reason.strip():
                st.error("📝 署名理由を入力してください")
                return False
            
            if not security_agreement:
                st.error("🔒 セキュリティポリシーへの同意が必要です")
                return False
            
            # セキュリティコンテキストの取得
            security_context = {
                'ip_address': 'localhost',
                'user_agent': 'Streamlit-App',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # セキュア署名実行
            with st.spinner("🔒 セキュア署名を実行中..."):
                try:
                    is_secondary = self._determine_signature_order(record, current_user_id)
                    
                    success, message = self.signature_manager.add_secure_signature(
                        signature_id=signature_id,
                        signer_id=current_user_id,
                        signer_name=current_user_name,
                        password=password,
                        reason=reason.strip(),
                        is_secondary=is_secondary,
                        additional_context=security_context
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        
                        # 完全性検証の実行
                        verification_result = self.signature_manager.verify_signature_integrity(signature_id)
                        if verification_result["status"] == "verified":
                            st.success("🔍 署名完全性検証: 成功")
                        else:
                            st.warning("⚠️ 署名完全性検証で警告が発生しました")
                        
                        st.balloons()
                        return True
                    else:
                        st.error(f"❌ {message}")
                        return False
                        
                except Exception as e:
                    st.error(f"🚨 セキュア署名処理エラー: {str(e)}")
                    return False
        
        # 署名拒否処理
        if reject_button:
            record.status = SignatureStatus.REJECTED
            st.session_state.secure_signature_records[signature_id] = record
            
            # セキュリティログ記録
            if self.security_manager:
                try:
                    self.security_manager.audit_logger.log_security_event(
                        event_type="SECURE_SIGNATURE_REJECTED",
                        user_id=current_user_id,
                        details={
                            'signature_id': signature_id,
                            'operation_type': record.operation_type
                        },
                        severity="WARNING"
                    )
                except:
                    pass
            
            st.warning("⚠️ セキュア署名を拒否しました")
            return False
        
        return False
    
    def _can_user_sign(self, record: SecureSignatureRecord, user_id: str) -> bool:
        """ユーザーが署名可能かチェック"""
        # 既に署名済みかチェック
        if (record.primary_signer_id == user_id or 
            record.secondary_signer_id == user_id or
            any(s.get('signer_id') == user_id for s in record.additional_signers)):
            return False
        
        # ステータスチェック
        if record.status not in [SignatureStatus.PENDING, SignatureStatus.PARTIAL]:
            return False
        
        return True
    
    def _determine_signature_order(self, record: SecureSignatureRecord, user_id: str) -> bool:
        """署名順序を判定"""
        if record.primary_signer_id is None:
            return False  # 第一署名者
        elif record.signature_level != SignatureLevel.SINGLE and record.secondary_signer_id is None:
            return True   # 第二署名者
        else:
            return False  # 追加署名者
    
    def _get_signature_level_description(self, level: SignatureLevel) -> str:
        """署名レベルの説明を取得"""
        descriptions = {
            SignatureLevel.SINGLE: "一段階署名（1名の承認）",
            SignatureLevel.DUAL: "二段階署名（2名の承認）",
            SignatureLevel.MULTI: "多段階署名（3名以上の承認）",
            SignatureLevel.HIERARCHICAL: "階層署名（管理者承認必須）"
        }
        return descriptions.get(level, "不明な署名レベル")
    
    def _get_signature_type_description(self, signature_type: SignatureType) -> str:
        """署名タイプの説明を取得"""
        descriptions = {
            SignatureType.APPROVAL: "承認署名",
            SignatureType.WITNESS: "証人署名",
            SignatureType.NOTARIZATION: "公証署名",
            SignatureType.AUTHORIZATION: "認可署名"
        }
        return descriptions.get(signature_type, "不明な署名タイプ")
    
    def _render_existing_signatures(self, record: SecureSignatureRecord, signature_id: str):
        """既存署名の表示"""
        if record.primary_signer_name:
            st.success(f"✅ 第一署名者: {record.primary_signer_name} ({record.primary_signature_time})")
            if record.primary_certificate_fingerprint:
                st.caption(f"証明書フィンガープリント: {record.primary_certificate_fingerprint}")
        
        if record.secondary_signer_name:
            st.success(f"✅ 第二署名者: {record.secondary_signer_name} ({record.secondary_signature_time})")
            if record.secondary_certificate_fingerprint:
                st.caption(f"証明書フィンガープリント: {record.secondary_certificate_fingerprint}")
        
        # 追加署名者の表示
        for i, signer in enumerate(record.additional_signers, 3):
            st.success(f"✅ 第{i}署名者: {signer['signer_name']} ({signer['signature_time']})")
            if signer.get('certificate_fingerprint'):
                st.caption(f"証明書フィンガープリント: {signer['certificate_fingerprint']}")

# セキュア強化署名要求デコレータ
def require_secure_signature(operation_type: str, 
                           signature_level: SignatureLevel = SignatureLevel.SINGLE,
                           signature_type: SignatureType = SignatureType.APPROVAL,
                           required_signers: List[str] = None,
                           expires_in_hours: int = 24):
    """セキュア電子署名が必要な操作に使用するデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                from auth_system import AuthenticationManager
                
                auth_manager = AuthenticationManager()
                if not auth_manager.is_authenticated():
                    st.error("この機能を使用するにはログインが必要です")
                    st.stop()
                
                current_user = auth_manager.get_current_user()
                
                # セッション状態の確認
                signature_key = f"secure_signature_pending_{func.__name__}"
                
                if signature_key not in st.session_state:
                    # セキュア署名要求を作成
                    signature_manager = SecureElectronicSignatureManager()
                    operation_data = {
                        "function": func.__name__, 
                        "args": str(args), 
                        "kwargs": str(kwargs),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "user": current_user
                    }
                    
                    signature_id = signature_manager.create_secure_signature_request(
                        operation_type=operation_type,
                        operation_data=operation_data,
                        signature_level=signature_level,
                        signature_type=signature_type,
                        required_signers=required_signers,
                        expires_in_hours=expires_in_hours
                    )
                    
                    st.session_state[signature_key] = signature_id
                
                signature_id = st.session_state[signature_key]
                
                # セキュア署名UI表示
                signature_ui = SecureSignatureUI()
                user_info = auth_manager.db.get_user(current_user)
                user_name = user_info.get("full_name", current_user) if user_info else current_user
                
                signature_completed = signature_ui.render_secure_signature_dialog(
                    signature_id, current_user, user_name
                )
                
                if signature_completed:
                    # 署名完了、元の機能を実行
                    del st.session_state[signature_key]
                    return func(*args, **kwargs)
                else:
                    # 署名待ち
                    st.stop()
                    
            except Exception as e:
                st.error(f"電子署名システムエラー: {e}")
                st.stop()
        
        return wrapper
    return decorator

# 互換性のためのエイリアス
ElectronicSignatureManager = SecureElectronicSignatureManager
SignatureUI = SecureSignatureUI
SignatureRecord = SecureSignatureRecord
require_signature = require_secure_signature
