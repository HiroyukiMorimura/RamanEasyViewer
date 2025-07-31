# -*- coding: utf-8 -*-
"""
電子署名システム
重要な操作に対する電子署名機能を提供

Created for RamanEye Easy Viewer
@author: Electronic Signature System
"""

import streamlit as st
import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

class SignatureLevel(Enum):
    """署名レベル"""
    SINGLE = "single"      # 一段階署名
    DUAL = "dual"         # 二段階署名（二人体制）

class SignatureStatus(Enum):
    """署名ステータス"""
    PENDING = "pending"           # 署名待ち
    PARTIAL = "partial"          # 部分署名済み（二段階の場合）
    COMPLETED = "completed"      # 署名完了
    REJECTED = "rejected"        # 署名拒否

@dataclass
class SignatureRecord:
    """署名記録"""
    signature_id: str
    operation_type: str          # 操作タイプ
    operation_data: str          # 操作データのハッシュ
    level: SignatureLevel       # 署名レベル
    status: SignatureStatus     # 署名ステータス
    
    # 第一署名者情報
    primary_signer_id: Optional[str] = None
    primary_signer_name: Optional[str] = None
    primary_signature_time: Optional[str] = None
    primary_signature_reason: Optional[str] = None
    primary_password_hash: Optional[str] = None
    
    # 第二署名者情報（二段階署名の場合）
    secondary_signer_id: Optional[str] = None
    secondary_signer_name: Optional[str] = None
    secondary_signature_time: Optional[str] = None
    secondary_signature_reason: Optional[str] = None
    secondary_password_hash: Optional[str] = None
    
    # メタデータ
    created_at: str = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

class ElectronicSignatureManager:
    """電子署名管理クラス"""
    
    def __init__(self):
        # セッション状態の初期化
        if "signature_records" not in st.session_state:
            st.session_state.signature_records = {}
        if "pending_signatures" not in st.session_state:
            st.session_state.pending_signatures = {}
    
    def create_signature_request(self, 
                               operation_type: str, 
                               operation_data: Any,
                               signature_level: SignatureLevel = SignatureLevel.SINGLE,
                               required_signers: List[str] = None) -> str:
        """署名要求を作成"""
        signature_id = str(uuid.uuid4())
        
        # 操作データのハッシュ化
        operation_hash = self._hash_operation_data(operation_data)
        
        # 署名記録を作成
        signature_record = SignatureRecord(
            signature_id=signature_id,
            operation_type=operation_type,
            operation_data=operation_hash,
            level=signature_level,
            status=SignatureStatus.PENDING
        )
        
        # 記録を保存
        st.session_state.signature_records[signature_id] = signature_record
        st.session_state.pending_signatures[signature_id] = {
            "record": signature_record,
            "required_signers": required_signers or [],
            "operation_data_original": operation_data
        }
        
        return signature_id
    
    def _hash_operation_data(self, data: Any) -> str:
        """操作データのハッシュ化"""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        elif isinstance(data, str):
            data_str = data
        else:
            data_str = str(data)
        
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    
    def _hash_password(self, password: str) -> str:
        """パスワードのハッシュ化"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    def verify_user_password(self, username: str, password: str) -> bool:
        """ユーザーパスワードの検証"""
        from auth_system import UserDatabase
        db = UserDatabase()
        success, _ = db.authenticate_user(username, password)
        return success
    
    def add_signature(self, 
                     signature_id: str, 
                     signer_id: str, 
                     signer_name: str,
                     password: str,
                     reason: str,
                     is_secondary: bool = False) -> tuple[bool, str]:
        """署名を追加"""
        
        if signature_id not in st.session_state.signature_records:
            return False, "署名要求が見つかりません"
        
        record = st.session_state.signature_records[signature_id]
        
        # パスワード検証
        if not self.verify_user_password(signer_id, password):
            return False, "パスワードが正しくありません"
        
        # 署名追加
        current_time = datetime.now(timezone.utc).isoformat()
        password_hash = self._hash_password(password)
        
        if not is_secondary:
            # 第一署名者
            record.primary_signer_id = signer_id
            record.primary_signer_name = signer_name
            record.primary_signature_time = current_time
            record.primary_signature_reason = reason
            record.primary_password_hash = password_hash
            
            if record.level == SignatureLevel.SINGLE:
                record.status = SignatureStatus.COMPLETED
            else:
                record.status = SignatureStatus.PARTIAL
        else:
            # 第二署名者
            if record.level != SignatureLevel.DUAL:
                return False, "この操作は一段階署名です"
            
            if record.status != SignatureStatus.PARTIAL:
                return False, "第一署名が完了していません"
            
            record.secondary_signer_id = signer_id
            record.secondary_signer_name = signer_name
            record.secondary_signature_time = current_time
            record.secondary_signature_reason = reason
            record.secondary_password_hash = password_hash
            record.status = SignatureStatus.COMPLETED
        
        # 記録を更新
        st.session_state.signature_records[signature_id] = record
        
        # 署名完了時の処理
        if record.status == SignatureStatus.COMPLETED:
            self._on_signature_completed(signature_id)
        
        return True, "署名が正常に完了しました"
    
    def _on_signature_completed(self, signature_id: str):
        """署名完了時の処理"""
        # 監査ログの記録
        record = st.session_state.signature_records[signature_id]
        
        # ペンディング署名から削除
        if signature_id in st.session_state.pending_signatures:
            del st.session_state.pending_signatures[signature_id]
        
        # 完了通知
        st.success(f"電子署名が完了しました: {record.operation_type}")
    
    def get_signature_record(self, signature_id: str) -> Optional[SignatureRecord]:
        """署名記録を取得"""
        return st.session_state.signature_records.get(signature_id)
    
    def get_pending_signatures(self, user_id: str = None) -> List[Dict]:
        """ペンディング署名一覧を取得"""
        pending = []
        
        for sig_id, sig_data in st.session_state.pending_signatures.items():
            record = sig_data["record"]
            required_signers = sig_data.get("required_signers", [])
            
            # ユーザーフィルタリング
            if user_id:
                if user_id not in required_signers and required_signers:
                    continue
            
            pending.append({
                "signature_id": sig_id,
                "operation_type": record.operation_type,
                "level": record.level.value,
                "status": record.status.value,
                "created_at": record.created_at,
                "required_signers": required_signers
            })
        
        return pending
    
    def get_signature_history(self, limit: int = 50) -> List[Dict]:
        """署名履歴を取得"""
        history = []
        
        for sig_id, record in st.session_state.signature_records.items():
            history.append({
                "signature_id": sig_id,
                "operation_type": record.operation_type,
                "level": record.level.value,
                "status": record.status.value,
                "primary_signer": record.primary_signer_name,
                "primary_time": record.primary_signature_time,
                "secondary_signer": record.secondary_signer_name,
                "secondary_time": record.secondary_signature_time,
                "created_at": record.created_at
            })
        
        # 作成日時でソート
        history.sort(key=lambda x: x["created_at"], reverse=True)
        return history[:limit]
    
    def export_signature_records(self) -> str:
        """署名記録のエクスポート"""
        records = []
        
        for sig_id, record in st.session_state.signature_records.items():
            records.append(asdict(record))
        
        return json.dumps(records, ensure_ascii=False, indent=2)

# 署名UIコンポーネント
class SignatureUI:
    """署名UI管理クラス"""
    
    def __init__(self):
        self.signature_manager = ElectronicSignatureManager()
    
    def render_signature_dialog(self, 
                               signature_id: str, 
                               current_user_id: str,
                               current_user_name: str) -> bool:
        """署名ダイアログを表示"""
        
        record = self.signature_manager.get_signature_record(signature_id)
        if not record:
            st.error("署名要求が見つかりません")
            return False
        
        # 署名が必要かどうかの判定
        is_secondary_needed = (
            record.level == SignatureLevel.DUAL and 
            record.status == SignatureStatus.PARTIAL and
            record.primary_signer_id != current_user_id
        )
        
        is_primary_needed = record.status == SignatureStatus.PENDING
        
        if not (is_primary_needed or is_secondary_needed):
            st.info("この操作の署名は既に完了しています")
            return True
        
        # 署名フォーム
        st.subheader("🔏 電子署名")
        
        # 操作情報の表示
        st.info(f"""
        **操作タイプ**: {record.operation_type}
        **署名レベル**: {'二段階署名' if record.level == SignatureLevel.DUAL else '一段階署名'}
        **ステータス**: {record.status.value}
        """)
        
        # 既存署名の表示
        if record.primary_signer_name:
            st.success(f"✅ 第一署名者: {record.primary_signer_name} ({record.primary_signature_time})")
        
        if record.secondary_signer_name:
            st.success(f"✅ 第二署名者: {record.secondary_signer_name} ({record.secondary_signature_time})")
        
        # 署名フォーム
        with st.form(f"signature_form_{signature_id}"):
            st.write(f"**署名者**: {current_user_name}")
            
            # パスワード再入力
            password = st.text_input(
                "パスワードを再入力してください", 
                type="password",
                help="本人確認のため、現在のパスワードを入力してください"
            )
            
            # 署名理由
            reason = st.text_area(
                "署名理由",
                placeholder="例：データ解析結果を確認し、承認いたします",
                help="署名する理由を明確に記載してください"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                sign_button = st.form_submit_button(
                    "🔏 署名実行", 
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                reject_button = st.form_submit_button(
                    "❌ 署名拒否",
                    use_container_width=True
                )
        
        # 署名処理
        if sign_button:
            if not password:
                st.error("パスワードを入力してください")
                return False
            
            if not reason.strip():
                st.error("署名理由を入力してください")
                return False
            
            # 署名実行
            success, message = self.signature_manager.add_signature(
                signature_id=signature_id,
                signer_id=current_user_id,
                signer_name=current_user_name,
                password=password,
                reason=reason.strip(),
                is_secondary=is_secondary_needed
            )
            
            if success:
                st.success(message)
                st.balloons()
                return True
            else:
                st.error(message)
                return False
        
        # 署名拒否処理
        if reject_button:
            record.status = SignatureStatus.REJECTED
            st.session_state.signature_records[signature_id] = record
            st.warning("署名を拒否しました")
            return False
        
        return False
    
    def render_signature_status(self, signature_id: str):
        """署名ステータスを表示"""
        record = self.signature_manager.get_signature_record(signature_id)
        if not record:
            return
        
        # ステータス表示
        status_colors = {
            SignatureStatus.PENDING: "🟡",
            SignatureStatus.PARTIAL: "🟠", 
            SignatureStatus.COMPLETED: "🟢",
            SignatureStatus.REJECTED: "🔴"
        }
        
        status_color = status_colors.get(record.status, "⚪")
        st.write(f"{status_color} **署名ステータス**: {record.status.value}")
        
        # 詳細情報
        with st.expander("署名詳細"):
            if record.primary_signer_name:
                st.write(f"**第一署名者**: {record.primary_signer_name}")
                st.write(f"**署名日時**: {record.primary_signature_time}")
                st.write(f"**署名理由**: {record.primary_signature_reason}")
            
            if record.secondary_signer_name:
                st.write(f"**第二署名者**: {record.secondary_signer_name}")
                st.write(f"**署名日時**: {record.secondary_signature_time}")
                st.write(f"**署名理由**: {record.secondary_signature_reason}")

# 署名要求デコレータ
def require_signature(operation_type: str, 
                     signature_level: SignatureLevel = SignatureLevel.SINGLE,
                     required_signers: List[str] = None):
    """電子署名が必要な操作に使用するデコレータ"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            from auth_system import AuthenticationManager
            
            auth_manager = AuthenticationManager()
            if not auth_manager.is_authenticated():
                st.error("この機能を使用するにはログインが必要です")
                st.stop()
            
            current_user = auth_manager.get_current_user()
            
            # セッション状態の確認
            signature_key = f"signature_pending_{func.__name__}"
            
            if signature_key not in st.session_state:
                # 署名要求を作成
                signature_manager = ElectronicSignatureManager()
                operation_data = {"function": func.__name__, "args": str(args), "kwargs": str(kwargs)}
                
                signature_id = signature_manager.create_signature_request(
                    operation_type=operation_type,
                    operation_data=operation_data,
                    signature_level=signature_level,
                    required_signers=required_signers
                )
                
                st.session_state[signature_key] = signature_id
            
            signature_id = st.session_state[signature_key]
            
            # 署名UI表示
            signature_ui = SignatureUI()
            user_info = auth_manager.auth_manager.db.get_user(current_user)
            user_name = user_info.get("full_name", current_user)
            
            signature_completed = signature_ui.render_signature_dialog(
                signature_id, current_user, user_name
            )
            
            if signature_completed:
                # 署名完了、元の機能を実行
                del st.session_state[signature_key]
                return func(*args, **kwargs)
            else:
                # 署名待ち
                st.stop()
        
        return wrapper
    return decorator
