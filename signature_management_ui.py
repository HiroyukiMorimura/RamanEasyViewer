# -*- coding: utf-8 -*-
"""
署名管理UI（修正版）
電子署名の管理・監視・履歴表示機能

Created for RamanEye Easy Viewer
@author: Signature Management UI System
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from electronic_signature import (
    SecureElectronicSignatureManager,
    SecureSignatureUI,
    SignatureLevel,
    SignatureStatus
)

class SignatureManagementUI:
    """署名管理UIクラス"""
    
    def __init__(self):
        self.signature_manager = SecureElectronicSignatureManager()
        self.signature_ui = SecureSignatureUI()
    
    def render_signature_management_page(self):
        """署名管理ページをレンダリング"""
        st.header("🔏 電子署名管理")
        
        # タブで機能を分割
        tab1, tab2, tab3, tab4 = st.tabs([
            "ペンディング署名", 
            "署名履歴", 
            "署名統計", 
            "署名設定"
        ])
        
        with tab1:
            self._render_pending_signatures()
        
        with tab2:
            self._render_signature_history()
        
        with tab3:
            self._render_signature_statistics()
        
        with tab4:
            self._render_signature_settings()
    
    def _render_pending_signatures(self):
        """ペンディング署名一覧"""
        st.subheader("📋 署名待ち一覧")
        
        try:
            from auth_system import SecureAuthenticationManager
            auth_manager = SecureAuthenticationManager()
            current_user = auth_manager.get_current_user()
        except ImportError:
            # フォールバック
            current_user = st.session_state.get('current_user', {}).get('username', 'unknown')
        
        # ペンディング署名を取得（修正されたメソッド名）
        pending_signatures = self.signature_manager.get_pending_secure_signatures(current_user)
        
        if not pending_signatures:
            st.info("現在、署名待ちの操作はありません")
            return
        
        # データフレーム作成
        pending_data = []
        for sig in pending_signatures:
            pending_data.append({
                "操作タイプ": sig["operation_type"],
                "署名レベル": "二段階" if sig["signature_level"] == "dual" else "一段階",
                "ステータス": sig["status"],
                "作成日時": datetime.fromisoformat(sig["created_at"]).strftime("%Y-%m-%d %H:%M"),
                "有効期限": datetime.fromisoformat(sig["expires_at"]).strftime("%Y-%m-%d %H:%M") if sig["expires_at"] else "なし",
                "署名ID": sig["signature_id"],
                "進捗": f"{sig['current_signatures']}/{sig['required_signatures']}"
            })
        
        df = pd.DataFrame(pending_data)
        
        # 署名選択
        if not df.empty:
            selected_signature = st.selectbox(
                "署名する操作を選択してください:",
                options=df["署名ID"].tolist(),
                format_func=lambda x: f"{df[df['署名ID']==x]['操作タイプ'].iloc[0]} ({df[df['署名ID']==x]['作成日時'].iloc[0]})"
            )
            
            # 選択された署名の詳細表示と署名実行
            if selected_signature:
                st.markdown("---")
                
                # 現在のユーザー情報取得
                try:
                    user_info = auth_manager.db.get_user(current_user)
                    user_name = user_info.get("full_name", current_user)
                except:
                    user_name = current_user
                
                # 署名ダイアログ表示
                self.signature_ui.render_secure_signature_dialog(
                    selected_signature, 
                    current_user, 
                    user_name
                )
    
    def _render_signature_history(self):
        """署名履歴表示"""
        st.subheader("📚 署名履歴")
        
        # フィルター設定
        col1, col2, col3 = st.columns(3)
        
        with col1:
            limit = st.number_input("表示件数", min_value=10, max_value=500, value=50, step=10)
        
        with col2:
            status_filter = st.selectbox(
                "ステータスフィルター",
                ["すべて", "完了", "拒否", "部分署名", "待機中", "期限切れ"]
            )
        
        with col3:
            if st.button("🔄 更新"):
                st.rerun()
        
        # 署名履歴を取得（修正されたメソッド名）
        history = self.signature_manager.get_secure_signature_history(limit)
        
        if not history:
            st.info("署名履歴がありません")
            return
        
        # ステータスフィルタリング
        if status_filter != "すべて":
            status_map = {
                "完了": "completed",
                "拒否": "rejected", 
                "部分署名": "partial",
                "待機中": "pending",
                "期限切れ": "expired"
            }
            filter_status = status_map.get(status_filter)
            history = [h for h in history if h["status"] == filter_status]
        
        # データフレーム作成
        history_data = []
        for h in history:
            history_data.append({
                "操作タイプ": h["operation_type"],
                "署名レベル": "二段階" if h["signature_level"] == "dual" else "一段階",
                "署名タイプ": h["signature_type"],
                "ステータス": h["status"],
                "第一署名者": h["primary_signer"] or "-",
                "第一署名日時": datetime.fromisoformat(h["primary_time"]).strftime("%Y-%m-%d %H:%M") if h["primary_time"] else "-",
                "第二署名者": h["secondary_signer"] or "-",
                "第二署名日時": datetime.fromisoformat(h["secondary_time"]).strftime("%Y-%m-%d %H:%M") if h["secondary_time"] else "-",
                "作成日時": datetime.fromisoformat(h["created_at"]).strftime("%Y-%m-%d %H:%M"),
                "有効期限": datetime.fromisoformat(h["expires_at"]).strftime("%Y-%m-%d %H:%M") if h["expires_at"] else "-",
                "ブロックチェーン": "✅" if h.get("blockchain_verified") else "❌",
                "署名数": h.get("signature_count", 0)
            })
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # ステータス別の色分け
            def highlight_status(row):
                if row["ステータス"] == "completed":
                    return ['background-color: #d4edda'] * len(row)
                elif row["ステータス"] == "rejected":
                    return ['background-color: #f8d7da'] * len(row)
                elif row["ステータス"] == "partial":
                    return ['background-color: #fff3cd'] * len(row)
                elif row["ステータス"] == "expired":
                    return ['background-color: #e2e3e5'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df.style.apply(highlight_status, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # エクスポート機能
            if st.button("📤 CSV エクスポート"):
                csv = df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="ダウンロード",
                    data=csv,
                    file_name=f"signature_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("フィルター条件に一致する履歴がありません")
    
    def _render_signature_statistics(self):
        """署名統計情報"""
        st.subheader("📊 署名統計")
        
        history = self.signature_manager.get_secure_signature_history(1000)
        
        if not history:
            st.info("統計データがありません")
            return
        
        # 基本統計
        total_signatures = len(history)
        completed_signatures = len([h for h in history if h["status"] == "completed"])
        rejected_signatures = len([h for h in history if h["status"] == "rejected"])
        pending_signatures = len([h for h in history if h["status"] in ["pending", "partial"]])
        expired_signatures = len([h for h in history if h["status"] == "expired"])
        
        # メトリクス表示
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("総署名数", total_signatures)
        
        with col2:
            completion_rate = (completed_signatures / total_signatures * 100) if total_signatures > 0 else 0
            st.metric("完了率", f"{completion_rate:.1f}%")
        
        with col3:
            st.metric("拒否数", rejected_signatures)
        
        with col4:
            st.metric("待機中", pending_signatures)
        
        with col5:
            st.metric("期限切れ", expired_signatures)
        
        # チャート表示
        col1, col2 = st.columns(2)
        
        with col1:
            # ステータス別分布
            status_counts = {}
            for h in history:
                status_counts[h["status"]] = status_counts.get(h["status"], 0) + 1
            
            if status_counts:
                st.bar_chart(status_counts)
                st.caption("ステータス別分布")
        
        with col2:
            # 署名レベル別分布
            level_counts = {}
            for h in history:
                level = "二段階" if h["signature_level"] == "dual" else "一段階"
                level_counts[level] = level_counts.get(level, 0) + 1
            
            if level_counts:
                st.bar_chart(level_counts)
                st.caption("署名レベル別分布")
        
        # セキュリティメトリクス
        st.subheader("🔒 セキュリティメトリクス")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            blockchain_verified = len([h for h in history if h.get("blockchain_verified")])
            st.metric("ブロックチェーン検証済み", blockchain_verified)
        
        with col2:
            compliance_signatures = len([h for h in history if h.get("compliance_flags")])
            st.metric("コンプライアンス対応", compliance_signatures)
        
        with col3:
            multi_signatures = len([h for h in history if h.get("signature_count", 1) > 1])
            st.metric("多段階署名", multi_signatures)
        
        # 時系列分析
        st.subheader("📈 時系列分析")
        
        # 日別署名数
        daily_counts = {}
        for h in history:
            date = datetime.fromisoformat(h["created_at"]).date()
            date_str = date.strftime("%Y-%m-%d")
            daily_counts[date_str] = daily_counts.get(date_str, 0) + 1
        
        if daily_counts:
            # 日付順にソート
            sorted_dates = sorted(daily_counts.items())
            dates = [item[0] for item in sorted_dates]
            counts = [item[1] for item in sorted_dates]
            
            chart_data = pd.DataFrame({
                "日付": dates,
                "署名数": counts
            })
            
            st.line_chart(chart_data.set_index("日付"))
    
    def _render_signature_settings(self):
        """署名設定"""
        st.subheader("⚙️ 署名設定")
        
        # 署名ポリシー設定
        st.markdown("#### 署名ポリシー")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.checkbox("データエクスポート時の署名を必須とする", value=True)
            st.checkbox("レポート確定時の署名を必須とする", value=True)
            st.checkbox("設定変更時の署名を必須とする", value=False)
        
        with col2:
            st.checkbox("重要データ削除時の二段階署名を必須とする", value=True)
            st.checkbox("ユーザー管理操作時の署名を必須とする", value=True)
            st.checkbox("システム設定変更時の二段階署名を必須とする", value=False)
        
        st.markdown("---")
        
        # 署名者設定
        st.markdown("#### 署名者設定")
        
        # 管理者による署名者指定
        try:
            from auth_system import SecureUserDatabase
            db = SecureUserDatabase()
            users = db.list_users()
        except ImportError:
            # フォールバック
            users = {"admin": {"full_name": "Administrator"}, "analyst": {"full_name": "Analyst"}}
        
        authorized_signers = st.multiselect(
            "署名権限のあるユーザー",
            options=list(users.keys()),
            default=list(users.keys()),
            help="選択されたユーザーのみが電子署名を実行できます"
        )
        
        # 二段階署名設定
        st.markdown("#### 二段階署名設定")
        
        dual_signature_roles = st.multiselect(
            "二段階署名が可能なロール",
            options=["admin", "analyst", "manager"],
            default=["admin"],
            help="選択されたロールのユーザーが二段階署名に参加できます"
        )
        
        st.markdown("---")
        
        # セキュリティ設定
        st.markdown("#### セキュリティ設定")
        
        col1, col2 = st.columns(2)
        
        with col1:
            signature_timeout = st.number_input(
                "署名タイムアウト（時間）",
                min_value=1,
                max_value=168,  # 1週間
                value=24,
                help="署名要求の有効期限"
            )
            
            require_password_reentry = st.checkbox(
                "署名時のパスワード再入力を必須とする",
                value=True,
                help="セキュリティ強化のため推奨"
            )
        
        with col2:
            enable_blockchain = st.checkbox(
                "ブロックチェーン検証を有効にする",
                value=True,
                help="署名の改ざん防止機能"
            )
            
            enable_geolocation = st.checkbox(
                "地理的位置情報の記録を有効にする",
                value=False,
                help="署名時の位置情報を記録"
            )
        
        # データ管理
        st.markdown("---")
        st.markdown("#### データ管理")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 署名記録エクスポート"):
                export_data = self.signature_manager.export_secure_signature_records()
                st.download_button(
                    label="JSONファイルをダウンロード",
                    data=export_data,
                    file_name=f"signature_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 完全性検証レポート"):
                self._generate_integrity_report()
        
        with col3:
            if st.button("🗑️ 古い署名記録を削除"):
                st.warning("⚠️ この機能は管理者にお問い合わせください")
                st.info("データの完全性を保つため、署名記録の削除は慎重に行う必要があります")
        
        # 設定保存
        if st.button("💾 設定を保存"):
            # 設定をセッション状態に保存
            st.session_state.signature_settings = {
                "authorized_signers": authorized_signers,
                "dual_signature_roles": dual_signature_roles,
                "signature_timeout": signature_timeout,
                "require_password_reentry": require_password_reentry,
                "enable_blockchain": enable_blockchain,
                "enable_geolocation": enable_geolocation
            }
            st.success("✅ 設定を保存しました")
    
    def _generate_integrity_report(self):
        """完全性検証レポートを生成"""
        st.subheader("🔍 署名完全性検証レポート")
        
        history = self.signature_manager.get_secure_signature_history(100)
        
        if not history:
            st.info("検証する署名がありません")
            return
        
        verification_results = []
        
        with st.spinner("署名の完全性を検証中..."):
            for record in history:
                sig_id = record.get("signature_id")
                if sig_id:
                    result = self.signature_manager.verify_signature_integrity(sig_id)
                    verification_results.append({
                        "署名ID": sig_id[:8] + "...",
                        "操作タイプ": record["operation_type"],
                        "検証結果": result["status"],
                        "エラー数": len(result.get("errors", [])),
                        "警告数": len(result.get("warnings", [])),
                        "チェック数": len(result.get("checks", []))
                    })
        
        if verification_results:
            df = pd.DataFrame(verification_results)
            
            # 結果サマリー
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                verified_count = len([r for r in verification_results if r["検証結果"] == "verified"])
                st.metric("検証成功", verified_count)
            
            with col2:
                failed_count = len([r for r in verification_results if r["検証結果"] == "failed"])
                st.metric("検証失敗", failed_count)
            
            with col3:
                warning_count = len([r for r in verification_results if r["検証結果"] == "warning"])
                st.metric("警告あり", warning_count)
            
            with col4:
                total_errors = sum(r["エラー数"] for r in verification_results)
                st.metric("総エラー数", total_errors)
            
            # 詳細結果
            st.dataframe(df, use_container_width=True)
            
            # レポートダウンロード
            csv_data = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 検証レポートをダウンロード",
                data=csv_data,
                file_name=f"signature_integrity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# デモ機能
def demo_signature_required_operation():
    """署名が必要な操作のデモ"""
    from electronic_signature import require_secure_signature, SignatureLevel
    
    @require_secure_signature(
        operation_type="重要レポート確定",
        signature_level=SignatureLevel.DUAL
    )
    def finalize_important_report():
        st.success("🎉 重要レポートが正常に確定されました！")
        st.info("この操作は電子署名により承認されました。")
        return "レポート確定成功"
    
    if st.button("📋 重要レポートを確定（署名必要）"):
        finalize_important_report()

# 統合UI表示関数
def render_signature_demo_page():
    """電子署名デモページ"""
    st.header("🔏 セキュア電子署名システム デモ")
    
    st.markdown("""
    このページでは、セキュア強化された電子署名システムの機能をデモンストレーションします。
    重要な操作には電子署名が必要となり、適切な承認プロセスが実行されます。
    
    **セキュリティ機能:**
    - 🔐 RSA-2048暗号化による デジタル署名
    - 🛡️ 改ざん防止シール（HMAC-SHA256）
    - 🔗 ブロックチェーンハッシュによる完全性保証
    - 📍 地理的位置情報とIP追跡
    - 📋 完全な監査証跡
    - ⚖️ コンプライアンス対応（GDPR、HIPAA等）
    """)
    
    # デモ操作
    with st.expander("🎯 署名が必要な操作を実行"):
        demo_signature_required_operation()
    
    # 署名管理UI
    signature_mgmt_ui = SignatureManagementUI()
    signature_mgmt_ui.render_signature_management_page()
