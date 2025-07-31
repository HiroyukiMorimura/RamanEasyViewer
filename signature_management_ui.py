# -*- coding: utf-8 -*-
"""
署名管理UI
電子署名の管理・監視・履歴表示機能

Created for RamanEye Easy Viewer
@author: Signature Management UI System
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from electronic_signature import (
    SecureElectronicSignatureManager,
    SignatureUI,
    SignatureLevel,
    SignatureStatus
)

class SignatureManagementUI:
    """署名管理UIクラス"""
    
    def __init__(self):
        self.signature_manager = ElectronicSignatureManager()
        self.signature_ui = SignatureUI()
    
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
        
        from auth_system import AuthenticationManager
        auth_manager = AuthenticationManager()
        current_user = auth_manager.get_current_user()
        
        # ペンディング署名を取得
        pending_signatures = self.signature_manager.get_pending_signatures(current_user)
        
        if not pending_signatures:
            st.info("現在、署名待ちの操作はありません")
            return
        
        # データフレーム作成
        pending_data = []
        for sig in pending_signatures:
            pending_data.append({
                "操作タイプ": sig["operation_type"],
                "署名レベル": "二段階" if sig["level"] == "dual" else "一段階",
                "ステータス": sig["status"],
                "作成日時": datetime.fromisoformat(sig["created_at"]).strftime("%Y-%m-%d %H:%M"),
                "署名ID": sig["signature_id"]
            })
        
        df = pd.DataFrame(pending_data)
        
        # 署名選択
        selected_signature = st.selectbox(
            "署名する操作を選択してください:",
            options=df["署名ID"].tolist(),
            format_func=lambda x: f"{df[df['署名ID']==x]['操作タイプ'].iloc[0]} ({df[df['署名ID']==x]['作成日時'].iloc[0]})"
        )
        
        # 選択された署名の詳細表示と署名実行
        if selected_signature:
            st.markdown("---")
            
            # 現在のユーザー情報取得
            user_info = auth_manager.auth_manager.db.get_user(current_user)
            user_name = user_info.get("full_name", current_user)
            
            # 署名ダイアログ表示
            self.signature_ui.render_signature_dialog(
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
                ["すべて", "完了", "拒否", "部分署名", "待機中"]
            )
        
        with col3:
            if st.button("🔄 更新"):
                st.rerun()
        
        # 署名履歴を取得
        history = self.signature_manager.get_signature_history(limit)
        
        if not history:
            st.info("署名履歴がありません")
            return
        
        # ステータスフィルタリング
        if status_filter != "すべて":
            status_map = {
                "完了": "completed",
                "拒否": "rejected", 
                "部分署名": "partial",
                "待機中": "pending"
            }
            filter_status = status_map.get(status_filter)
            history = [h for h in history if h["status"] == filter_status]
        
        # データフレーム作成
        history_data = []
        for h in history:
            history_data.append({
                "操作タイプ": h["operation_type"],
                "署名レベル": "二段階" if h["level"] == "dual" else "一段階",
                "ステータス": h["status"],
                "第一署名者": h["primary_signer"] or "-",
                "第一署名日時": datetime.fromisoformat(h["primary_time"]).strftime("%Y-%m-%d %H:%M") if h["primary_time"] else "-",
                "第二署名者": h["secondary_signer"] or "-",
                "第二署名日時": datetime.fromisoformat(h["secondary_time"]).strftime("%Y-%m-%d %H:%M") if h["secondary_time"] else "-",
                "作成日時": datetime.fromisoformat(h["created_at"]).strftime("%Y-%m-%d %H:%M")
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
        
        history = self.signature_manager.get_signature_history(1000)
        
        if not history:
            st.info("統計データがありません")
            return
        
        # 基本統計
        total_signatures = len(history)
        completed_signatures = len([h for h in history if h["status"] == "completed"])
        rejected_signatures = len([h for h in history if h["status"] == "rejected"])
        pending_signatures = len([h for h in history if h["status"] in ["pending", "partial"]])
        
        # メトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総署名数", total_signatures)
        
        with col2:
            completion_rate = (completed_signatures / total_signatures * 100) if total_signatures > 0 else 0
            st.metric("完了率", f"{completion_rate:.1f}%")
        
        with col3:
            st.metric("拒否数", rejected_signatures)
        
        with col4:
            st.metric("待機中", pending_signatures)
        
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
                level = "二段階" if h["level"] == "dual" else "一段階"
                level_counts[level] = level_counts.get(level, 0) + 1
            
            if level_counts:
                st.bar_chart(level_counts)
                st.caption("署名レベル別分布")
        
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
        from auth_system import UserDatabase
        db = UserDatabase()
        users = db.list_users()
        
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
            options=["admin", "analyst"],
            default=["admin"],
            help="選択されたロールのユーザーが二段階署名に参加できます"
        )
        
        st.markdown("---")
        
        # データ管理
        st.markdown("#### データ管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 署名記録エクスポート"):
                export_data = self.signature_manager.export_signature_records()
                st.download_button(
                    label="JSONファイルをダウンロード",
                    data=export_data,
                    file_name=f"signature_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🗑️ 古い署名記録を削除"):
                # 実装は要検討（データの完全性を保つため）
                st.warning("この機能は管理者にお問い合わせください")
        
        # セキュリティ設定
        st.markdown("#### セキュリティ設定")
        
        signature_timeout = st.number_input(
            "署名タイムアウト（分）",
            min_value=5,
            max_value=1440,
            value=30,
            help="署名要求の有効期限"
        )
        
        require_password_reentry = st.checkbox(
            "署名時のパスワード再入力を必須とする",
            value=True,
            help="セキュリティ強化のため推奨"
        )
        
        # 設定保存
        if st.button("💾 設定を保存"):
            # 設定をセッション状態に保存
            st.session_state.signature_settings = {
                "authorized_signers": authorized_signers,
                "dual_signature_roles": dual_signature_roles,
                "signature_timeout": signature_timeout,
                "require_password_reentry": require_password_reentry
            }
            st.success("設定を保存しました")

# デモ機能
def demo_signature_required_operation():
    """署名が必要な操作のデモ"""
    from electronic_signature import require_signature, SignatureLevel
    
    @require_signature(
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
    st.header("🔏 電子署名システム デモ")
    
    st.markdown("""
    このページでは、電子署名システムの機能をデモンストレーションします。
    重要な操作には電子署名が必要となり、適切な承認プロセスが実行されます。
    """)
    
    # デモ操作
    with st.expander("🎯 署名が必要な操作を実行"):
        demo_signature_required_operation()
    
    # 署名管理UI
    signature_mgmt_ui = SignatureManagementUI()
    signature_mgmt_ui.render_signature_management_page()
