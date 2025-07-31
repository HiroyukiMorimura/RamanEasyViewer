# -*- coding: utf-8 -*-
"""
電子署名統合例
既存の解析機能に電子署名を統合する方法を示すサンプルコード

Created for RamanEye Easy Viewer
@author: Signature Integration Example
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# 電子署名システムの遅延インポート
def get_signature_system():
    """電子署名システムを遅延インポート"""
    try:
        from electronic_signature import require_signature, SignatureLevel
        return {
            'require_signature': require_signature,
            'SignatureLevel': SignatureLevel
        }
    except ImportError:
        # 電子署名システムが利用できない場合のフォールバック
        return None

# 既存の解析機能に電子署名を追加する例
class SecureAnalysisOperations:
    """セキュアな解析操作クラス"""
    
    def __init__(self):
        self.signature_system = get_signature_system()
    
    def export_spectrum_data(self, data, filename):
        """スペクトルデータのエクスポート（署名必要）"""
        if self.signature_system:
            # 電子署名システムが利用可能な場合
            @self.signature_system['require_signature'](
                operation_type="スペクトルデータエクスポート",
                signature_level=self.signature_system['SignatureLevel'].SINGLE
            )
            def _secure_export():
                return self._do_export(data, filename)
            
            return _secure_export()
        else:
            # フォールバック：警告を表示して通常のエクスポート
            st.warning("⚠️ 電子署名システムが利用できません。通常のエクスポートを実行します。")
            return self._do_export(data, filename)
    
    def _do_export(self, data, filename):
        """実際のエクスポート処理"""
        st.success(f"✅ スペクトルデータを正常にエクスポートしました: {filename}")
        
        # 実際のエクスポート処理をここに実装
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📥 ダウンロード",
            data=csv_data,
            file_name=filename,
            mime="text/csv"
        )
        
        return True
    
    def finalize_analysis_report(self, report_data):
        """解析レポートの確定（二段階署名必要）"""
        if self.signature_system:
            # 電子署名システムが利用可能な場合
            @self.signature_system['require_signature'](
                operation_type="解析レポート確定",
                signature_level=self.signature_system['SignatureLevel'].DUAL,
                required_signers=["admin", "analyst"]  # 特定のユーザーのみ署名可能
            )
            def _secure_finalize():
                return self._do_finalize(report_data)
            
            return _secure_finalize()
        else:
            # フォールバック：確認ダイアログを表示
            st.warning("⚠️ 電子署名システムが利用できません。")
            if st.button("⚠️ 確認：レポートを確定しますか？"):
                return self._do_finalize(report_data)
            return False
    
    def _do_finalize(self, report_data):
        """実際のレポート確定処理"""
        st.success("🎉 解析レポートが正常に確定されました！")
        st.info("この操作は電子署名により承認されました。")
        
        # レポート確定処理
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**確定日時**: {timestamp}")
        st.write(f"**レポートID**: RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        return "レポート確定成功"
    
    def update_database(self, update_data):
        """データベース更新（署名必要）"""
        if self.signature_system:
            # 電子署名システムが利用可能な場合
            @self.signature_system['require_signature'](
                operation_type="データベース更新",
                signature_level=self.signature_system['SignatureLevel'].SINGLE
            )
            def _secure_update():
                return self._do_update(update_data)
            
            return _secure_update()
        else:
            # フォールバック：確認ダイアログを表示
            st.warning("⚠️ 電子署名システムが利用できません。")
            if st.button("⚠️ 確認：データベースを更新しますか？"):
                return self._do_update(update_data)
            return False
    
    def _do_update(self, update_data):
        """実際のデータベース更新処理"""
        st.success("✅ データベースを正常に更新しました")
        
        # データベース更新処理をここに実装
        st.write("更新されたレコード数: 125")
        st.write("更新完了時刻:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return True
    
    def change_system_settings(self, settings):
        """システム設定変更（二段階署名必要）"""
        if self.signature_system:
            # 電子署名システムが利用可能な場合
            @self.signature_system['require_signature'](
                operation_type="システム設定変更",
                signature_level=self.signature_system['SignatureLevel'].DUAL
            )
            def _secure_change():
                return self._do_change_settings(settings)
            
            return _secure_change()
        else:
            # フォールバック：二重確認ダイアログ
            st.warning("⚠️ 電子署名システムが利用できません。")
            st.error("⚠️ 重要な設定変更には二段階承認が必要です。")
            
            confirm1 = st.checkbox("第一承認：設定変更を理解しました")
            confirm2 = st.checkbox("第二承認：責任を持って実行します")
            
            if confirm1 and confirm2:
                if st.button("⚠️ 最終確認：設定を変更する"):
                    return self._do_change_settings(settings)
            return False
    
    def _do_change_settings(self, settings):
        """実際の設定変更処理"""
        st.success("✅ システム設定を正常に変更しました")
        
        # 設定変更処理をここに実装
        st.write("変更された設定:")
        for key, value in settings.items():
            st.write(f"- {key}: {value}")
        
        return True

def demo_secure_operations():
    """セキュアな操作のデモンストレーション"""
    st.header("🔐 セキュアな操作デモ")
    
    # 電子署名システムの利用可能性をチェック
    signature_system = get_signature_system()
    
    if signature_system:
        st.success("✅ 電子署名システムが利用可能です")
    else:
        st.warning("⚠️ 電子署名システムが利用できません（フォールバックモードで動作）")
    
    operations = SecureAnalysisOperations()
    
    # タブで各操作を分離
    tab1, tab2, tab3, tab4 = st.tabs([
        "データエクスポート",
        "レポート確定", 
        "データベース更新",
        "システム設定"
    ])
    
    with tab1:
        st.subheader("📤 スペクトルデータエクスポート")
        if signature_system:
            st.info("この操作には一段階電子署名が必要です")
        else:
            st.warning("電子署名システムが無効です")
        
        # サンプルデータ作成
        sample_data = pd.DataFrame({
            "波数": [1000, 1100, 1200, 1300, 1400],
            "強度": [0.5, 0.8, 1.2, 0.9, 0.6]
        })
        
        st.write("エクスポート対象データ:")
        st.dataframe(sample_data, use_container_width=True)
        
        filename = st.text_input("ファイル名", value="spectrum_data.csv")
        
        if st.button("📤 データをエクスポート"):
            operations.export_spectrum_data(sample_data, filename)
    
    with tab2:
        st.subheader("📋 解析レポート確定")
        if signature_system:
            st.warning("この操作には二段階電子署名が必要です")
        else:
            st.error("電子署名システムが無効です（手動確認が必要）")
        
        report_data = {
            "分析対象": "Sample A",
            "ピーク数": 5,
            "主要ピーク": "1050 cm⁻¹",
            "信頼度": "95%"
        }
        
        st.write("確定対象レポート:")
        for key, value in report_data.items():
            st.write(f"**{key}**: {value}")
        
        if st.button("📋 レポートを確定"):
            operations.finalize_analysis_report(report_data)
    
    with tab3:
        st.subheader("💾 データベース更新")
        if signature_system:
            st.info("この操作には一段階電子署名が必要です")
        else:
            st.warning("電子署名システムが無効です")
        
        update_data = {
            "テーブル": "spectrum_records",
            "更新件数": 125,
            "更新内容": "品質スコア再計算"
        }
        
        st.write("更新内容:")
        for key, value in update_data.items():
            st.write(f"**{key}**: {value}")
        
        if st.button("💾 データベースを更新"):
            operations.update_database(update_data)
    
    with tab4:
        st.subheader("⚙️ システム設定変更")
        if signature_system:
            st.warning("この操作には二段階電子署名が必要です")
        else:
            st.error("電子署名システムが無効です（手動二重確認が必要）")
        
        settings = {}
        
        settings["自動バックアップ"] = st.checkbox("自動バックアップを有効化", value=True)
        settings["ログレベル"] = st.selectbox("ログレベル", ["INFO", "DEBUG", "WARNING", "ERROR"])
        settings["セッションタイムアウト"] = st.number_input("セッションタイムアウト（分）", value=60)
        
        if st.button("⚙️ 設定を変更"):
            operations.change_system_settings(settings)

# 署名統合のベストプラクティス
def signature_integration_guide():
    """電子署名統合ガイド"""
    st.header("📚 電子署名統合ガイド")
    
    st.markdown("""
    ## 🎯 電子署名を統合すべき操作
    
    ### **一段階署名が推奨される操作**:
    - ✅ データエクスポート
    - ✅ レポート生成
    - ✅ 設定の軽微な変更
    - ✅ データベースの通常更新
    
    ### **二段階署名が必要な操作**:
    - ⚠️ 重要レポートの確定
    - ⚠️ システム設定の重要な変更
    - ⚠️ データの削除・初期化
    - ⚠️ ユーザー権限の変更
    
    ## 🛠️ 実装方法
    
    ### **Step 1: 電子署名システムのインポート**
    ```python
    from electronic_signature import require_signature, SignatureLevel
    ```
    
    ### **Step 2: デコレータの追加**
    ```python
    @require_signature(
        operation_type="操作の説明",
        signature_level=SignatureLevel.SINGLE,  # または DUAL
        required_signers=["user1", "user2"]     # オプション
    )
    def your_function():
        # 実際の処理
        pass
    ```
    
    ### **Step 3: フォールバック機能の実装**
    ```python
    def secure_operation():
        if signature_system_available:
            # 電子署名付きで実行
            return secure_execute()
        else:
            # 手動確認で実行
            return manual_confirm_execute()
    ```
    
    ## 🔒 セキュリティ考慮事項
    
    - **パスワード再入力**: 本人確認の強化
    - **タイムスタンプ**: 改ざん防止
    - **理由記録**: 監査証跡の充実
    - **アクセス制御**: 適切な権限管理
    - **フォールバック**: システム障害時の対応
    
    ## 📋 コンプライアンス対応
    
    - **FDA 21 CFR Part 11**: 電子記録・電子署名規制
    - **ISO 17025**: 試験所認定基準
    - **GLP/GMP**: 医薬品品質管理基準
    - **J-SOX**: 内部統制報告制度
    
    ## 🚨 エラーハンドリング
    
    ### **電子署名システムが利用できない場合**:
    1. **警告表示**: ユーザーに状況を通知
    2. **手動確認**: 代替の承認プロセス
    3. **ログ記録**: 実行状況の記録
    4. **管理者通知**: システム管理者への報告
    
    ### **実装例**:
    ```python
    try:
        from electronic_signature import require_signature
        signature_available = True
    except ImportError:
        signature_available = False
        st.warning("電子署名システムが利用できません")
    
    if signature_available:
        @require_signature("重要操作", SignatureLevel.DUAL)
        def secure_operation():
            # セキュアな実行
            pass
    else:
        def secure_operation():
            # フォールバック実行
            if manual_confirmation():
                # 手動確認後の実行
                pass
    ```
    
    ## 🔧 統合のベストプラクティス
    
    1. **段階的導入**: 重要度の高い操作から順次適用
    2. **テスト環境**: 本番適用前の十分なテスト
    3. **ユーザー教育**: 操作方法の周知徹底
    4. **定期監査**: 署名記録の定期的な確認
    5. **バックアップ**: 署名記録の安全な保管
    
    ## 📊 運用監視
    
    - **署名成功率**: システムの正常性確認
    - **署名拒否率**: セキュリティ状況の把握
    - **応答時間**: パフォーマンス監視
    - **エラー率**: システム安定性の確認
    """)

# メイン関数から呼び出される関数
def render_signature_integration_demo():
    """電子署名統合デモのメイン関数"""
    st.set_page_config(
        page_title="電子署名統合デモ",
        page_icon="🔐",
        layout="wide"
    )
    
    st.title("🔐 電子署名システム統合デモ")
    
    st.markdown("""
    このページでは、電子署名システムの統合例と実装方法を示します。
    実際の業務操作に電子署名を統合する方法を学習できます。
    """)
    
    # 電子署名システムの状態表示
    signature_system = get_signature_system()
    
    if signature_system:
        st.success("✅ 電子署名システムが正常に動作しています")
    else:
        st.warning("⚠️ 電子署名システムが利用できません（デモモードで動作）")
        st.info("electronic_signature.py モジュールをインストールすると完全な機能が利用できます")
    
    tab1, tab2 = st.tabs(["セキュア操作デモ", "統合ガイド"])
    
    with tab1:
        demo_secure_operations()
    
    with tab2:
        signature_integration_guide()

# メイン実行部分（スタンドアロンで実行される場合）
if __name__ == "__main__":
    render_signature_integration_demo()
