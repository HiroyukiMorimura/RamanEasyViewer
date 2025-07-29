# -*- coding: utf-8 -*-
"""
ラマンスペクトルデータベース比較ツール
RamanEye Easy Viewer用のデータベース比較機能
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
from pathlib import Path
import pickle
from datetime import datetime
import io
from scipy import signal
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags

# 共通ユーティリティから必要な関数をインポート
from common_utils import *

class RamanDatabaseAnalyzer:
    def __init__(self, storage_dir=None, comparison_threshold=0.7):
        """
        初期化
        
        Parameters:
        storage_dir: スペクトルデータを保存するディレクトリ（Noneの場合は実行ディレクトリに設定）
        comparison_threshold: 詳細計算を実行する一致度の閾値
        """
        if storage_dir is None:
            # スクリプトが配置されているディレクトリにraman_spectraフォルダを作成
            script_dir = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd()
            storage_dir = script_dir / "raman_spectra"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.metadata_file = self.storage_dir / "metadata.pkl"
        self.comparison_threshold = comparison_threshold
        self.metadata = self.load_metadata()
        
    def load_metadata(self):
        """メタデータを読み込み"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def save_metadata(self):
        """メタデータを保存"""
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load_spectrum(self, spectrum_id):
        """保存されたスペクトルを読み込み"""
        if spectrum_id not in self.metadata:
            print(f"スペクトルIDが見つかりません: {spectrum_id}")
            return None
        
        spectrum_file = self.storage_dir / self.metadata[spectrum_id]['filename']
        with open(spectrum_file, 'rb') as f:
            return pickle.load(f)
    
    def load_all_spectra(self):
        """保存されている全スペクトルを読み込み"""
        spectra = {}
        for spectrum_id in self.metadata.keys():
            spectra[spectrum_id] = self.load_spectrum(spectrum_id)
        return spectra
    
    def downsample_spectrum(self, spectrum, pool_size=4):
        """
        スペクトルをプーリングによってダウンサンプリング
        
        Parameters:
        spectrum: 入力スペクトル
        pool_size: プーリングサイズ
        
        Returns:
        downsampled_spectrum: ダウンサンプリングされたスペクトル
        """
        # スペクトルの長さをプーリングサイズで割り切れるように調整
        trim_length = len(spectrum) - (len(spectrum) % pool_size)
        trimmed_spectrum = spectrum[:trim_length]
        
        # プーリング（平均値）
        reshaped = trimmed_spectrum.reshape(-1, pool_size)
        downsampled = np.mean(reshaped, axis=1)
        
        return downsampled
    
    def calculate_cross_correlation(self, spectrum1, spectrum2):
        """
        2つのスペクトル間の正規化相互相関を計算
        
        Parameters:
        spectrum1, spectrum2: 比較するスペクトル
        
        Returns:
        max_correlation: 最大相関値
        """
        # スペクトルを正規化
        spectrum1_norm = (spectrum1 - np.mean(spectrum1)) / np.std(spectrum1)
        spectrum2_norm = (spectrum2 - np.mean(spectrum2)) / np.std(spectrum2)
        
        # 相互相関を計算
        correlation = np.correlate(spectrum1_norm, spectrum2_norm, mode='full')
        
        # 正規化相関係数を計算
        max_correlation = np.max(correlation) / len(spectrum1_norm)
        
        return max_correlation

def init_database_session_state():
    """データベース比較用のセッション状態を初期化"""
    if 'database_analyzer' not in st.session_state:
        st.session_state.database_analyzer = RamanDatabaseAnalyzer()
    if 'uploaded_database_spectra' not in st.session_state:
        st.session_state.uploaded_database_spectra = []
    if 'comparison_results' not in st.session_state:
        st.session_state.comparison_results = None
    if 'top_spectra_ids' not in st.session_state:
        st.session_state.top_spectra_ids = []

def upload_and_process_database_files():
    """データベース比較用のファイルアップロードと処理"""
    st.header("📁 データベース比較用スペクトルファイルアップロード")
    
    uploaded_files = st.file_uploader(
        "ラマンスペクトルファイルをアップロード (CSV/TXT)",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="データベース比較用に複数のスペクトルファイルをアップロードしてください",
        key="database_file_uploader"
    )
    
    if uploaded_files:
        # 処理パラメータ設定
        col1, col2 = st.columns(2)
        with col1:
            start_wavenum = st.number_input("開始波数", value=200, step=10, key="db_start_wave")
            end_wavenum = st.number_input("終了波数", value=3000, step=10, key="db_end_wave")
        with col2:
            dssn_th = st.slider("ベースライン補正閾値", 0.001, 0.1, 0.01, step=0.001, key="db_dssn")
            savgol_wsize = st.selectbox("Savitzky-Golayウィンドウサイズ", [3, 5, 7, 9], index=0, key="db_savgol")
        
        if st.button("全ファイルを処理", type="primary", key="process_database_files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            st.session_state.uploaded_database_spectra = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"処理中: {uploaded_file.name}...")
                
                try:
                    # スペクトルを処理
                    result = process_spectrum_file(
                        uploaded_file,
                        start_wavenum=start_wavenum,
                        end_wavenum=end_wavenum,
                        dssn_th=dssn_th,
                        savgol_wsize=savgol_wsize
                    )
                    
                    wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                    
                    if wavenum is not None:
                        # スペクトルデータを保存
                        spectrum_id = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        
                        spectrum_data = {
                            'wavenum': wavenum,
                            'spectrum': BSremoval_specta_pos,  # ベースライン削除済み（移動平均無し）
                            'original_filename': file_name,
                            'file_type': file_type,
                            'processing_params': {
                                'start_wavenum': start_wavenum,
                                'end_wavenum': end_wavenum,
                                'dssn_th': dssn_th,
                                'savgol_wsize': savgol_wsize
                            }
                        }
                        
                        # アナライザーに保存
                        st.session_state.database_analyzer.metadata[spectrum_id] = {
                            'filename': f"{spectrum_id}.pkl",
                            'original_filename': file_name,
                            'file_type': file_type,
                            'wavenum_range': (wavenum[0], wavenum[-1]),
                            'data_points': len(wavenum),
                            'saved_at': datetime.now().isoformat()
                        }
                        
                        # データをメモリに保存
                        spectrum_file = st.session_state.database_analyzer.storage_dir / f"{spectrum_id}.pkl"
                        spectrum_file.parent.mkdir(exist_ok=True)
                        with open(spectrum_file, 'wb') as f:
                            pickle.dump(spectrum_data, f)
                        
                        st.session_state.uploaded_database_spectra.append({
                            'id': spectrum_id,
                            'filename': file_name
                        })
                        processed_count += 1
                    else:
                        st.error(f"{uploaded_file.name}の処理に失敗しました: スペクトルデータを抽出できませんでした")
                
                except Exception as e:
                    st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.database_analyzer.save_metadata()
            status_text.text(f"処理完了！ {processed_count}/{len(uploaded_files)} ファイルが正常に処理されました。")
            if processed_count > 0:
                st.success(f"{processed_count} 個のスペクトルファイルが正常に処理されました！")
            else:
                st.warning("ファイルが正常に処理されませんでした。ファイル形式を確認してください。")

def display_uploaded_database_spectra():
    """アップロードされたスペクトルを表示"""
    if st.session_state.uploaded_database_spectra:
        st.header("📊 アップロード済みスペクトル")
        
        # アップロードされたファイルのリスト表示
        spectra_df = pd.DataFrame(st.session_state.uploaded_database_spectra)
        spectra_df.columns = ['ID', 'ファイル名']
        st.dataframe(spectra_df, use_container_width=True)
        
        # 個別スペクトルの表示
        with st.expander("個別スペクトルを表示", expanded=False):
            selected_spectrum = st.selectbox(
                "表示するスペクトルを選択:",
                options=[spec['id'] for spec in st.session_state.uploaded_database_spectra],
                format_func=lambda x: next(spec['filename'] for spec in st.session_state.uploaded_database_spectra if spec['id'] == x),
                key="individual_spectrum_selector"
            )
            
            if selected_spectrum:
                spectrum_data = st.session_state.database_analyzer.load_spectrum(selected_spectrum)
                if spectrum_data:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=spectrum_data['wavenum'],
                        y=spectrum_data['spectrum'],
                        mode='lines',
                        name=spectrum_data['original_filename'],
                        line=dict(width=2)
                    ))
                    fig.update_layout(
                        title=f"スペクトル: {spectrum_data['original_filename']}",
                        xaxis_title="波数 (cm⁻¹)",
                        yaxis_title="強度",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

def run_database_comparison():
    """データベース比較を実行"""
    if len(st.session_state.uploaded_database_spectra) < 2:
        st.warning("データベース比較には少なくとも2つのスペクトルファイルをアップロードしてください。")
        return
    
    st.header("🔍 データベース比較")
    
    # 比較計算パラメータ
    col1, col2, col3 = st.columns(3)
    with col1:
        pool_size = st.selectbox("プーリングサイズ", [2, 4, 8], index=1, key="db_pool_size")
    with col2:
        comparison_threshold = st.slider("比較閾値", 0.5, 0.95, 0.7, step=0.05, key="db_threshold")
    with col3:
        max_spectra = len(st.session_state.uploaded_database_spectra)
        top_n = st.slider("解析対象スペクトル数", 2, min(max_spectra, 20), min(10, max_spectra), key="db_top_n")
    
    if st.button("比較計算を実行", type="primary", key="calculate_comparison_btn"):
        with st.spinner("比較マトリックスを計算中..."):
            # 全スペクトルを粗く計算して上位N個を選択
            all_spectra = st.session_state.database_analyzer.load_all_spectra()
            spectrum_ids = list(all_spectra.keys())
            
            if len(spectrum_ids) > top_n:
                st.info(f"全{len(spectrum_ids)}スペクトルから上位{top_n}スペクトルを事前選択中...")
                
                # 代表的なスペクトルを選択（例：最初のスペクトルとの一致度で選択）
                reference_spectrum = all_spectra[spectrum_ids[0]]['spectrum']
                matches = []
                
                for spectrum_id in spectrum_ids[1:]:
                    spectrum = all_spectra[spectrum_id]['spectrum']
                    # プーリングして粗い計算
                    pooled_ref = st.session_state.database_analyzer.downsample_spectrum(reference_spectrum, pool_size)
                    pooled_spec = st.session_state.database_analyzer.downsample_spectrum(spectrum, pool_size)
                    match = st.session_state.database_analyzer.calculate_cross_correlation(pooled_ref, pooled_spec)
                    matches.append((spectrum_id, match))
                
                # 一致度でソートして上位を選択
                matches.sort(key=lambda x: x[1], reverse=True)
                selected_ids = [spectrum_ids[0]] + [match[0] for match in matches[:top_n-1]]
                st.session_state.top_spectra_ids = selected_ids
                
                st.success(f"詳細解析用に上位{len(selected_ids)}スペクトルを選択しました")
            else:
                st.session_state.top_spectra_ids = spectrum_ids
            
            # 選択されたスペクトルで詳細な比較マトリックスを計算
            selected_spectra = {sid: all_spectra[sid] for sid in st.session_state.top_spectra_ids}
            
            n_spectra = len(selected_spectra)
            comparison_matrix = np.zeros((n_spectra, n_spectra))
            
            progress_bar = st.progress(0)
            total_pairs = n_spectra * (n_spectra - 1) // 2
            pair_count = 0
            
            for i, (id1, spec1) in enumerate(selected_spectra.items()):
                for j, (id2, spec2) in enumerate(selected_spectra.items()):
                    if i == j:
                        comparison_matrix[i, j] = 1.0
                        continue
                    elif i < j:
                        # 詳細な一致度計算
                        match = st.session_state.database_analyzer.calculate_cross_correlation(
                            spec1['spectrum'], spec2['spectrum']
                        )
                        comparison_matrix[i, j] = match
                        comparison_matrix[j, i] = match  # 対称行列
                        pair_count += 1
                        progress_bar.progress(pair_count / total_pairs)
            
            # DataFrame化
            comparison_df = pd.DataFrame(
                comparison_matrix,
                index=[spec['original_filename'] for spec in selected_spectra.values()],
                columns=[spec['original_filename'] for spec in selected_spectra.values()]
            )
            
            st.session_state.comparison_results = {
                'matrix': comparison_df,
                'spectrum_ids': st.session_state.top_spectra_ids,
                'spectra_data': selected_spectra
            }
            
            st.success("データベース比較が完了しました！")

def display_comparison_results():
    """比較結果を表示"""
    if st.session_state.comparison_results is None:
        return
    
    results = st.session_state.comparison_results
    comparison_matrix = results['matrix']
    
    st.header("📊 比較結果")
    
    # 統計サマリー
    col1, col2, col3, col4 = st.columns(4)
    
    # 対角線要素を除外して統計計算
    values = comparison_matrix.values.copy()
    np.fill_diagonal(values, np.nan)
    
    with col1:
        st.metric("平均一致度", f"{np.nanmean(values):.3f}")
    with col2:
        st.metric("最大一致度", f"{np.nanmax(values):.3f}")
    with col3:
        st.metric("最小一致度", f"{np.nanmin(values):.3f}")
    with col4:
        st.metric("標準偏差", f"{np.nanstd(values):.3f}")
    
    # 比較マトリックス（クリック可能）
    with st.expander("🔍 比較マトリックス（クリックして展開）", expanded=False):
        st.subheader("比較マトリックス")
        
        # ヒートマップを作成
        fig = px.imshow(
            comparison_matrix.values,
            labels=dict(x="スペクトル", y="スペクトル", color="一致スコア"),
            x=comparison_matrix.columns,
            y=comparison_matrix.index,
            color_continuous_scale="viridis",
            aspect="auto"
        )
        
        fig.update_layout(
            title="スペクトル比較マトリックス",
            xaxis_title="スペクトル",
            yaxis_title="スペクトル",
            height=600
        )
        
        # X軸ラベルを回転
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickangle=0)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 数値テーブルも表示
        st.subheader("数値表")
        st.dataframe(comparison_matrix.round(4), use_container_width=True)
    
    # 最高一致ペアの検索と表示
    st.header("⭐ 最高一致ペア")
    
    # 対角線を除外して最高一致度を検索
    matrix_values = comparison_matrix.values.copy()
    np.fill_diagonal(matrix_values, 0)  # 対角線を0に
    
    max_match = np.max(matrix_values)
    max_indices = np.unravel_index(np.argmax(matrix_values), matrix_values.shape)
    
    most_matched_pair = (
        comparison_matrix.index[max_indices[0]], 
        comparison_matrix.columns[max_indices[1]]
    )
    
    spectrum1_name = most_matched_pair[0]
    spectrum2_name = most_matched_pair[1]
    
    st.success(f"**最高一致スコア: {max_match:.4f}**")
    st.info(f"**ペア: {spectrum1_name} ↔ {spectrum2_name}**")
    
    # 最高一致ペアのスペクトルを表示
    col1, col2 = st.columns(2)
    
    # スペクトルIDを取得
    spectrum_ids = results['spectrum_ids']
    spectra_data = results['spectra_data']
    
    spectrum1_id = spectrum_ids[max_indices[0]]
    spectrum2_id = spectrum_ids[max_indices[1]]
    
    spectrum1_data = spectra_data[spectrum1_id]
    spectrum2_data = spectra_data[spectrum2_id]
    
    with col1:
        st.subheader(f"スペクトル1: {spectrum1_name}")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=spectrum1_data['wavenum'],
            y=spectrum1_data['spectrum'],
            mode='lines',
            name=spectrum1_name,
            line=dict(color='blue', width=2)
        ))
        fig1.update_layout(
            xaxis_title="波数 (cm⁻¹)",
            yaxis_title="強度",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader(f"スペクトル2: {spectrum2_name}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=spectrum2_data['wavenum'],
            y=spectrum2_data['spectrum'],
            mode='lines',
            name=spectrum2_name,
            line=dict(color='red', width=2)
        ))
        fig2.update_layout(
            xaxis_title="波数 (cm⁻¹)",
            yaxis_title="強度",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 重ね合わせ表示
    st.subheader("重ね合わせ比較")
    fig_overlay = go.Figure()
    
    fig_overlay.add_trace(go.Scatter(
        x=spectrum1_data['wavenum'],
        y=spectrum1_data['spectrum'],
        mode='lines',
        name=spectrum1_name,
        line=dict(color='blue', width=2)
    ))
    
    fig_overlay.add_trace(go.Scatter(
        x=spectrum2_data['wavenum'],
        y=spectrum2_data['spectrum'],
        mode='lines',
        name=spectrum2_name,
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig_overlay.update_layout(
        title=f"重ね合わせ: {spectrum1_name} vs {spectrum2_name} (一致スコア: {max_match:.4f})",
        xaxis_title="波数 (cm⁻¹)",
        yaxis_title="強度",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_overlay, use_container_width=True)
    
    # 高一致ペアのリスト
    with st.expander("📋 全高一致ペア", expanded=False):
        st.subheader("高一致ペア (> 0.8)")
        
        high_match_pairs = []
        for i in range(len(comparison_matrix)):
            for j in range(i+1, len(comparison_matrix)):
                match_value = comparison_matrix.iloc[i, j]
                if match_value > 0.8:
                    high_match_pairs.append({
                        'スペクトル1': comparison_matrix.index[i],
                        'スペクトル2': comparison_matrix.columns[j],
                        '一致スコア': match_value
                    })
        
        if high_match_pairs:
            high_match_df = pd.DataFrame(high_match_pairs)
            high_match_df = high_match_df.sort_values('一致スコア', ascending=False)
            st.dataframe(high_match_df, use_container_width=True)
        else:
            st.info("一致スコア > 0.8 のペアが見つかりませんでした。")

def export_comparison_results():
    """比較結果のエクスポート"""
    if st.session_state.comparison_results is None:
        return
    
    st.header("💾 結果をエクスポート")
    
    results = st.session_state.comparison_results
    comparison_matrix = results['matrix']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("比較マトリックスをダウンロード (CSV)", key="download_comparison_csv"):
            csv_buffer = io.StringIO()
            comparison_matrix.to_csv(csv_buffer)
            
            st.download_button(
                label="📥 CSVダウンロード",
                data=csv_buffer.getvalue(),
                file_name=f"比較マトリックス_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_comp_matrix"
            )
    
    with col2:
        if st.button("処理済みスペクトル情報をダウンロード", key="download_spectra_info"):
            spectra_info = []
            for spectrum_id in results['spectrum_ids']:
                spectrum_data = results['spectra_data'][spectrum_id]
                spectra_info.append({
                    'スペクトルID': spectrum_id,
                    '元ファイル名': spectrum_data['original_filename'],
                    'ファイルタイプ': spectrum_data['file_type'],
                    '波数範囲': f"{spectrum_data['wavenum'][0]:.1f} - {spectrum_data['wavenum'][-1]:.1f}",
                    'データ点数': len(spectrum_data['wavenum'])
                })
            
            info_df = pd.DataFrame(spectra_info)
            csv_buffer = io.StringIO()
            info_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 スペクトル情報ダウンロード",
                data=csv_buffer.getvalue(),
                file_name=f"スペクトル情報_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_spec_info"
            )

def database_comparison_mode():
    """データベース比較モードのメイン関数"""
    # セッション状態を初期化
    init_database_session_state()
    
    st.header("🔍 スペクトルデータベース比較")
    st.markdown("---")
    
    # タブを作成
    tab1, tab2, tab3 = st.tabs(["📁 アップロード・処理", "🔍 データベース比較", "📊 結果・エクスポート"])
    
    with tab1:
        upload_and_process_database_files()
        display_uploaded_database_spectra()
    
    with tab2:
        run_database_comparison()
    
    with tab3:
        display_comparison_results()
        export_comparison_results()

# メイン実行部分（単独実行時用）
if __name__ == "__main__":
    st.set_page_config(
        page_title="ラマンスペクトルデータベース比較",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 ラマンスペクトルデータベース比較")
    database_comparison_mode()
