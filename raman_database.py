# -*- coding: utf-8 -*-
"""
ラマンスペクトルデータベース比較ツール（統合版）
RamanEye Easy Viewer用のデータベース比較機能（WebUI統合）
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
            print(f"Spectrum ID not found: {spectrum_id}")
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
    st.header("📁 Spectrum File Upload for Database Comparison")
    
    uploaded_files = st.file_uploader(
        "Upload Raman spectrum files (CSV/TXT)",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple spectrum files for database comparison",
        key="database_file_uploader"
    )
    
    if uploaded_files:
        # 処理パラメータ設定
        col1, col2 = st.columns(2)
        with col1:
            start_wavenum = st.number_input("Start Wavenumber", value=200, step=10, key="db_start_wave")
            end_wavenum = st.number_input("End Wavenumber", value=3000, step=10, key="db_end_wave")
        with col2:
            dssn_th = st.slider("Baseline Correction Threshold", 0.001, 0.1, 0.01, step=0.001, key="db_dssn")
            savgol_wsize = st.selectbox("Savitzky-Golay Window Size", [3, 5, 7, 9], index=0, key="db_savgol")
        
        # デバッグ用：ファイル内容確認オプション
        debug_mode = st.checkbox("Enable debug mode (show file contents)", key="debug_mode")
        
        if st.button("Process All Files", type="primary", key="process_database_files"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            st.session_state.uploaded_database_spectra = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # デバッグモード：ファイル内容を表示
                    if debug_mode:
                        st.write(f"**Debug info for {uploaded_file.name}:**")
                        uploaded_file.seek(0)
                        
                        # ファイルタイプを確認
                        try:
                            data_preview = read_csv_file(uploaded_file, uploaded_file.name.split('.')[-1].lower())
                            if data_preview is not None:
                                file_type = detect_file_type(data_preview)
                                st.write(f"Detected file type: {file_type}")
                                st.write("First few rows:")
                                st.dataframe(data_preview.head())
                                st.write("Column names:")
                                st.write(list(data_preview.columns))
                                
                                # Wasatchファイルの場合は特別な処理
                                if file_type == "wasatch":
                                    st.write("**Wasatch file detected - checking header structure:**")
                                    uploaded_file.seek(0)
                                    
                                    # 複数のskiprows値を試す
                                    for skiprows in [0, 10, 20, 30, 40, 46, 50]:
                                        try:
                                            test_data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=skiprows, nrows=5)
                                            st.write(f"With skiprows={skiprows}:")
                                            st.write(f"Columns: {list(test_data.columns)}")
                                            if 'Wavelength' in test_data.columns:
                                                st.success(f"✅ Found 'Wavelength' column with skiprows={skiprows}")
                                                break
                                            uploaded_file.seek(0)
                                        except Exception as e:
                                            st.write(f"skiprows={skiprows}: Error - {str(e)}")
                                            uploaded_file.seek(0)
                            else:
                                st.error(f"Could not read file: {uploaded_file.name}")
                        except Exception as e:
                            st.error(f"Error analyzing file: {str(e)}")
                        
                        uploaded_file.seek(0)
                    
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
                        st.error(f"Failed to process {uploaded_file.name}: Could not extract spectrum data")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    if debug_mode:
                        st.exception(e)
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            st.session_state.database_analyzer.save_metadata()
            status_text.text(f"Processing complete! {processed_count}/{len(uploaded_files)} files processed successfully.")
            if processed_count > 0:
                st.success(f"Successfully processed {processed_count} spectrum files!")
            else:
                st.warning("No files were processed successfully. Please check the file formats and try debug mode.")

def display_uploaded_database_spectra():
    """アップロードされたスペクトルを表示"""
    if st.session_state.uploaded_database_spectra:
        st.header("📊 Uploaded Spectra")
        
        # アップロードされたファイルのリスト表示
        spectra_df = pd.DataFrame(st.session_state.uploaded_database_spectra)
        st.dataframe(spectra_df, use_container_width=True)
        
        # 個別スペクトルの表示
        with st.expander("View Individual Spectra", expanded=False):
            selected_spectrum = st.selectbox(
                "Select spectrum to view:",
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
                        title=f"Spectrum: {spectrum_data['original_filename']}",
                        xaxis_title="Wavenumber (cm⁻¹)",
                        yaxis_title="Intensity",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

def run_database_comparison():
    """データベース比較を実行"""
    if len(st.session_state.uploaded_database_spectra) < 2:
        st.warning("Please upload at least 2 spectrum files for database comparison.")
        return
    
    st.header("🔍 Database Comparison")
    
    # 比較計算パラメータ
    col1, col2, col3 = st.columns(3)
    with col1:
        pool_size = st.selectbox("Pooling Size", [2, 4, 8], index=1, key="db_pool_size")
    with col2:
        comparison_threshold = st.slider("Comparison Threshold", 0.5, 0.95, 0.7, step=0.05, key="db_threshold")
    with col3:
        max_spectra = len(st.session_state.uploaded_database_spectra)
        top_n = st.slider("Number of spectra for analysis", 2, min(max_spectra, 20), min(10, max_spectra), key="db_top_n")
    
    if st.button("Calculate Comparison", type="primary", key="calculate_comparison_btn"):
        with st.spinner("Calculating comparison matrix..."):
            # 全スペクトルを粗く計算して上位N個を選択
            all_spectra = st.session_state.database_analyzer.load_all_spectra()
            spectrum_ids = list(all_spectra.keys())
            
            if len(spectrum_ids) > top_n:
                st.info(f"Pre-selecting top {top_n} spectra from {len(spectrum_ids)} total spectra...")
                
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
                
                st.success(f"Selected top {len(selected_ids)} spectra for detailed analysis")
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
            
            st.success("Database comparison completed!")

def display_comparison_results():
    """比較結果を表示"""
    if st.session_state.comparison_results is None:
        return
    
    results = st.session_state.comparison_results
    comparison_matrix = results['matrix']
    
    st.header("📊 Comparison Results")
    
    # 統計サマリー
    col1, col2, col3, col4 = st.columns(4)
    
    # 対角線要素を除外して統計計算
    values = comparison_matrix.values.copy()
    np.fill_diagonal(values, np.nan)
    
    with col1:
        st.metric("Average Match", f"{np.nanmean(values):.3f}")
    with col2:
        st.metric("Max Match", f"{np.nanmax(values):.3f}")
    with col3:
        st.metric("Min Match", f"{np.nanmin(values):.3f}")
    with col4:
        st.metric("Std Deviation", f"{np.nanstd(values):.3f}")
    
    # 比較マトリックス（クリック可能）
    with st.expander("🔍 Comparison Matrix (Click to expand)", expanded=False):
        st.subheader("Comparison Matrix")
        
        # ヒートマップを作成
        fig = px.imshow(
            comparison_matrix.values,
            labels=dict(x="Spectrum", y="Spectrum", color="Match Score"),
            x=comparison_matrix.columns,
            y=comparison_matrix.index,
            color_continuous_scale="viridis",
            aspect="auto"
        )
        
        fig.update_layout(
            title="Spectrum Comparison Matrix",
            xaxis_title="Spectrum",
            yaxis_title="Spectrum",
            height=600
        )
        
        # X軸ラベルを回転
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(tickangle=0)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 数値テーブルも表示
        st.subheader("Numerical Values")
        st.dataframe(comparison_matrix.round(4), use_container_width=True)
    
    # 最高一致ペアの検索と表示
    st.header("⭐ Highest Match Pair")
    
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
    
    st.success(f"**Highest match score: {max_match:.4f}**")
    st.info(f"**Pair: {spectrum1_name} ↔ {spectrum2_name}**")
    
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
        st.subheader(f"Spectrum 1: {spectrum1_name}")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=spectrum1_data['wavenum'],
            y=spectrum1_data['spectrum'],
            mode='lines',
            name=spectrum1_name,
            line=dict(color='blue', width=2)
        ))
        fig1.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity",
            height=400
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader(f"Spectrum 2: {spectrum2_name}")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=spectrum2_data['wavenum'],
            y=spectrum2_data['spectrum'],
            mode='lines',
            name=spectrum2_name,
            line=dict(color='red', width=2)
        ))
        fig2.update_layout(
            xaxis_title="Wavenumber (cm⁻¹)",
            yaxis_title="Intensity",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # 重ね合わせ表示
    st.subheader("Overlay Comparison")
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
        title=f"Overlay: {spectrum1_name} vs {spectrum2_name} (Match Score: {max_match:.4f})",
        xaxis_title="Wavenumber (cm⁻¹)",
        yaxis_title="Intensity",
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    st.plotly_chart(fig_overlay, use_container_width=True)
    
    # 高一致ペアのリスト
    with st.expander("📋 All High Match Pairs", expanded=False):
        st.subheader("High Match Pairs (> 0.8)")
        
        high_match_pairs = []
        for i in range(len(comparison_matrix)):
            for j in range(i+1, len(comparison_matrix)):
                match_value = comparison_matrix.iloc[i, j]
                if match_value > 0.8:
                    high_match_pairs.append({
                        'Spectrum 1': comparison_matrix.index[i],
                        'Spectrum 2': comparison_matrix.columns[j],
                        'Match Score': match_value
                    })
        
        if high_match_pairs:
            high_match_df = pd.DataFrame(high_match_pairs)
            high_match_df = high_match_df.sort_values('Match Score', ascending=False)
            st.dataframe(high_match_df, use_container_width=True)
        else:
            st.info("No pairs with match score > 0.8 found.")

def export_comparison_results():
    """比較結果のエクスポート"""
    if st.session_state.comparison_results is None:
        return
    
    st.header("💾 Export Results")
    
    results = st.session_state.comparison_results
    comparison_matrix = results['matrix']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Download Comparison Matrix (CSV)", key="download_comparison_csv"):
            csv_buffer = io.StringIO()
            comparison_matrix.to_csv(csv_buffer)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_buffer.getvalue(),
                file_name=f"comparison_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_comp_matrix"
            )
    
    with col2:
        if st.button("Download Processed Spectra Info", key="download_spectra_info"):
            spectra_info = []
            for spectrum_id in results['spectrum_ids']:
                spectrum_data = results['spectra_data'][spectrum_id]
                spectra_info.append({
                    'Spectrum ID': spectrum_id,
                    'Original Filename': spectrum_data['original_filename'],
                    'File Type': spectrum_data['file_type'],
                    'Wavenumber Range': f"{spectrum_data['wavenum'][0]:.1f} - {spectrum_data['wavenum'][-1]:.1f}",
                    'Data Points': len(spectrum_data['wavenum'])
                })
            
            info_df = pd.DataFrame(spectra_info)
            csv_buffer = io.StringIO()
            info_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 Download Spectra Info",
                data=csv_buffer.getvalue(),
                file_name=f"spectra_info_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_spec_info"
            )

def database_comparison_mode():
    """データベース比較モードのメイン関数"""
    # セッション状態を初期化
    init_database_session_state()
    
    st.header("🔍 Spectrum Database Comparison")
    st.markdown("---")
    
    # タブを作成
    tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "🔍 Database Comparison", "📊 Results & Export"])
    
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
        page_title="Raman Spectrum Database Comparison",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🔬 Raman Spectrum Database Comparison")
    database_comparison_mode()
