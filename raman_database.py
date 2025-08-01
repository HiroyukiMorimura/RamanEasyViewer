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
    
    def interpolate_spectrum_to_common_grid(self, wavenum, spectrum, common_wavenum):
        """
        スペクトルを共通の波数グリッドに補間
        
        Parameters:
        wavenum: 元の波数配列
        spectrum: 元のスペクトル配列
        common_wavenum: 共通の波数グリッド
        
        Returns:
        interpolated_spectrum: 補間されたスペクトル
        """
        # 線形補間を使用して共通グリッドに補間
        interpolated_spectrum = np.interp(common_wavenum, wavenum, spectrum)
        return interpolated_spectrum
    
    def normalize_spectrum(self, spectrum):
        """
        スペクトルを正規化
        
        Parameters:
        spectrum: 入力スペクトル
        
        Returns:
        normalized_spectrum: 正規化されたスペクトル
        """
        # 平均を0、標準偏差を1に正規化
        mean_val = np.mean(spectrum)
        std_val = np.std(spectrum)
        
        if std_val == 0:
            return np.zeros_like(spectrum)
        
        normalized_spectrum = (spectrum - mean_val) / std_val
        return normalized_spectrum
    
    def calculate_cosine_similarity(self, spectrum1, spectrum2):
        """
        2つのスペクトル間のコサイン類似度を計算
        
        Parameters:
        spectrum1, spectrum2: 比較するスペクトル（同じ長さである必要がある）
        
        Returns:
        cosine_similarity: コサイン類似度 (0-1の範囲)
        """
        # ベクトルの内積を計算
        dot_product = np.dot(spectrum1, spectrum2)
        
        # ベクトルのノルムを計算
        norm1 = np.linalg.norm(spectrum1)
        norm2 = np.linalg.norm(spectrum2)
        
        # ゼロベクトルの場合は0を返す
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # コサイン類似度を計算
        cosine_sim = dot_product / (norm1 * norm2)
        
        # 負の値を0にクリップ（類似度として0-1の範囲にする）
        cosine_sim = max(0.0, cosine_sim)
        
        return cosine_sim
    
    def calculate_pearson_correlation(self, spectrum1, spectrum2):
        """
        2つのスペクトル間のピアソン相関係数を計算
        
        Parameters:
        spectrum1, spectrum2: 比較するスペクトル（同じ長さである必要がある）
        
        Returns:
        correlation: ピアソン相関係数の絶対値 (0-1の範囲)
        """
        # ピアソン相関係数を計算
        correlation_matrix = np.corrcoef(spectrum1, spectrum2)
        correlation = correlation_matrix[0, 1]
        
        # NaNの場合は0を返す
        if np.isnan(correlation):
            return 0.0
        
        # 絶対値を取って0-1の範囲にする
        return abs(correlation)
    
    def calculate_spectrum_similarity(self, spectrum1_data, spectrum2_data, method='cosine'):
        """
        2つのスペクトル間の類似度を計算（波数軸の統一を含む）
        
        Parameters:
        spectrum1_data: 第1スペクトルのデータ辞書 ({'wavenum': array, 'spectrum': array})
        spectrum2_data: 第2スペクトルのデータ辞書 ({'wavenum': array, 'spectrum': array})
        method: 類似度計算手法 ('cosine', 'pearson', 'cross_correlation')
        
        Returns:
        similarity: 類似度スコア (0-1の範囲)
        """
        wavenum1 = spectrum1_data['wavenum']
        spectrum1 = spectrum1_data['spectrum']
        wavenum2 = spectrum2_data['wavenum']
        spectrum2 = spectrum2_data['spectrum']
        
        # 共通の波数範囲を決定
        min_wavenum = max(wavenum1.min(), wavenum2.min())
        max_wavenum = min(wavenum1.max(), wavenum2.max())
        
        # 共通範囲が存在しない場合は類似度0
        if min_wavenum >= max_wavenum:
            return 0.0
        
        # 共通の波数グリッドを作成（より細かい解像度に合わせる）
        resolution1 = len(wavenum1) / (wavenum1.max() - wavenum1.min())
        resolution2 = len(wavenum2) / (wavenum2.max() - wavenum2.min())
        target_resolution = max(resolution1, resolution2)
        
        num_points = int((max_wavenum - min_wavenum) * target_resolution)
        num_points = max(100, min(num_points, 2000))  # 100-2000点の範囲に制限
        
        common_wavenum = np.linspace(min_wavenum, max_wavenum, num_points)
        
        # 両スペクトルを共通グリッドに補間
        interp_spectrum1 = self.interpolate_spectrum_to_common_grid(wavenum1, spectrum1, common_wavenum)
        interp_spectrum2 = self.interpolate_spectrum_to_common_grid(wavenum2, spectrum2, common_wavenum)
        
        # スペクトルを正規化
        norm_spectrum1 = self.normalize_spectrum(interp_spectrum1)
        norm_spectrum2 = self.normalize_spectrum(interp_spectrum2)
        
        # 指定された手法で類似度を計算
        if method == 'cosine':
            similarity = self.calculate_cosine_similarity(norm_spectrum1, norm_spectrum2)
        elif method == 'pearson':
            similarity = self.calculate_pearson_correlation(norm_spectrum1, norm_spectrum2)
        elif method == 'cross_correlation':
            similarity = self.calculate_cross_correlation(norm_spectrum1, norm_spectrum2)
        else:
            # デフォルトはコサイン類似度
            similarity = self.calculate_cosine_similarity(norm_spectrum1, norm_spectrum2)
        
        return similarity
    
    def calculate_cross_correlation(self, spectrum1, spectrum2):
        """
        2つのスペクトル間の正規化相互相関を計算（改良版）
        
        Parameters:
        spectrum1, spectrum2: 比較するスペクトル（同じ長さである必要がある）
        
        Returns:
        max_correlation: 最大相関値 (0-1の範囲)
        """
        # 既に正規化されているスペクトルを前提とする
        
        # 相互相関を計算
        correlation = np.correlate(spectrum1, spectrum2, mode='full')
        
        # 正規化相関係数を計算
        max_correlation = np.max(correlation) / len(spectrum1)
        
        # 0-1の範囲にクリップ
        max_correlation = max(0.0, min(1.0, max_correlation))
        
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
    # st.header("データベース比較")
    
    # パラメータ設定をサイドバーに移動
    st.sidebar.subheader("🔧 処理パラメータ")
    start_wavenum = st.sidebar.number_input(
        "波数（開始）を入力してください:", 
        min_value=-200, 
        max_value=4800, 
        value=400, 
        step=100,
        key="db_start_wave"
    )
    end_wavenum = st.sidebar.number_input(
        "波数（終了）を入力してください:", 
        min_value=-200, 
        max_value=4800, 
        value=2000, 
        step=100,
        key="db_end_wave"
    )
    
    dssn_th_input = st.sidebar.number_input(
        "ベースライン補正閾値を入力してください:", 
        min_value=1, 
        max_value=10000, 
        value=1000, 
        step=1,
        key="db_dssn_input"
    )
    dssn_th = dssn_th_input / 10000000
    
    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（単数）",
        type=['csv', 'txt'],
        accept_multiple_files=False,
        key="database_file_uploader"
    )
    
    # 各ファイルのデータを格納するリスト
    all_spectrum_data = []
    
    if uploaded_files:
        processed_count = 0
        # 既存のスペクトルをクリアしないように修正
        if not hasattr(st.session_state, 'uploaded_database_spectra'):
            st.session_state.uploaded_database_spectra = []
        
        # 色の設定
        selected_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'black']
        Fsize = 14
        
        for i, uploaded_file in enumerate(uploaded_files):
            try:
                # スペクトルを処理（savgol_wsizeはデフォルト値3を使用）
                result = process_spectrum_file(
                    uploaded_file,
                    start_wavenum=start_wavenum,
                    end_wavenum=end_wavenum,
                    dssn_th=dssn_th,
                    savgol_wsize=3  # 固定値
                )
                
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                
                if wavenum is not None:
                    # スペクトルデータを保存（データベース用）
                    spectrum_id = f"{file_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                    
                    spectrum_data = {
                        'wavenum': wavenum,
                        'spectrum': BSremoval_specta_pos,  # ベースライン削除済み（移動平均無し）
                        'original_filename': file_name,
                        'file_type': file_type,
                        'processing_params': {
                            'start_wavenum': start_wavenum,
                            'end_wavenum': end_wavenum,
                            'dssn_th': dssn_th
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
                    
                    # 表示用データも追加
                    file_data = {
                        'wavenum': wavenum,
                        'raw_spectrum': spectra,
                        'baseline_removed': BSremoval_specta_pos,
                        'file_name': file_name,
                        'file_type': file_type
                    }
                    all_spectrum_data.append(file_data)
                    
                    processed_count += 1
                else:
                    st.error(f"{uploaded_file.name}の処理に失敗しました: スペクトルデータを抽出できませんでした")
            
            except Exception as e:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {str(e)}")
        
        st.session_state.database_analyzer.save_metadata()
        
        if processed_count > 0:
            # スペクトル表示（spectrum_analysis.pyと同じスタイル）
            import matplotlib.pyplot as plt
            
            # 元のスペクトルを重ねてプロット
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, data in enumerate(all_spectrum_data):
                ax.plot(data['wavenum'], data['raw_spectrum'], 
                       linestyle='-', 
                       color=selected_colors[i % len(selected_colors)], 
                       label=f"{data['file_name']} ({data['file_type']})")
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Raw Spectra', fontsize=Fsize)
            ax.legend(title="Spectra", loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Raw spectraのCSVダウンロード
            raw_csv_data = create_interpolated_csv(all_spectrum_data, 'raw_spectrum')
            st.download_button(
                label="Raw Spectra CSV ダウンロード",
                data=raw_csv_data,
                file_name=f'raw_spectra_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv',
                key="download_raw_csv"
            )
            
            # ベースライン補正後のスペクトルを重ねてプロット（legendなし）
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, data in enumerate(all_spectrum_data):
                ax.plot(data['wavenum'], data['baseline_removed'], 
                       linestyle='-', 
                       color=selected_colors[i % len(selected_colors)])
            
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Baseline Removed Spectra', fontsize=Fsize)
            plt.tight_layout()
            st.pyplot(fig)
            
            # pickleファイルとして保存
            pickle_data = {
                'spectra_data': all_spectrum_data,
                'processing_params': {
                    'start_wavenum': start_wavenum,
                    'end_wavenum': end_wavenum,
                    'dssn_th': dssn_th
                },
                'saved_at': datetime.now().isoformat()
            }
            pickle_buffer = pickle.dumps(pickle_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="スペクトルデータ保存 (pickle)",
                    data=pickle_buffer,
                    file_name=f'spectrum_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl',
                    mime='application/octet-stream',
                    key="download_pickle"
                )
            
            with col2:
                # 既存データに追加する機能
                st.subheader("既存データに追加")
                existing_pickle = st.file_uploader(
                    "既存のpickleファイルを選択",
                    type=['pkl'],
                    help="新しいスペクトルを既存のデータに追加します",
                    key="existing_pickle_uploader"
                )
                
                if existing_pickle is not None and st.button("📝 スペクトルデータ追加", type="primary", key="add_to_existing"):
                    try:
                        # 既存データを読み込み
                        existing_data = pickle.load(existing_pickle)
                        
                        if 'spectra_data' in existing_data:
                            # 既存データに新しいデータを追加
                            combined_spectra_data = existing_data['spectra_data'] + all_spectrum_data
                            
                            # 新しいpickleデータを作成
                            combined_pickle_data = {
                                'spectra_data': combined_spectra_data,
                                'processing_params': {
                                    'start_wavenum': start_wavenum,
                                    'end_wavenum': end_wavenum,
                                    'dssn_th': dssn_th
                                },
                                'saved_at': datetime.now().isoformat(),
                                'original_count': len(existing_data['spectra_data']),
                                'added_count': len(all_spectrum_data),
                                'total_count': len(combined_spectra_data)
                            }
                            
                            combined_pickle_buffer = pickle.dumps(combined_pickle_data)
                            
                            st.success(f"✅ {len(existing_data['spectra_data'])}個の既存スペクトル + {len(all_spectrum_data)}個の新規スペクトル = 合計{len(combined_spectra_data)}個")
                            
                            st.download_button(
                                label="📥 統合スペクトルデータダウンロード",
                                data=combined_pickle_buffer,
                                file_name=f'combined_spectrum_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl',
                                mime='application/octet-stream',
                                key="download_combined_pickle"
                            )
                        else:
                            st.error("❌ 無効なpickleファイル形式です")
                    
                    except Exception as e:
                        st.error(f"❌ ファイルの処理中にエラーが発生しました: {str(e)}")
    
    # データベース比較機能をここに統合
    st.markdown("---")
    
    # 利用可能なスペクトルの数を確認
    total_spectra = len(st.session_state.uploaded_database_spectra)
    """
    # デバッグ情報（開発時のみ表示）
    if st.sidebar.checkbox("🐛 デバッグ情報", key="debug_mode"):
        st.info(f"🔍 デバッグ: 現在のスペクトル数 = {total_spectra}")
        if total_spectra > 0:
            st.write("登録済みスペクトル（最初の5個）:")
            for i, spec in enumerate(st.session_state.uploaded_database_spectra[:5]):
                st.write(f"  {i+1}. ID: {spec['id'][:50]}..., Name: {spec['filename']}")
            if total_spectra > 5:
                st.write(f"  ... 他 {total_spectra - 5} 個")
    """
    if total_spectra >= 2:
        st.header("🎯 基準スペクトル選択・比較実行")
        
        # 基準スペクトル選択
        spectrum_names = [spec['filename'] for spec in st.session_state.uploaded_database_spectra]
        spectrum_ids = [spec['id'] for spec in st.session_state.uploaded_database_spectra]
        
        selected_index = st.selectbox(
            "比較の基準とするスペクトルを選択してください:",
            range(len(spectrum_names)),
            format_func=lambda x: spectrum_names[x],
            key="reference_spectrum_select"
        )
        
        if selected_index is not None:
            reference_spectrum_id = spectrum_ids[selected_index]
            reference_spectrum_name = spectrum_names[selected_index]
            
            st.info(f"📌 基準スペクトル: **{reference_spectrum_name}**")
            
            # 類似度計算手法の説明
            # st.markdown("""
            # **類似度計算手法:**
            # - **コサイン類似度**: ベクトル角度に基づく類似度（形状の類似性を重視）
            # - **ピアソン相関**: 線形関係の強さ（相関の強さを重視）  
            # - **相互相関**: 時系列的な類似性（位置ずれに対応）
            # """)
            
            # 比較計算パラメータ
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                similarity_method = "cosine"
            """
                similarity_method = st.selectbox(
                    "類似度計算手法", 
                    ['cosine', 'pearson', 'cross_correlation'], 
                    index=0,
                    format_func=lambda x: {
                        'cosine': 'コサイン類似度',
                        'pearson': 'ピアソン相関',
                        'cross_correlation': '相互相関'
                    }[x],
                    key="similarity_method"
                )
            """
            with col2:
                pool_size = st.selectbox("プーリングサイズ", [2, 4, 8], index=1, key="db_pool_size")
            with col3:
                comparison_threshold = st.slider("比較閾値", 0.5, 0.95, 0.7, step=0.05, key="db_threshold")
            with col4:
                max_spectra = len(st.session_state.uploaded_database_spectra)
                top_n = st.slider("解析対象スペクトル数", 2, min(max_spectra, 20), min(10, max_spectra), key="db_top_n")
            
            with st.expander("⚙️ 詳細設定", expanded=False):
                st.markdown("""
                - **プーリングサイズ**: スペクトルのダウンサンプリング設定（現在は未使用）
                - **比較閾値**: 詳細計算の実行閾値（現在は未使用）
                - **解析対象スペクトル数**: 表示する上位結果数（現在は未使用）
                
                **注意**: 異なる波数範囲・間隔のスペクトルは自動的に共通グリッドに補間されます。
                """)
            
            if st.button("比較計算を実行", type="primary", key="calculate_comparison_btn"):
                with st.spinner("基準スペクトルとの比較を計算中..."):
                    # 全スペクトルを読み込み
                    all_spectra = st.session_state.database_analyzer.load_all_spectra()
                    
                    # 基準スペクトルを取得
                    reference_spectrum_data = all_spectra[reference_spectrum_id]
                    
                    # 他のスペクトルとの類似度を計算
                    comparison_results = []
                    
                    progress_bar = st.progress(0)
                    
                    for i, spectrum_id in enumerate(spectrum_ids):
                        if spectrum_id == reference_spectrum_id:
                            # 基準スペクトル自身の場合
                            comparison_results.append({
                                'spectrum_id': spectrum_id,
                                'spectrum_name': spectrum_names[i],
                                'match_score': 1.0
                            })
                        else:
                            # 他のスペクトルとの比較
                            spectrum_data = all_spectra[spectrum_id]
                            match_score = st.session_state.database_analyzer.calculate_spectrum_similarity(
                                reference_spectrum_data, spectrum_data, method=similarity_method
                            )
                            comparison_results.append({
                                'spectrum_id': spectrum_id,
                                'spectrum_name': spectrum_names[i],
                                'match_score': match_score
                            })
                        
                        progress_bar.progress((i + 1) / len(spectrum_ids))
                    
                    # 結果をスコア順にソート
                    comparison_results.sort(key=lambda x: x['match_score'], reverse=True)
                    
                    # 結果をセッション状態に保存
                    st.session_state.comparison_results = {
                        'reference_spectrum_id': reference_spectrum_id,
                        'reference_spectrum_name': reference_spectrum_name,
                        'results': comparison_results,
                        'spectra_data': all_spectra
                    }
                    
                    st.success("データベース比較が完了しました！")
    
    elif total_spectra == 1:
        st.info("💡 比較には少なくとも2つのスペクトルが必要です。もう1つスペクトルを追加してください。")
    elif total_spectra == 0:
        st.info("💡 スペクトルファイルをアップロードするか、保存済みデータを読み込んでください。")
    
    # アップロード済みスペクトルの表示
    if st.session_state.uploaded_database_spectra:
        st.markdown("---")
        with st.expander("アップロード済みスペクトル", expanded=False):
            # アップロードされたファイルのリスト表示
            spectra_df = pd.DataFrame(st.session_state.uploaded_database_spectra)
            spectra_df.columns = ['ID', 'ファイル名']
            st.dataframe(spectra_df, use_container_width=True)

def create_interpolated_csv(all_data, spectrum_type):
    """
    異なる波数データを持つスペクトラムを統一された波数グリッドで補間してCSVを作成
    
    Parameters:
    all_data: 全ファイルのデータリスト
    spectrum_type: 'raw_spectrum', 'baseline_removed'のいずれか
    
    Returns:
    str: CSV形式の文字列
    """
    if not all_data:
        return ""
    
    # 全ファイルの波数範囲を取得
    min_wavenum = min(data['wavenum'].min() for data in all_data)
    max_wavenum = max(data['wavenum'].max() for data in all_data)
    
    # 最も細かい波数間隔を取得（最大データ点数に基づく）
    max_points = max(len(data['wavenum']) for data in all_data)
    
    # 統一された波数グリッドを作成
    common_wavenum = np.linspace(min_wavenum, max_wavenum, max_points)
    
    # DataFrameを作成
    export_df = pd.DataFrame({'WaveNumber': common_wavenum})
    
    # 各ファイルのスペクトラムを共通の波数グリッドに補間
    for data in all_data:
        interpolated_spectrum = np.interp(common_wavenum, data['wavenum'], data[spectrum_type])
        export_df[data['file_name']] = interpolated_spectrum
    
    return export_df.to_csv(index=False, encoding='utf-8-sig')

def load_pickle_spectra_sidebar():
    """サイドバーでpickleファイルからスペクトルデータを読み込み"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 保存済みデータ読み込み")
    
    # リセットボタンを追加
    if st.sidebar.button("🗑️ 全データクリア", help="登録済みスペクトルを全て削除"):
        st.session_state.uploaded_database_spectra = []
        st.session_state.processed_pickle_files = set()
        st.session_state.loaded_spectrum_names = set()
        if hasattr(st.session_state, 'database_analyzer'):
            st.session_state.database_analyzer.metadata = {}
            st.session_state.database_analyzer.save_metadata()
        st.sidebar.success("✅ 全データをクリアしました")
        st.rerun()
    
    uploaded_pickle = st.sidebar.file_uploader(
        "pickleファイルを選択",
        type=['pkl'],
        help="以前に保存したスペクトルデータファイルを読み込みます",
        key="sidebar_pickle_uploader"
    )
    
    # 処理済みファイルとスペクトル名を追跡するためのセッション状態
    if 'processed_pickle_files' not in st.session_state:
        st.session_state.processed_pickle_files = set()
    if 'loaded_spectrum_names' not in st.session_state:
        st.session_state.loaded_spectrum_names = set()
    
    if uploaded_pickle is not None:
        # ファイル名と内容のハッシュを生成して重複処理を防ぐ
        file_content = uploaded_pickle.read()
        uploaded_pickle.seek(0)  # ファイルポインタをリセット
        file_identifier = f"{uploaded_pickle.name}_{len(file_content)}_{hash(file_content[:1000])}"
        
        # すでに処理済みのファイルかチェック
        if file_identifier in st.session_state.processed_pickle_files:
            st.sidebar.info(f"ℹ️ ファイル '{uploaded_pickle.name}' は既に読み込み済みです")
            return
        
        try:
            pickle_data = pickle.load(uploaded_pickle)
            
            if 'spectra_data' in pickle_data:
                spectra_data = pickle_data['spectra_data']
                processing_params = pickle_data.get('processing_params', {})
                
                # 重複しないスペクトルのみを追加
                added_count = 0
                skipped_count = 0
                
                for i, data in enumerate(spectra_data):
                    spectrum_name = data['file_name']
                    
                    # スペクトル名で重複チェック
                    if spectrum_name in st.session_state.loaded_spectrum_names:
                        skipped_count += 1
                        continue
                    
                    try:
                        # スペクトルデータを保存（データベース用）
                        spectrum_id = f"{spectrum_name}_loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
                        
                        spectrum_data_db = {
                            'wavenum': data['wavenum'],
                            'spectrum': data['baseline_removed'],  # ベースライン削除済み
                            'original_filename': spectrum_name,
                            'file_type': data.get('file_type', 'loaded'),
                            'processing_params': processing_params
                        }
                        
                        # アナライザーに保存
                        st.session_state.database_analyzer.metadata[spectrum_id] = {
                            'filename': f"{spectrum_id}.pkl",
                            'original_filename': spectrum_name,
                            'file_type': data.get('file_type', 'loaded'),
                            'wavenum_range': (data['wavenum'][0], data['wavenum'][-1]),
                            'data_points': len(data['wavenum']),
                            'saved_at': datetime.now().isoformat()
                        }
                        
                        # データをメモリに保存
                        spectrum_file = st.session_state.database_analyzer.storage_dir / f"{spectrum_id}.pkl"
                        spectrum_file.parent.mkdir(exist_ok=True)
                        with open(spectrum_file, 'wb') as f:
                            pickle.dump(spectrum_data_db, f)
                        
                        st.session_state.uploaded_database_spectra.append({
                            'id': spectrum_id,
                            'filename': spectrum_name
                        })
                        
                        # スペクトル名を読み込み済みとしてマーク
                        st.session_state.loaded_spectrum_names.add(spectrum_name)
                        
                        added_count += 1
                        
                    except Exception as e:
                        st.sidebar.error(f"エラー: {spectrum_name}")
                        skipped_count += 1
                
                st.session_state.database_analyzer.save_metadata()
                
                # ファイルを処理済みとしてマーク
                st.session_state.processed_pickle_files.add(file_identifier)
                
                # 結果を表示
                if added_count > 0:
                    st.sidebar.success(f"✅ {added_count}個の新規スペクトルを追加")
                if skipped_count > 0:
                    st.sidebar.info(f"ℹ️ {skipped_count}個のスペクトルは既存のためスキップ")
                
            else:
                st.sidebar.error("❌ 無効なpickleファイル形式です")
                
        except Exception as e:
            st.sidebar.error(f"❌ 読み込み失敗: {str(e)}")
    
    # 現在の状態を表示
    total_spectra = len(st.session_state.uploaded_database_spectra)
    unique_names = len(st.session_state.loaded_spectrum_names) if 'loaded_spectrum_names' in st.session_state else 0
    
    if total_spectra > 0:
        st.sidebar.markdown("---")
        st.sidebar.metric("📊 登録済みスペクトル数", total_spectra)
        st.sidebar.metric("🔤 ユニーク名数", unique_names)
        
        if total_spectra != unique_names:
            st.sidebar.warning(f"⚠️ 重複の可能性があります")
            if st.sidebar.button("🔧 重複を除去", help="同じ名前のスペクトルを1つに統合"):
                # 重複除去処理
                unique_spectra = {}
                for spec in st.session_state.uploaded_database_spectra:
                    if spec['filename'] not in unique_spectra:
                        unique_spectra[spec['filename']] = spec
                
                st.session_state.uploaded_database_spectra = list(unique_spectra.values())
                st.sidebar.success(f"✅ {total_spectra - len(unique_spectra)}個の重複を除去")

def display_uploaded_database_spectra():
    """アップロードされたスペクトルを表示（この関数は現在使用されていません）"""
    pass

def run_database_comparison():
    """データベース比較を実行（この関数は現在使用されていません）"""
    pass

def display_comparison_results():
    """比較結果を表示"""
    if st.session_state.comparison_results is None:
        return
    
    results = st.session_state.comparison_results
    reference_name = results['reference_spectrum_name']
    comparison_results = results['results']
    spectra_data = results['spectra_data']
    
    st.header("📊 比較結果")
    
    # 基準スペクトル情報
    st.info(f"🎯 **基準スペクトル**: {reference_name}")
    
    # 統計サマリー（基準スペクトル除く）
    scores = [r['match_score'] for r in comparison_results if r['match_score'] < 1.0]
    
    if scores:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均類似度", f"{np.mean(scores):.3f}")
        with col2:
            st.metric("最大類似度", f"{np.max(scores):.3f}")
        with col3:
            st.metric("最小類似度", f"{np.min(scores):.3f}")
        with col4:
            st.metric("標準偏差", f"{np.std(scores):.3f}")
    
    # 比較結果テーブル
    st.subheader("🏆 類似度スコアランキング")
    
    # DataFrameを作成
    results_df = pd.DataFrame([
        {
            'ランク': i + 1,
            'スペクトル名': r['spectrum_name'],
            '類似度スコア': f"{r['match_score']:.4f}",
            '基準スペクトル': '⭐' if r['match_score'] == 1.0 else ''
        }
        for i, r in enumerate(comparison_results)
    ])
    
    st.dataframe(results_df, use_container_width=True, hide_index=True)
    
    # 最高一致スペクトルの表示（基準スペクトル除く）
    best_match = next((r for r in comparison_results if r['match_score'] < 1.0), None)
    
    if best_match:
        st.header("⭐ 最高類似スペクトル")
        
        best_spectrum_name = best_match['spectrum_name']
        best_score = best_match['match_score']
        
        st.success(f"**最高類似度スコア: {best_score:.4f}**")
        st.info(f"**スペクトル: {best_spectrum_name}**")
        
        # スペクトル比較表示
        col1, col2 = st.columns(2)
        
        # 基準スペクトルデータ
        reference_spectrum_id = results['reference_spectrum_id']
        reference_data = spectra_data[reference_spectrum_id]
        
        # 最高一致スペクトルデータ
        best_spectrum_id = best_match['spectrum_id']
        best_data = spectra_data[best_spectrum_id]
        
        with col1:
            st.subheader(f"基準: {reference_name}")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=reference_data['wavenum'],
                y=reference_data['spectrum'],
                mode='lines',
                name=reference_name,
                line=dict(color='blue', width=2)
            ))
            fig1.update_layout(
                xaxis_title="波数 (cm⁻¹)",
                yaxis_title="強度",
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader(f"最高一致: {best_spectrum_name}")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=best_data['wavenum'],
                y=best_data['spectrum'],
                mode='lines',
                name=best_spectrum_name,
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
            x=reference_data['wavenum'],
            y=reference_data['spectrum'],
            mode='lines',
            name=f"基準: {reference_name}",
            line=dict(color='blue', width=2)
        ))
        
        fig_overlay.add_trace(go.Scatter(
            x=best_data['wavenum'],
            y=best_data['spectrum'],
            mode='lines',
            name=f"最高一致: {best_spectrum_name}",
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig_overlay.update_layout(
            title=f"重ね合わせ: {reference_name} vs {best_spectrum_name} (類似度スコア: {best_score:.4f})",
            xaxis_title="波数 (cm⁻¹)",
            yaxis_title="強度",
            height=500,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        
        st.plotly_chart(fig_overlay, use_container_width=True)
    
    # 高類似スペクトルリスト
    with st.expander("📋 高類似スペクトル (> 0.8)", expanded=False):
        high_match_results = [r for r in comparison_results if r['match_score'] > 0.8 and r['match_score'] < 1.0]
        
        if high_match_results:
            high_match_df = pd.DataFrame([
                {
                    'スペクトル名': r['spectrum_name'],
                    '類似度スコア': f"{r['match_score']:.4f}"
                }
                for r in high_match_results
            ])
            st.dataframe(high_match_df, use_container_width=True, hide_index=True)
        else:
            st.info("類似度スコア > 0.8 のスペクトルが見つかりませんでした。")

def export_comparison_results():
    """比較結果のエクスポート"""
    if st.session_state.comparison_results is None:
        return
    
    st.header("💾 結果をエクスポート")
    
    results = st.session_state.comparison_results
    comparison_results = results['results']
    reference_name = results['reference_spectrum_name']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("比較結果をダウンロード (CSV)", key="download_comparison_csv"):
            # 比較結果をDataFrameに変換
            results_df = pd.DataFrame([
                {
                    'ランク': i + 1,
                    'スペクトル名': r['spectrum_name'],
                    '類似度スコア': r['match_score'],
                    '基準スペクトル': reference_name if r['match_score'] == 1.0 else ''
                }
                for i, r in enumerate(comparison_results)
            ])
            
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📥 CSVダウンロード",
                data=csv_buffer.getvalue(),
                file_name=f"比較結果_{reference_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="download_comp_results"
            )
    
    with col2:
        if st.button("スペクトル情報をダウンロード", key="download_spectra_info"):
            spectra_info = []
            spectra_data = results['spectra_data']
            
            for r in comparison_results:
                spectrum_data = spectra_data[r['spectrum_id']]
                spectra_info.append({
                    'スペクトル名': r['spectrum_name'],
                    '類似度スコア': r['match_score'],
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
    
    # サイドバーでpickleファイル読み込み機能
    load_pickle_spectra_sidebar()
    
    st.header("ラマンスペクトルデータベース")
    st.markdown("---")
    
    # タブを作成（2つに変更）
    tab1, tab2 = st.tabs(["📁 アップロード・処理・比較", "📊 結果・エクスポート"])
    
    with tab1:
        upload_and_process_database_files()
    
    with tab2:
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
