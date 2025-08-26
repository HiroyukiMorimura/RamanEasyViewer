# -*- coding: utf-8 -*-
"""
スペクトル解析モジュール
ラマンスペクトルの基本解析機能 + フォルダ監視機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import threading
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from common_utils import *

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

class SpectrumFileHandler(FileSystemEventHandler):
    """ファイル変更を監視するハンドラー"""
    
    def __init__(self, callback):
        self.callback = callback
        self.valid_extensions = {'.csv', '.txt'}
    
    def on_created(self, event):
        """新しいファイルが作成された時"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.valid_extensions:
                self.callback(file_path, 'created')
    
    def on_modified(self, event):
        """ファイルが変更された時"""
        if not event.is_directory:
            file_path = Path(event.src_path)
            if file_path.suffix.lower() in self.valid_extensions:
                self.callback(file_path, 'modified')

def setup_folder_monitoring(folder_path, callback):
    """フォルダ監視を設定"""
    if not os.path.exists(folder_path):
        return None, None
    
    event_handler = SpectrumFileHandler(callback)
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    return observer, event_handler

def file_change_callback(file_path, event_type):
    """ファイル変更時のコールバック関数"""
    if 'file_changes' not in st.session_state:
        st.session_state.file_changes = []
    
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    change_info = {
        'file_path': str(file_path),
        'event_type': event_type,
        'timestamp': timestamp
    }
    
    st.session_state.file_changes.append(change_info)
    st.session_state.auto_update_trigger = time.time()

def load_folder_files(folder_path, file_extensions=['.csv', '.txt']):
    """フォルダ内の対象ファイルを取得"""
    if not os.path.exists(folder_path):
        return []
    
    files = []
    for ext in file_extensions:
        files.extend(Path(folder_path).glob(f'*{ext}'))
    
    return sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)

def process_spectrum_file_from_path(file_path, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
    """ファイルパスから直接スペクトラムを処理する関数"""
    try:
        file_path = Path(file_path)
        file_name = file_path.stem
        file_extension = file_path.suffix.lower()
        
        # ファイルの読み込み
        if file_extension in ['.csv', '.txt']:
            try:
                df = pd.read_csv(file_path)
                file_type = f"CSV file ({file_extension})"
            except:
                try:
                    df = pd.read_csv(file_path, sep='\t')
                    file_type = f"Tab-separated file ({file_extension})"
                except:
                    df = pd.read_csv(file_path, sep='\s+')
                    file_type = f"Space-separated file ({file_extension})"
        else:
            return None, None, None, None, None, None
        
        # データの列数チェック
        if df.shape[1] < 2:
            return None, None, None, None, None, None
        
        # 波数とスペクトルデータの抽出
        wavenum = df.iloc[:, 0].values
        spectra = df.iloc[:, 1].values
        
        # 波数範囲でのフィルタリング
        mask = (wavenum >= start_wavenum) & (wavenum <= end_wavenum)
        wavenum = wavenum[mask]
        spectra = spectra[mask]
        
        # 基本的なベースライン補正（簡易版）
        BSremoval_specta_pos = spectra - np.min(spectra)
        
        # 移動平均（簡易版）
        if savgol_wsize > 1:
            kernel = np.ones(savgol_wsize) / savgol_wsize
            Averemoval_specta_pos = np.convolve(BSremoval_specta_pos, kernel, mode='same')
        else:
            Averemoval_specta_pos = BSremoval_specta_pos
        
        return wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name
        
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None, None, None, None, None

def process_selected_folder_files(folder_path, selected_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
    """選択されたフォルダ内ファイルを処理"""
    all_data = []
    
    for file_name in selected_files:
        file_path = Path(folder_path) / file_name
        try:
            result = process_spectrum_file_from_path(
                str(file_path), start_wavenum, end_wavenum, dssn_th, savgol_wsize
            )
            
            if result and result[0] is not None:
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name_processed = result
                
                file_data = {
                    'wavenum': wavenum,
                    'raw_spectrum': spectra,
                    'baseline_removed': BSremoval_specta_pos,
                    'moving_avg': Averemoval_specta_pos,
                    'file_name': file_name_processed,
                    'file_path': str(file_path)
                }
                all_data.append(file_data)
                
        except Exception as e:
            st.warning(f"ファイル {file_name} の処理中にエラーが発生しました: {e}")
    
    return all_data

def spectrum_analysis_mode():
    """
    Spectrum analysis mode with folder monitoring
    """
    st.header("ラマンスペクトル表示")
    
    # セッションステートの初期化
    if 'observer' not in st.session_state:
        st.session_state.observer = None
    if 'file_changes' not in st.session_state:
        st.session_state.file_changes = []
    if 'auto_update_trigger' not in st.session_state:
        st.session_state.auto_update_trigger = 0
    
    # パラメータ設定
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    Fsize = 14
    
    # 波数範囲の設定
    start_wavenum = st.sidebar.number_input(
        "波数（開始）を入力してください:", 
        min_value=-200, 
        max_value=4800, 
        value=pre_start_wavenum, 
        step=100
    )
    end_wavenum = st.sidebar.number_input(
        "波数（終了）を入力してください:", 
        min_value=-200, 
        max_value=4800, 
        value=pre_end_wavenum, 
        step=100
    )

    dssn_th = st.sidebar.number_input(
        "ベースラインパラメーターを入力してください:", 
        min_value=1, 
        max_value=10000, 
        value=1000, 
        step=1
    )
    dssn_th = dssn_th / 10000000
    
    # 平滑化パラメーター
    savgol_wsize = st.sidebar.number_input(
        "移動平均のウィンドウサイズを入力してください:",
        min_value=1,
        max_value=35,
        value=5,
        step=2,
        key='unique_savgol_wsize_key'
    )
    
    # フォルダパスの指定
    folder_path = st.sidebar.text_input(
        "フォルダパスを入力してください:",
        value="C:\\",
        key="folder_path"
    )
    
    # 自動で監視開始
    if os.path.exists(folder_path):
        if st.session_state.observer is None:
            observer, handler = setup_folder_monitoring(folder_path, file_change_callback)
            if observer:
                st.session_state.observer = observer
    
    # フォルダ内ファイルの一覧取得
    if os.path.exists(folder_path):
        available_files = load_folder_files(folder_path)
        file_names = [f.name for f in available_files]
        
        if file_names:
            # ファイル選択
            selected_files = st.sidebar.multiselect(
                "処理するファイルを選択してください:",
                file_names,
                default=file_names,  # デフォルトで全選択
                key="selected_files"
            )
            
            if selected_files:
                all_data = process_selected_folder_files(
                    folder_path, selected_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                
                if all_data:
                    display_spectra(all_data, Fsize)
                else:
                    st.info("選択されたファイルの処理に失敗しました")
            else:
                st.info("ファイルを選択してください")
        else:
            st.info("フォルダ内に処理可能なファイルがありません")
    else:
        st.warning("指定されたフォルダが存在しません")

def display_spectra(all_data, Fsize):
    """スペクトラを表示する共通関数"""
    if not all_data:
        return
    
    selected_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'black']
    
    # 元のスペクトルを重ねてプロット
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, data in enumerate(all_data):
        ax.plot(data['wavenum'], data['raw_spectrum'], 
               linestyle='-', 
               color=selected_colors[i % len(selected_colors)], 
               label=f"{data['file_name']}")
    ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
    ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
    ax.set_title('Raw Spectra', fontsize=Fsize)
    ax.legend(title="Spectra")
    st.pyplot(fig)
    
    # Raw spectraのCSVダウンロード
    raw_csv_data = create_interpolated_csv(all_data, 'raw_spectrum')
    st.download_button(
        label="Download Raw Spectra as CSV",
        data=raw_csv_data,
        file_name='raw_spectra.csv',
        mime='text/csv'
    )
    
    # ベースライン補正後のスペクトルを重ねてプロット
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, data in enumerate(all_data):
        ax.plot(data['wavenum'], data['baseline_removed'], 
               linestyle='-', 
               color=selected_colors[i % len(selected_colors)], 
               label=f"{data['file_name']}")
    
    ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
    ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
    ax.set_title('Baseline Removed', fontsize=Fsize)
    ax.legend(title="Spectra")
    st.pyplot(fig)
    
    # Baseline removedのCSVダウンロード
    baseline_csv_data = create_interpolated_csv(all_data, 'baseline_removed')
    st.download_button(
        label="Download Baseline Removed Spectra as CSV",
        data=baseline_csv_data,
        file_name='baseline_removed_spectra.csv',
        mime='text/csv'
    )
    
    # ベースライン補正後+移動平均のスペクトルを重ねてプロット
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, data in enumerate(all_data):
        ax.plot(data['wavenum'], data['moving_avg'], 
               linestyle='-', 
               color=selected_colors[i % len(selected_colors)], 
               label=f"{data['file_name']}")
    
    ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
    ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
    ax.set_title('Baseline Removed + Moving Average', fontsize=Fsize)
    ax.legend(title="Spectra")
    st.pyplot(fig)
    
    # Moving AverageのCSVダウンロード
    moving_avg_csv_data = create_interpolated_csv(all_data, 'moving_avg')
    st.download_button(
        label="Download Baseline Removed + Moving Average Spectra as CSV",
        data=moving_avg_csv_data,
        file_name='baseline_removed_moving_avg_spectra.csv',
        mime='text/csv'
    )
    
    # ラマン相関表
    display_raman_correlation_table()

def create_interpolated_csv(all_data, spectrum_type):
    """
    異なる波数データを持つスペクトラムを統一された波数グリッドで補間してCSVを作成
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

def display_raman_correlation_table():
    """ラマン分光の帰属表を表示"""
    raman_data = {
        "ラマンシフト (cm⁻¹)": [
            "100–200", "150–450", "250–400", "290–330", "430–550", "450–550", "480–660", "500–700",
            "550–800", "630–790", "800–970", "1000–1250", "1300–1400", "1500–1600", "1600–1800", "2100–2250",
            "2800–3100", "3300–3500"
        ],
        "振動モード / 化学基": [
            "格子振動 (Lattice vibrations)", "金属-酸素結合 (Metal-O)", "C-C アリファティック鎖", "Se-Se", "S-S",
            "Si-O-Si", "C-I", "C-Br", "C-Cl", "C-S", "C-O-C", "C=S", "CH₂, CH₃ (変角振動)",
            "芳香族 C=C", "C=O (カルボニル基)", "C≡C, C≡N", "C-H (sp³, sp²)", "N-H, O-H"
        ],
        "強度": [
            "強い (Strong)", "中〜弱", "強い", "強い", "強い", "強い", "強い", "強い", "強い", "中〜強", "中〜弱", "強い",
            "中〜弱", "強い", "中程度", "中〜強", "強い", "中程度"
        ]
    }
    
    raman_df = pd.DataFrame(raman_data)
    
    # ラマン相関表を表示
    st.subheader("（参考）ラマン分光の帰属表")
    st.table(raman_df)
