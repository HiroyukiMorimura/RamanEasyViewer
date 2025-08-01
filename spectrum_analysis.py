# -*- coding: utf-8 -*-
"""
スペクトル解析モジュール
ラマンスペクトルの基本解析機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from common_utils import *

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def spectrum_analysis_mode():
    """
    Spectrum analysis mode (original functionality)
    """
    st.header("ラマンスペクトル表示")
    
    # パラメータ設定
    # savgol_wsize = 21  # Savitzky-Golayフィルタのウィンドウサイズ
    pre_start_wavenum = 400  # 波数の開始
    pre_end_wavenum = 2000  # 波数の終了
    Fsize = 14  # フォントサイズ
    
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

    # 複数ファイルのアップロード
    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（複数可）",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        key="mv_uploader"
    )
    
    # 各ファイルのデータを格納するリスト（波数データも含む）
    all_data = []  # 各要素は辞書形式 {'wavenum': array, 'raw_spectrum': array, 'baseline_removed': array, 'moving_avg': array, 'file_name': str}
    
    if uploaded_files:
        # すべてのファイルに対して処理
        for uploaded_file in uploaded_files:
            try:
                result = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                st.write(file_type)
                if wavenum is None:
                    st.error(f"{file_name}の処理中にエラーが発生しました")
                    continue
                
                st.write(f"ファイルタイプ: {file_type} - {file_name}")
                
                # 各ファイルのデータを辞書として格納
                file_data = {
                    'wavenum': wavenum,
                    'raw_spectrum': spectra,
                    'baseline_removed': BSremoval_specta_pos,
                    'moving_avg': Averemoval_specta_pos,
                    'file_name': file_name
                }
                all_data.append(file_data)
                
            except Exception as e:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {e}")
    
        # すべてのファイルが処理された後に重ねてプロット
        if all_data:
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
            
            # ベースライン補正後+スパイク修正後のスペクトルを重ねてプロット
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
            
            # ベースライン補正後+スパイク修正後+移動平均のスペクトルを重ねてプロット
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
            
            # Baseline removed + Moving AverageのCSVダウンロード
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
    
    Parameters:
    all_data: 全ファイルのデータリスト
    spectrum_type: 'raw_spectrum', 'baseline_removed', 'moving_avg'のいずれか
    
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

def display_raman_correlation_table():
    """
    ラマン分光の帰属表を表示
    """
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
