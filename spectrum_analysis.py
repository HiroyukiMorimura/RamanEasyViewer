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
        "ファイルを選択してください",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple CSV files with spectral data. Files should be named as GroupName_Number.csv",
        key="mv_uploader"
    )
    
    all_spectra = []  # すべてのスペクトルを格納するリスト
    all_bsremoval_spectra = []  # ベースライン補正後のスペクトルを格納するリスト
    all_averemoval_spectra = []  # ベースライン補正後移動平均を行ったスペクトルを格納するリスト
    file_labels = []  # 各ファイル名のリスト
    
    if uploaded_files:
        # すべてのファイルに対して処理
        for uploaded_file in uploaded_files:
            try:
                result = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                
                if wavenum is None:
                    st.error(f"{file_name}の処理中にエラーが発生しました")
                    continue
                
                st.write(f"ファイルタイプ: {file_type} - {file_name}")
                
                # 各スペクトルを格納
                file_labels.append(file_name)
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)
                all_averemoval_spectra.append(Averemoval_specta_pos)
                
            except Exception as e:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {e}")
    
        # すべてのファイルが処理された後に重ねてプロット
        if all_spectra:
            selected_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'black']
            
            # 元のスペクトルを重ねてプロット
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, spectrum in enumerate(all_spectra):
                ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i % len(selected_colors)], label=f"{file_labels[i]}")
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Raw Spectra', fontsize=Fsize)
            ax.legend(title="Spectra")
            st.pyplot(fig)
            
            # Raw spectraのCSVダウンロード
            export_df = pd.DataFrame({'WaveNumber': wavenum})
            for i, spectrum in enumerate(all_spectra):
                export_df[file_labels[i]] = spectrum
            
            csv_data = export_df.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="Download Raw Spectra as CSV",
                data=csv_data,
                file_name='raw_spectra.csv',
                mime='text/csv'
            )
            
            # ベースライン補正後+スパイク修正後のスペクトルを重ねてプロット
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, spectrum in enumerate(all_bsremoval_spectra):
                ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i % len(selected_colors)], label=f"{file_labels[i]}")
            
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Baseline Removed', fontsize=Fsize)
            st.pyplot(fig)
            
            # Baseline removedのCSVダウンロード
            export_df_bs = pd.DataFrame({'WaveNumber': wavenum})
            for i, spectrum in enumerate(all_bsremoval_spectra):
                export_df_bs[file_labels[i]] = spectrum
            
            csv_data_bs = export_df_bs.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="Download Baseline Removed Spectra as CSV",
                data=csv_data_bs,
                file_name='baseline_removed_spectra.csv',
                mime='text/csv'
            )
            
            # ベースライン補正後+スパイク修正後+移動平均のスペクトルを重ねてプロット
            fig, ax = plt.subplots(figsize=(10, 5))
            for i, spectrum in enumerate(all_averemoval_spectra):
                ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i % len(selected_colors)], label=f"{file_labels[i]}")
            
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Baseline Removed + Moving Average', fontsize=Fsize)
            st.pyplot(fig)
            
            # Baseline removed + Moving AverageのCSVダウンロード
            export_df_avg = pd.DataFrame({'WaveNumber': wavenum})
            for i, spectrum in enumerate(all_averemoval_spectra):
                export_df_avg[file_labels[i]] = spectrum
            
            csv_data_avg = export_df_avg.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="Download Baseline Removed + Moving Average Spectra as CSV",
                data=csv_data_avg,
                file_name='baseline_removed_moving_avg_spectra.csv',
                mime='text/csv'
            )
            
            # ラマン相関表
            display_raman_correlation_table()

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