# -*- coding: utf-8 -*-
"""
多変量解析モジュール
多変量解析機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
import os
import glob
import traceback
from common_utils import *

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def plot_spectra_matplotlib(grouped_data, title="Baseline Corrected Spectra"):
    """
    Plot spectra using matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (group, spectra_list) in enumerate(grouped_data.items()):
        color = colors[i % len(colors)]
        
        for j, spectrum in enumerate(spectra_list):
            label = group if j == 0 else None
            ax.plot(spectrum[:, 0], spectrum[:, 2], color=color, alpha=0.7, label=label)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def perform_nmf_analysis(all_spectra, n_components):
    """
    Perform NMF analysis
    """
    if isinstance(all_spectra, list):
        all_spectra = np.vstack(all_spectra)
        
    # Ensure non-negative values for NMF
    all_spectra[all_spectra < 0] = 0
    
    # Apply NMF
    nmf_model = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=1000)
    W = nmf_model.fit_transform(all_spectra)  # Concentration profiles
    H = nmf_model.components_  # Spectral profiles
    
    return W, H, nmf_model

def plot_nmf_components(H, wavenumber):
    """
    Plot NMF components
    """
    fig, axes = plt.subplots(len(H), 1, figsize=(12, 3 * len(H)))
    if len(H) == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (ax, spectrum) in enumerate(zip(axes, H)):
        ax.plot(wavenumber, spectrum, color=colors[i % len(colors)], linewidth=2)
        ax.set_title(f"Component {i+1}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Wavenumber (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_nmf_scores(W, all_labels, group_names):
    """
    Plot NMF concentration scores
    """
    if W.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, group in enumerate(group_names):
            group_indices = [j for j, label in enumerate(all_labels) if label == group]
            
            ax.scatter(W[group_indices, 0], W[group_indices, 1], 
                      label=group, color=colors[i % len(colors)], 
                      s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        ax.set_title("Score Plot (Component 1 vs Component 2)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    else:
        st.warning("Need at least 2 components for score plot")
        return None

def plot_contribution_ratios(contribution_ratios):
    """
    Plot component contribution ratios as bar chart
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = [f"Component {i+1}" for i in range(len(contribution_ratios))]
    percentages = contribution_ratios * 100
    
    bars = ax.bar(components, percentages, color=plt.cm.Set3(np.linspace(0, 1, len(components))))
    
    # Add value labels on bars
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Component Contribution Ratios", fontsize=14, fontweight='bold')
    ax.set_xlabel("Components", fontsize=12)
    ax.set_ylabel("Contribution (%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    return fig

def extract_group_name(file_name: str) -> str:
    """
    ファイル名から「GroupName」を取り出す。
    フォーマットが GroupName_XXX.csv ならアンダースコア区切りの2番目の要素を返し、
    そうでなければ拡張子を除いたファイル名全体を返す。
    """
    parts = file_name.split('_')
    if len(parts) >= 2:
        return parts[-2]
    return file_name.rsplit('.', 1)[0]

def multivariate_analysis_mode():
    """
    Multivariate analysis mode
    """
    st.header("多変量解析")
    
    # Parameters in sidebar
    start_wavenum = st.sidebar.number_input("波数（開始）を入力してください:", value=400, min_value=0, key="mv_start")
    end_wavenum = st.sidebar.number_input("波数（終了）を入力してください:", value=2000, min_value=start_wavenum + 1, key="mv_end")
    dssn_th = st.sidebar.number_input("ベースラインパラメーターを入力してください:", 1, 10000, value=10000, step=1) / 1e7
    savgol_wsize = st.sidebar.number_input("移動平均のウィンドウサイズを入力してください:", min_value=1, max_value=35, value=5, step=2, key='unique_savgol_wsize_key')
    
    st.sidebar.subheader("多変量解析")
    n_components = st.sidebar.selectbox("コンポーネント数", options=[2, 3, 4, 5], index=0, key="mv_components")
    
    # File upload
    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（複数可）",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple CSV files with spectral data. Files should be named as GroupName_Number.csv",
        key="mv_uploader"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Initialize data containers
        grouped_data = {}
        baseline_list = []
        label_list = []
        wavenum = None
        
        # Process each file
        for uploaded_file in uploaded_files:
            try:
                # Process spectrum file
                result = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                
                # Extract group name
                group = extract_group_name(file_name)
                
                # Store processed data
                processed = np.column_stack([wavenum, spectra, Averemoval_specta_pos])
                grouped_data.setdefault(group, []).append(processed)
                
                # Store baseline removed spectra and labels
                baseline_list.append(Averemoval_specta_pos)
                label_list.append(group)

            except Exception as e:
                st.error(f"{uploaded_file.name} の処理でエラー: {e}")
                st.error(traceback.format_exc())
        
        if grouped_data and baseline_list:
            # Store in session state
            st.session_state.mv_all_labels = label_list
            st.session_state.mv_grouped_data = grouped_data
            st.session_state.mv_all_baseline_spectra = baseline_list
            st.session_state.mv_wavenumber = wavenum
            st.success("データプロセス完了!")
            
            # Display spectrum plot
            st.header("スペクトルプロット")
            fig_spectra = plot_spectra_matplotlib(grouped_data)
            st.pyplot(fig_spectra)
            
            # Display data summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Spectra", len(baseline_list))
            with col2:
                st.metric("Number of Groups", len(grouped_data))
            with col3:
                st.metric("Data Points per Spectrum", len(wavenum))
            
            # Perform NMF analysis
            st.header("多変量解析")
            
            # Check data consistency
            if len(baseline_list) != len(label_list):
                st.error(f"データの不整合: スペクトル数 {len(baseline_list)} != ラベル数 {len(label_list)}")
                return
            
            # Perform NMF
            W, H, nmf_model = perform_nmf_analysis(baseline_list, n_components)
            
            # Store NMF results
            st.session_state.mv_W = W
            st.session_state.mv_H = H
            st.session_state.mv_nmf_model = nmf_model
            
            # Calculate reconstruction error
            reconstruction_error = nmf_model.reconstruction_err_
            st.session_state.mv_reconstruction_error = reconstruction_error
            
            st.success(f"解析完了! Reconstruction error: {reconstruction_error:.4f}")
            
            # Display NMF results
            st.info(f"**Reconstruction Error:** {reconstruction_error:.4f}")
            
            # Plot NMF components
            st.subheader("コンポーネントスペクトル")
            fig_components = plot_nmf_components(H, wavenum)
            st.pyplot(fig_components)
            
            # Plot NMF scores
            st.subheader("スコアプロット")
            group_names = list(grouped_data.keys())
            fig_scores = plot_nmf_scores(W, label_list, group_names)
            if fig_scores:
                st.pyplot(fig_scores)
            
            # Display contribution ratios
            st.subheader("Component Contribution Ratios")
            contribution_ratios = np.sum(W, axis=0)
            contribution_ratios = contribution_ratios / np.sum(contribution_ratios)
            
            contrib_df = pd.DataFrame({
                'Component': [f"Component {i+1}" for i in range(len(contribution_ratios))],
                'Contribution Ratio': contribution_ratios,
                'Percentage': contribution_ratios * 100
            })
            
            st.dataframe(contrib_df, use_container_width=True)
            
            # Bar chart of contributions
            fig_contrib = plot_contribution_ratios(contribution_ratios)
            st.pyplot(fig_contrib)
            
            # Download results
            st.subheader("📥 データダウンロード")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download concentration matrix
                W_df = pd.DataFrame(W, 
                                  columns=[f"Component_{i+1}" for i in range(W.shape[1])])
                W_df['Sample'] = label_list
                csv_W = W_df.to_csv(index=False)
                st.download_button(
                    label="Download Concentration Matrix",
                    data=csv_W,
                    file_name="nmf_concentration_matrix.csv",
                    mime="text/csv",
                    key="mv_download_W"
                )
            
            with col2:
                # Download spectral components
                H_df = pd.DataFrame(H.T,
                                    columns=[f"Component_{i+1}" for i in range(H.shape[0])])
                H_df['Wavenumber'] = wavenum
            
                # reorder so that Wavenumber comes first
                cols = ['Wavenumber'] + [c for c in H_df.columns if c.startswith('Component_')]
                H_df = H_df[cols]
            
                csv_H = H_df.to_csv(index=False)
                st.download_button(
                    label="Download Spectral Components",
                    data=csv_H,
                    file_name="nmf_spectral_components.csv",
                    mime="text/csv",
                    key="mv_download_H"
                )
            
        else:
            st.error("有効なデータが見つかりませんでした。")
    
    else:
        st.info("CSVファイルをアップロードしてください。")
