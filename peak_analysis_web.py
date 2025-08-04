# -*- coding: utf-8 -*-
"""
ラマンピーク解析モジュール
ピーク検出、手動調整、グリッドサーチ最適化機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from common_utils import *

def optimize_thresholds_via_gridsearch(
    wavenum,
    spectrum,
    manual_add_peaks,
    manual_exclude_indices,
    current_prom_thres,
    current_deriv_thres,
    current_smooth,
    detected_original_peaks,
    resolution,
    smooth_range,
):
    """
    グリッドサーチによる閾値最適化
    """
    best_score = -np.inf
    best_prom_thres = current_prom_thres
    best_deriv_thres = current_deriv_thres
    best_smooth = current_smooth

    # prominence と deriv の範囲
    prom_range = np.logspace(np.log10(0.01), np.log10(10), num=50)
    prom_range = [round(p, 2) for p in prom_range]
    deriv_range = np.logspace(np.log10(0.01), np.log10(10), num=50)
    deriv_range = [round(q, 2) for q in deriv_range]

    # 最初に安全にリスト化
    if detected_original_peaks is None:
        orig_peaks = []
    else:
        orig_peaks = detected_original_peaks.tolist() if hasattr(detected_original_peaks, "tolist") else list(detected_original_peaks)
    
    # 三重ループでグリッドサーチ
    for smooth in smooth_range:
        sd = savgol_filter(spectrum, int(smooth), 2, deriv=2)
    
        for deriv_thres in deriv_range:
            peaks, _ = find_peaks(-sd, height=deriv_thres)
            prominences = peak_prominences(-sd, peaks)[0]
    
            for prom_thres in prom_range:
                mask = prominences > prom_thres
                final_peaks = set(peaks[mask])
    
                # スコア計算
                score = 0
    
                # 1. 元のピークを正しく残せたか（+2）/ 消えたか（-1）
                for idx in orig_peaks:
                    score += 5 if idx in final_peaks else -1
    
                # 2. 手動追加ピークを正しく拾えたか（+2）/ 見逃したか（-1）
                for x in manual_add_peaks:
                    idx = np.argmin(np.abs(wavenum - x))
                    score += 5 if idx in final_peaks else -1
    
                # 3. 手動除外ピークを正しく除外できたか（+2）/ 残ってしまったか（-1）
                for idx in manual_exclude_indices:
                    score += 5 if idx not in final_peaks else -1
    
                # 4. 余分なピークはペナルティ
                for idx in final_peaks:
                    if idx not in orig_peaks and all(abs(x - wavenum[idx]) > 0 for x in manual_add_peaks):
                        score -= 5

                # ベスト更新
                if score > best_score:
                    best_score = score
                    best_prom_thres = prom_thres
                    best_deriv_thres = deriv_thres
                    best_smooth = smooth
                
    return {
        "prominence_threshold": best_prom_thres,
        "second_deriv_threshold": best_deriv_thres,
        "second_deriv_smooth": best_smooth,
        "score": best_score
    }

def peak_analysis_mode():
    """
    Peak analysis mode - 修正版
    """
    st.header("ラマンピークファインダー")
    
    # 事前パラメータ
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    
    # temporary変数の処理（ウィジェット作成後にクリア）
    for param in ["second_deriv_smooth", "prominence_threshold", "second_deriv_threshold"]:
        temp_key = f"{param}_temp"
        if temp_key in st.session_state:
            st.session_state.pop(temp_key)  # temporary変数をクリア
    
    # グリッドサーチ結果の適用フラグをクリア
    if st.session_state.get("apply_grid_result", False):
        st.session_state.pop("apply_grid_result", None)
            
    # セッションステートの初期化
    for key, default in {
        "prominence_threshold": 100,
        "second_deriv_threshold": 100,
        "savgol_wsize": 5,
        "spectrum_type_select": "ベースライン削除",
        "second_deriv_smooth": 5,
        "manual_peak_keys": []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # UIパネル（Sidebar）
    start_wavenum = st.sidebar.number_input("波数（開始）を入力してください:", -200, 4800, value=pre_start_wavenum, step=100)
    end_wavenum = st.sidebar.number_input("波数（終了）を入力してください:", -200, 4800, value=pre_end_wavenum, step=100)
    dssn_th = st.sidebar.number_input("ベースラインパラメーターを入力してください:", 1, 10000, value=1000, step=1) / 1e7
    savgol_wsize = st.sidebar.number_input("移動平均のウィンドウサイズを入力してください:", 5, 101, step=2, key="savgol_wsize")
    
    st.sidebar.subheader("ピーク検出設定")
    
    spectrum_type = st.sidebar.selectbox(
        "解析スペクトル:", ["ベースライン削除", "移動平均後"], 
        index=0, key="spectrum_type_select"
    )
    
    # ─── 2次微分平滑化 ────────────────────────────────────────
    second_deriv_smooth = st.sidebar.number_input(
        "2次微分平滑化:",
        min_value=3,
        max_value=35,
        step=2,
        key="second_deriv_smooth"
    )

    # ─── 2次微分閾値 ─────────────────────────────────────────
    second_deriv_threshold = st.sidebar.number_input(
        "2次微分閾値:",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key="second_deriv_threshold"
    )

    # ─── ピーク卓立度閾値 ────────────────────────────────────
    peak_prominence_threshold = st.sidebar.number_input(
        "ピーク卓立度閾値:",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key="prominence_threshold"
    )
    
    # ファイルアップロード - 修正版
    uploaded_file = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（単数）",
        type=['csv', 'txt'],
        accept_multiple_files=False,  # 単一ファイル
        key="mv_uploader"
    )
    
    # アップロードファイル変更検出 - 修正版
    if uploaded_file is not None:
        new_filenames = [uploaded_file.name]
        uploaded_files = [uploaded_file]  # リスト形式に統一
    else:
        new_filenames = []
        uploaded_files = []
    
    prev_filenames = st.session_state.get("uploaded_filenames", [])

    # 設定変更検出
    config_keys = ["spectrum_type_select", "second_deriv_smooth", "second_deriv_threshold", "prominence_threshold"]
    config_changed = any(
        st.session_state.get(f"prev_{key}") != st.session_state[key] for key in config_keys
    )
    file_changed = new_filenames != prev_filenames

    # 手動ピーク初期化条件
    if config_changed or file_changed:
        for key in list(st.session_state.keys()):
            if key.endswith("_manual_peaks") or key.endswith("_excluded_peaks"):
                del st.session_state[key]
        st.session_state["manual_peak_keys"] = []
        st.session_state["uploaded_filenames"] = new_filenames
        for k in config_keys:
            st.session_state[f"prev_{k}"] = st.session_state[k]
            
    file_labels = []
    all_spectra = []
    all_bsremoval_spectra = []
    all_averemoval_spectra = []
    all_wavenum = []
    
    if uploaded_files:  # 修正版：uploaded_filesがリストになっている
        config_keys = [
            "spectrum_type_select",
            "second_deriv_smooth",
            "second_deriv_threshold",
            "prominence_threshold"
        ]
        # セーフな代入処理
        for k in config_keys:
            st.session_state[f"prev_{k}"] = st.session_state.get(k)

        # ファイル処理
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
                all_wavenum.append(wavenum)
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)
                all_averemoval_spectra.append(Averemoval_specta_pos)
                
            except Exception as e:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {e}")
        
        # ピーク検出の実行
        if 'peak_detection_triggered' not in st.session_state:
            st.session_state['peak_detection_triggered'] = False
    
        if st.button("ピーク検出を実行"):
            st.session_state['peak_detection_triggered'] = True
        
        if st.session_state['peak_detection_triggered']:
            perform_peak_detection(
                file_labels, all_wavenum, all_bsremoval_spectra, all_averemoval_spectra,
                spectrum_type, second_deriv_smooth, second_deriv_threshold, peak_prominence_threshold
            )

def perform_peak_detection(file_labels, all_wavenum, all_bsremoval_spectra, all_averemoval_spectra,
                         spectrum_type, second_deriv_smooth, second_deriv_threshold, peak_prominence_threshold):
    """
    ピーク検出を実行
    """
    st.subheader("ピーク検出結果")
    
    peak_results = []
    
    # 現在の設定を表示
    st.info(f"""
    **検出設定:**
    - スペクトルタイプ: {spectrum_type}
    - 2次微分平滑化: {second_deriv_smooth}, 閾値: {second_deriv_threshold} (ピーク検出用)
    - ピーク卓立度閾値: {peak_prominence_threshold}
    """)
    
    for i, file_name in enumerate(file_labels):
        # 選択されたスペクトルタイプに応じてデータを選択
        if spectrum_type == "ベースライン削除":
            selected_spectrum = all_bsremoval_spectra[i]
        else:
            selected_spectrum = all_averemoval_spectra[i]
        
        wavenum = all_wavenum[i]
        
        # 2次微分計算
        if len(selected_spectrum) > second_deriv_smooth:
            wl = int(second_deriv_smooth)
            second_derivative = savgol_filter(selected_spectrum, wl, 2, deriv=2)
        else:
            second_derivative = np.gradient(np.gradient(selected_spectrum))
        
        # 2次微分のみによるピーク検出
        peaks, properties = find_peaks(-second_derivative, height=second_deriv_threshold)
        all_peaks, properties = find_peaks(-second_derivative)

        if len(peaks) > 0:
            prominences = peak_prominences(-second_derivative, peaks)[0]
            all_prominences = peak_prominences(-second_derivative, all_peaks)[0]

            # Prominence 閾値でフィルタリング
            mask = prominences > peak_prominence_threshold
            filtered_peaks = peaks[mask]
            filtered_prominences = prominences[mask]
            
            # ピーク位置の補正
            corrected_peaks = []
            corrected_prominences = []
            
            for peak_idx, prom in zip(filtered_peaks, filtered_prominences):
                window_start = max(0, peak_idx - 2)
                window_end = min(len(selected_spectrum), peak_idx + 3)
                local_window = selected_spectrum[window_start:window_end]
                
                local_max_idx = np.argmax(local_window)
                corrected_idx = window_start + local_max_idx
            
                corrected_peaks.append(corrected_idx)
                
                local_prom = peak_prominences(-second_derivative, [corrected_idx])[0][0]
                corrected_prominences.append(local_prom)
            
            filtered_peaks = np.array(corrected_peaks)
            filtered_prominences = np.array(corrected_prominences)
        else:
            filtered_peaks = np.array([])
            filtered_prominences = np.array([])
        
        # 結果を保存
        peak_data = {
            'file_name': file_name,
            'detected_peaks': filtered_peaks,
            'detected_prominences': filtered_prominences,
            'wavenum': wavenum,
            'spectrum': selected_spectrum,
            'second_derivative': second_derivative,
            'second_deriv_smooth': second_deriv_smooth,
            'second_deriv_threshold': second_deriv_threshold,
            'prominence_threshold': peak_prominence_threshold,
            'all_peaks': all_peaks,
            'all_prominences': all_prominences,
        }
        peak_results.append(peak_data)
        
        # ピーク情報をテーブルで表示
        if len(filtered_peaks) > 0:
            peak_wavenums = wavenum[filtered_peaks]
            peak_intensities = selected_spectrum[filtered_peaks]
            st.write("**検出されたピーク:**")
            peak_table = pd.DataFrame({
                'ピーク番号': range(1, len(peak_wavenums) + 1),
                '波数 (cm⁻¹)': [f"{wn:.1f}" for wn in peak_wavenums],
                '強度': [f"{intensity:.3f}" for intensity in peak_intensities],
                '卓立度': [f"{prom:.4f}" for prom in filtered_prominences]
            })
            st.table(peak_table)
        else:
            st.write("ピークが検出されませんでした")
    
    # ファイルごとの描画と詳細解析
    for result in peak_results:
        file_key = result['file_name']
        # ここで必ず初期化する
        if f"{file_key}_manual_peaks" not in st.session_state:
            st.session_state[f"{file_key}_manual_peaks"] = []
        if f"{file_key}_excluded_peaks" not in st.session_state:
            st.session_state[f"{file_key}_excluded_peaks"] = set()

        render_static_plot_with_manual_controls(
            result,
            result['file_name'],
            spectrum_type
        )
    
    # ピーク解析結果の集計とダウンロード
    all_peaks_data = []
    for result in peak_results:
        file_key = result['file_name']
        
        # 有効なピーク（除外されていない自動検出ピーク + 手動追加ピーク）
        excluded_peaks = st.session_state.get(f"{file_key}_excluded_peaks", set())
        valid_auto_peaks = [i for i in result['detected_peaks'] if i not in excluded_peaks]
        
        # 手動追加ピーク
        manual_peaks = st.session_state.get(f"{file_key}_manual_peaks", [])
        
        # 自動検出ピーク（有効なもの）
        for j, idx in enumerate(valid_auto_peaks):
            wn = result['wavenum'][idx]
            intensity = result['spectrum'][idx]
            
            # Prominenceを取得
            prom_idx = np.where(result['detected_peaks'] == idx)[0]
            if len(prom_idx) > 0:
                prom = result['detected_prominences'][prom_idx[0]]
            else:
                prom = 0.0

            # FWHM の計算
            fwhm = calculate_peak_width(
                spectrum=result['spectrum'],
                peak_idx=idx,
                wavenum=result['wavenum']
            )
            # 面積の計算
            start_idx, end_idx = find_peak_width(
                spectra=result['spectrum'],
                first_dev=result['second_derivative'],
                peak_position=idx,
                window_size=20
            )
            area = find_peak_area(
                spectra=result['spectrum'],
                local_start_idx=start_idx,
                local_end_idx=end_idx
            )

            all_peaks_data.append({
                'ファイル名': file_key,
                'ピーク種別': '自動検出',
                'ピーク番号': j + 1,
                '波数 (cm⁻¹)': f"{wn:.1f}",
                '強度 (a.u.)': f"{intensity:.6f}",
                'Prominence': f"{prom:.6f}",
                '半値幅 FWHM (cm⁻¹)': f"{fwhm:.2f}",
                'ピーク面積 (a.u.)': f"{area:.4f}",
            })
        
        # 手動追加ピーク
        for j, wn in enumerate(manual_peaks):
            idx = np.argmin(np.abs(result['wavenum'] - wn))
            intensity = result['spectrum'][idx]
            
            try:
                prom = peak_prominences(-result['second_derivative'], [idx])[0][0]
            except:
                prom = 0.0

            # FWHM の計算
            fwhm = calculate_peak_width(
                spectrum=result['spectrum'],
                peak_idx=idx,
                wavenum=result['wavenum']
            )
            # 面積の計算
            start_idx, end_idx = find_peak_width(
                spectra=result['spectrum'],
                first_dev=result['second_derivative'],
                peak_position=idx,
                window_size=20
            )
            area = find_peak_area(
                spectra=result['spectrum'],
                local_start_idx=start_idx,
                local_end_idx=end_idx
            )

            all_peaks_data.append({
                'ファイル名': file_key,
                'ピーク種別': '手動追加',
                'ピーク番号': j + 1,
                '波数 (cm⁻¹)': f"{wn:.1f}",
                '強度 (a.u.)': f"{intensity:.6f}",
                'Prominence': f"{prom:.6f}",
                '半値幅 FWHM (cm⁻¹)': f"{fwhm:.2f}",
                'ピーク面積 (a.u.)': f"{area:.4f}",
            })

    if all_peaks_data:
        peaks_df = pd.DataFrame(all_peaks_data)
        st.subheader("✨ ピーク解析結果 (強度・Prominence・FWHM・面積)")
        st.dataframe(peaks_df, use_container_width=True)

        csv = peaks_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="🔽 ピーク解析結果をCSVでダウンロード",
            data=csv,
            file_name=f"peak_analysis_results_{spectrum_type}.csv",
            mime="text/csv"
        )

def render_static_plot_with_manual_controls(result, file_key, spectrum_type):
    """
    静的プロットと手動ピーク制御UIの描画
    """
    st.subheader(f"📊 {file_key} - {spectrum_type}")
    
    # ---- セッション初期化 ----
    if f"{file_key}_excluded_peaks" not in st.session_state:
        st.session_state[f"{file_key}_excluded_peaks"] = set()
    if f"{file_key}_manual_peaks" not in st.session_state:
        st.session_state[f"{file_key}_manual_peaks"] = []

    # ---- 手動制御UI ----
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔹 ピーク手動追加**")
        add_wavenum = st.number_input(
            "追加する波数 (cm⁻¹):",
            min_value=float(result['wavenum'].min()),
            max_value=float(result['wavenum'].max()),
            value=float(result['wavenum'][len(result['wavenum'])//2]),
            step=1.0,
            key=f"add_wavenum_{file_key}"
        )
        
        if st.button(f"波数 {add_wavenum:.1f} のピークを追加", key=f"add_peak_{file_key}"):
            # 重複チェック（±2 cm⁻¹以内）
            is_duplicate = any(abs(existing_wn - add_wavenum) < 2.0 
                             for existing_wn in st.session_state[f"{file_key}_manual_peaks"])
            
            if not is_duplicate:
                st.session_state[f"{file_key}_manual_peaks"].append(add_wavenum)
                st.success(f"波数 {add_wavenum:.1f} cm⁻¹ にピークを追加しました")
                st.rerun()
            else:
                st.warning("近接する位置にすでにピークが存在します")
    
    with col2:
        st.write("**🔸 検出ピーク除外**")
        if len(result['detected_peaks']) > 0:
            # 検出ピークの選択肢を作成
            detected_options = []
            for i, idx in enumerate(result['detected_peaks']):
                wn = result['wavenum'][idx]
                intensity = result['spectrum'][idx]
                status = "除外済み" if idx in st.session_state[f"{file_key}_excluded_peaks"] else "有効"
                detected_options.append(f"ピーク{i+1}: {wn:.1f} cm⁻¹ ({intensity:.3f}) - {status}")
            
            selected_peak = st.selectbox(
                "除外/復活させるピークを選択:",
                options=range(len(detected_options)),
                format_func=lambda x: detected_options[x],
                key=f"select_peak_{file_key}"
            )
            
            peak_idx = result['detected_peaks'][selected_peak]
            is_excluded = peak_idx in st.session_state[f"{file_key}_excluded_peaks"]
            
            if is_excluded:
                if st.button(f"ピーク{selected_peak+1}を復活", key=f"restore_peak_{file_key}"):
                    st.session_state[f"{file_key}_excluded_peaks"].remove(peak_idx)
                    st.success(f"ピーク{selected_peak+1}を復活させました")
                    st.rerun()
            else:
                if st.button(f"ピーク{selected_peak+1}を除外", key=f"exclude_peak_{file_key}"):
                    st.session_state[f"{file_key}_excluded_peaks"].add(peak_idx)
                    st.success(f"ピーク{selected_peak+1}を除外しました")
                    st.rerun()
        else:
            st.info("検出されたピークがありません")

    # ---- 手動追加ピーク管理テーブル ----
    if st.session_state[f"{file_key}_manual_peaks"]:
        st.write("**📝 手動追加ピーク一覧**")
        manual_peaks = st.session_state[f"{file_key}_manual_peaks"]
        
        # テーブル作成
        manual_data = []
        for i, wn in enumerate(manual_peaks):
            idx = np.argmin(np.abs(result['wavenum'] - wn))
            intensity = result['spectrum'][idx]
            manual_data.append({
                '番号': i + 1,
                '波数 (cm⁻¹)': f"{wn:.1f}",
                '強度': f"{intensity:.3f}"
            })
        
        manual_df = pd.DataFrame(manual_data)
        st.dataframe(manual_df, use_container_width=True)
        
        # 削除選択
        if len(manual_peaks) > 0:
            col_del1, col_del2 = st.columns([3, 1])
            with col_del1:
                delete_idx = st.selectbox(
                    "削除する手動ピークを選択:",
                    options=range(len(manual_peaks)),
                    format_func=lambda x: f"ピーク{x+1}: {manual_peaks[x]:.1f} cm⁻¹",
                    key=f"delete_manual_{file_key}"
                )
            with col_del2:
                if st.button("削除", key=f"delete_manual_btn_{file_key}"):
                    removed_wn = st.session_state[f"{file_key}_manual_peaks"].pop(delete_idx)
                    st.success(f"波数 {removed_wn:.1f} cm⁻¹ のピークを削除しました")
                    st.rerun()

    # ---- フィルタリング済みピーク配列 ----
    filtered_peaks = [
        i for i in result["detected_peaks"]
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result["detected_peaks"], result["detected_prominences"])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    
    # ---- 静的プロット描画 ----
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.4, 0.3, 0.3]
    )

    # 1段目：メインスペクトル
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'],
            y=result['spectrum'],
            mode='lines',
            name=spectrum_type,
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )

    # 自動検出ピーク（有効なもののみ）
    if len(filtered_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][filtered_peaks],
                y=result['spectrum'][filtered_peaks],
                mode='markers',
                name='検出ピーク（有効）',
                marker=dict(color='red', size=8, symbol='circle')
            ),
            row=1, col=1
        )

    # 除外されたピーク
    excluded_peaks = list(st.session_state[f"{file_key}_excluded_peaks"])
    if len(excluded_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][excluded_peaks],
                y=result['spectrum'][excluded_peaks],
                mode='markers',
                name='除外ピーク',
                marker=dict(color='gray', size=8, symbol='x')
            ),
            row=1, col=1
        )

    # 手動ピーク
    for wn in st.session_state[f"{file_key}_manual_peaks"]:
        idx = np.argmin(np.abs(result['wavenum'] - wn))
        intensity = result['spectrum'][idx]
        fig.add_trace(
            go.Scatter(
                x=[wn],
                y=[intensity],
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='star'),
                text=["手動"],
                textposition='top center',
                name="手動ピーク",
                showlegend=False
            ),
            row=1, col=1
        )

    # 2段目：2次微分
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'],
            y=result['second_derivative'],
            mode='lines',
            name='2次微分',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

    # 3段目：Prominenceプロット
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'][result['all_peaks']],
            y=result['all_prominences'],
            mode='markers',
            name='全ピークの卓立度',
            marker=dict(color='orange', size=4)
        ),
        row=3, col=1
    )
    if len(filtered_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][filtered_peaks],
                y=filtered_prominences,
                mode='markers',
                name='有効な卓立度',
                marker=dict(color='red', size=7, symbol='circle')
            ),
            row=3, col=1
        )

    fig.update_layout(height=800, margin=dict(t=80, b=150))
    
    for r in [1, 2, 3]:
        fig.update_xaxes(
            showticklabels=True,
            title_text="波数 (cm⁻¹)" if r == 3 else "",
            row=r, col=1,
            automargin=True
        )
    
    fig.update_yaxes(title_text="Intensity (a.u.)", row=1, col=1)
    fig.update_yaxes(title_text="2次微分", row=2, col=1)
    fig.update_yaxes(title_text="Prominence", row=3, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

    # ---- グリッドサーチ機能 ----
    render_gridsearch_controls(result, file_key)

def render_gridsearch_controls(result, file_key):
    """
    グリッドサーチ制御UIの描画
    """
    st.subheader("🔍 グリッドサーチ最適化")
    
    with st.expander("グリッドサーチ実行", expanded=False):
        st.write("手動で追加/除外したピークに基づいて最適なパラメータを探索します")
        
        if st.button("🔁 最適閾値を探索", key=f"optimize_{file_key}"):
            # 手動追加ピーク
            manual_add = st.session_state.get(f"{file_key}_manual_peaks", [])
            
            # 除外ピークインデックス
            manual_exclude = st.session_state.get(f"{file_key}_excluded_peaks", set())
        
            smooth_list = list(range(3, 26, 2))
            result_opt = optimize_thresholds_via_gridsearch(
                wavenum=result['wavenum'],
                spectrum=result['spectrum'],
                manual_add_peaks=manual_add,
                manual_exclude_indices=manual_exclude,
                current_prom_thres=st.session_state['prominence_threshold'],
                current_deriv_thres=st.session_state['second_deriv_threshold'],
                current_smooth=st.session_state['second_deriv_smooth'],
                detected_original_peaks=result["detected_peaks"],
                resolution=40,
                smooth_range=smooth_list
            )
        
            st.session_state[f"{file_key}_grid_result"] = result_opt

            # temp に保存（次回実行時に使用）
            st.session_state["second_deriv_smooth_temp"] = int(result_opt["second_deriv_smooth"])
            st.session_state["prominence_threshold_temp"] = float(result_opt["prominence_threshold"])
            st.session_state["second_deriv_threshold_temp"] = float(result_opt["second_deriv_threshold"])
            st.session_state["apply_grid_result"] = True
            
            st.rerun()
        
        # グリッドサーチ結果表示
        if f"{file_key}_grid_result" in st.session_state:
            result_grid = st.session_state[f"{file_key}_grid_result"]
            st.success(f"""
            ✅ グリッドサーチ最適化結果:
            - 2次微分平滑化ウィンドウ: {int(result_grid['second_deriv_smooth'])}
            - Prominence: {result_grid['prominence_threshold']:.4f}
            - 微分閾値: {result_grid['second_deriv_threshold']:.4f}
            - スコア: {result_grid['score']}
            """)
        
            # グリッドサーチ結果で再検出
            if st.button("🔄 グリッドサーチ結果で再検出", key=f"reapply_{file_key}"):
                # 手動設定をクリア
                st.session_state[f"{file_key}_manual_peaks"] = []
                st.session_state[f"{file_key}_excluded_peaks"] = set()
            
                # 再検出フラグを設定
                st.session_state["apply_grid_result"] = True
                st.session_state["peak_detection_triggered"] = True
            
                st.rerun()
