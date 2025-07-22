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

# Interactive plotting
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    st.warning("streamlit_plotly_events not available. Interactive peak selection may not work.")
    plotly_events = None

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
    prom_range = list(range(10, 401, 10))
    deriv_range = list(range(10, 401, 10))

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
                    score += 2 if idx in final_peaks else -1
    
                # 2. 手動追加ピークを正しく拾えたか（+2）/ 見逃したか（-1）
                for x, _ in manual_add_peaks:
                    idx = np.argmin(np.abs(wavenum - x))
                    score += 2 if idx in final_peaks else -1
    
                # 3. 手動除外ピークを正しく除外できたか（+2）/ 残ってしまったか（-1）
                for idx in manual_exclude_indices:
                    score += 2 if idx not in final_peaks else -1
    
                # 4. 余分なピークはペナルティ
                for idx in final_peaks:
                    if idx not in orig_peaks and all(abs(x - wavenum[idx]) > 0 for x, _ in manual_add_peaks):
                        score -= 2

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
    Peak analysis mode
    """
    st.header("ラマンピークファインダー")
    
    # 事前パラメータ
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    
    # temporary変数の処理
    for param in ["second_deriv_smooth", "prominence_threshold", "second_deriv_threshold"]:
        temp_key = f"{param}_temp"
        if temp_key in st.session_state:
            st.session_state[param] = st.session_state.pop(temp_key)
            
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
    
    second_deriv_smooth = st.sidebar.number_input(
        "2次微分平滑化:", 3, 35,
        step=2, key="second_deriv_smooth"
    )
    
    second_deriv_threshold = st.sidebar.number_input(
        "2次微分閾値:",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key="second_deriv_threshold"
    )
    
    peak_prominence_threshold = st.sidebar.number_input(
        "ピーク卓立度閾値:",
        min_value=0.0,
        max_value=1000.0,
        step=10.0,
        key="prominence_threshold"
    )

    # ファイルアップロード
    uploaded_files = st.file_uploader(
        "ファイルを選択してください",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple CSV files with spectral data. Files should be named as GroupName_Number.csv",
        key="mv_uploader"
    )
    
    # アップロードファイル変更検出
    new_filenames = [f.name for f in uploaded_files] if uploaded_files else []
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
            if key.endswith("_manual_peaks"):
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
    
    if uploaded_files:
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
        # ▼ ここで必ず初期化する
        if f"{file_key}_manual_peaks" not in st.session_state:
            st.session_state[f"{file_key}_manual_peaks"] = []
        if f"{file_key}_excluded_peaks" not in st.session_state:
            st.session_state[f"{file_key}_excluded_peaks"] = set()

        render_interactive_plot(
            result,
            result['file_name'],
            spectrum_type
        )
    
    # ピーク解析結果の集計とダウンロード
    all_peaks_data = []
    for result in peak_results:
        detected = result['detected_peaks']
        prominences = result['detected_prominences']
        for j, idx in enumerate(detected):
            wn = result['wavenum'][idx]
            intensity = result['spectrum'][idx]
            prom = prominences[j]

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
        st.table(peaks_df)

        csv = peaks_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="🔽 ピーク解析結果をCSVでダウンロード",
            data=csv,
            file_name=f"peak_analysis_results_{spectrum_type}.csv",
            mime="text/csv"
        )
def render_interactive_plot(result, file_key, spectrum_type):
    """インタラクティブプロットを描画"""
    # 除外を反映したピークインデックス
    filtered_peaks = [
        i for i in result['detected_peaks']
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result['detected_peaks'], result['detected_prominences'])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]

    # フィギュア作成
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f'{file_key} - {spectrum_type}',
            f'{file_key} - 微分スペクトル比較',
            f'{file_key} - Prominence vs 波数'
        ],
        vertical_spacing=0.07,
        row_heights=[0.4, 0.3, 0.3]
    )

    # 上段スペクトル
    fig.add_trace(
        go.Scatter(x=result['wavenum'], y=result['spectrum'], mode='lines', name=spectrum_type),
        row=1, col=1
    )
    # 自動検出ピーク
    if filtered_peaks:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][filtered_peaks],
                y=result['spectrum'][filtered_peaks],
                mode='markers', name='検出ピーク（有効）', marker=dict(size=8, symbol='circle')
            ),
            row=1, col=1
        )
    # 除外ピーク
    excl = list(st.session_state[f"{file_key}_excluded_peaks"])
    if excl:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][excl],
                y=result['spectrum'][excl],
                mode='markers', name='除外ピーク', marker=dict(symbol='x', size=8)
            ),
            row=1, col=1
        )
    # 手動ピーク
    for x, y in st.session_state[f"{file_key}_manual_peaks"]:
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y], mode='markers+text', text=["手動"],
                textposition='top center', name="手動ピーク", marker=dict(symbol='star', size=10)
            ), row=1, col=1
        )

    # 2次微分
    fig.add_trace(
        go.Scatter(x=result['wavenum'], y=result['second_derivative'], mode='lines', name='2次微分'),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", row=2, col=1)

    # Prominence
    fig.add_trace(
        go.Scatter(x=result['wavenum'][result['all_peaks']], y=result['all_prominences'], mode='markers', name='全ピークの卓立度', marker=dict(size=4)),
        row=3, col=1
    )
    if filtered_peaks:
        fig.add_trace(
            go.Scatter(
                x=result['wavenum'][filtered_peaks],
                y=filtered_prominences,
                mode='markers', name='有効な卓立度', marker=dict(symbol='circle', size=7)
            ), row=3, col=1
        )

    fig.update_layout(height=800, margin=dict(t=80, b=150))
    fig.update_xaxes(title_text="波数 (cm⁻¹)", row=3, col=1)

    # ① まず常に描画
    st.plotly_chart(fig, use_container_width=True)

    # ② クリック／ダブルクリックの両方をキャプチャ
    if plotly_events:
        event_key = f"{file_key}_click_event"
        clicked_points = plotly_events(
            fig,
            click_event=True,
            select_event=True,
            hover_event=False,
            override_height=800,
            key=event_key
        ) or []
        
        # クリック処理
        for pt in clicked_points:
            if pt["curveNumber"] == 0:  # メインスペクトルレイヤー
                x, y = pt["x"], pt["y"]
                idx = np.argmin(np.abs(result['wavenum'] - x))
             # 自動検出ピークならトグル、違えば手動追加
            if idx in result['detected_peaks']:
                excl = st.session_state[f"{file_key}_excluded_peaks"]
                if idx in excl: excl.remove(idx)
                else: excl.add(idx)
            else:
                existing = [abs(px - x) < 1.0 for px, _ in st.session_state[f"{file_key}_manual_peaks"]]
                if not any(existing):
                    st.session_state[f"{file_key}_manual_peaks"].append((x, y))
    else:
        st.info("Interactive peak selection is unavailable. 'streamlit_plotly_events'をインストールしてください。")
        
def render_peak_analysis(result, spectrum_type):
    """
    個別ファイルのピーク解析結果を描画
    """
    file_key = result['file_name']

    # 初期化
    if f"{file_key}_excluded_peaks" not in st.session_state:
        st.session_state[f"{file_key}_excluded_peaks"] = set()
    if f"{file_key}_manual_peaks" not in st.session_state:
        st.session_state[f"{file_key}_manual_peaks"] = []

    # 除外を反映したピークインデックス
    filtered_peaks = [
        i for i in result['detected_peaks']
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result['detected_peaks'], result['detected_prominences'])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f'{file_key} - {spectrum_type}',
            f'{file_key} - 微分スペクトル比較',
            f'{file_key} - Prominence vs 波数'
        ],
        vertical_spacing=0.07,
        row_heights=[0.4, 0.3, 0.3]
    )

    # 上段スペクトル
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'],
            y=result['spectrum'],
            mode='lines',
            name=spectrum_type,
            line=dict(color='blue', width=1)
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
    for x, y in st.session_state[f"{file_key}_manual_peaks"]:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='star'),
                text=["手動"],
                textposition='top center',
                name="手動ピーク",
                showlegend=False
            ),
            row=1, col=1
        )

    # 2次微分
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

    # Prominenceプロット
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
    st.plotly_chart(fig, use_container_width=True)
    
    # クリック処理 
    if plotly_events:
        event_key = f"{file_key}_click_event"
        clicked_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=800,
            key=event_key
        )
        
        clicked_main = [pt for pt in clicked_points if pt["curveNumber"] == 0]
        
        if clicked_main:
            pt = clicked_main[-1]
            click_id = str(pt['x']) + str(pt['y'])
        
            last_click_id = st.session_state.get(f"{event_key}_last", None)
            if click_id != last_click_id:
                st.session_state[f"{event_key}_last"] = click_id
        
                x = pt['x']
                y = pt['y']
                wavenum_arr = result['wavenum']
                idx = np.argmin(np.abs(wavenum_arr - x))
        
                # 自動検出ピークならトグル
                if idx in result['detected_peaks']:
                    if idx in st.session_state[f"{file_key}_excluded_peaks"]:
                        st.session_state[f"{file_key}_excluded_peaks"].remove(idx)
                    else:
                        st.session_state[f"{file_key}_excluded_peaks"].add(idx)
                else:
                    # すでに同じ場所に手動ピークがあれば追加しない
                    is_duplicate = any(abs(existing_x - x) < 1.0 for existing_x, _ in st.session_state[f"{file_key}_manual_peaks"])
                    if not is_duplicate:
                        st.session_state[f"{file_key}_manual_peaks"].append((x, y))
    else:
        st.plotly_chart(fig, use_container_width=True)
        st.info("Interactive peak selection not available. Please install streamlit_plotly_events.")
    
    # 手動ピーク情報とグリッドサーチ
    render_manual_peak_info(result, file_key)

def render_manual_peak_info(result, file_key):
    """
    手動ピーク情報とグリッドサーチを表示
    """
    # セッションフラグの初期化
    if f"show_info_{file_key}" not in st.session_state:
        st.session_state[f"show_info_{file_key}"] = False
    
    # 手動ピークの情報を表示ボタン
    if st.button("🔍 手動ピークの情報を表示", key=f"show_manual_info_{file_key}"):
        st.session_state[f"show_info_{file_key}"] = True
    
    # 表示フラグに応じて手動情報とグリッドサーチ処理を実行
    if st.session_state[f"show_info_{file_key}"]:
        manual_peaks = st.session_state.get(f"{file_key}_manual_peaks", [])
        excluded_peaks = st.session_state.get(f"{file_key}_excluded_peaks", set())
    
        wavenum = result['wavenum']
        spectrum = result['spectrum']
        second_derivative = result['second_derivative']
    
        # 手動で追加されたピーク情報
        manual_peak_table = []
        if manual_peaks:
            for x, y in manual_peaks:
                idx = np.argmin(np.abs(wavenum - x))
                window_size = 15
                local_start = max(0, idx - window_size)
                local_end = min(len(second_derivative), idx + window_size + 1)
                local_window = -second_derivative[local_start:local_end]
                local_max_idx = np.argmax(local_window)
                peak_idx = local_start + local_max_idx
    
                try:
                    prom = peak_prominences(-second_derivative, [peak_idx])[0][0]
                except Exception:
                    prom = 0.0
    
                manual_peak_table.append({
                    "波数 (cm⁻¹)": f"{x:.1f}",
                    "強度": f"{y:.3f}",
                    "卓立度": f"{prom:.3f}"
                })
    
            st.write(f"**{file_key} の手動で追加されたピーク:**")
            st.table(pd.DataFrame(manual_peak_table))
        else:
            st.info("手動で追加されたピークはありません。")
    
        # 手動で除外されたピーク情報
        excluded_table = []
        if excluded_peaks:
            for idx in sorted(excluded_peaks):
                if idx < len(wavenum):
                    x = wavenum[idx]
                    y = spectrum[idx]
                    try:
                        prom = peak_prominences(-second_derivative, [idx])[0][0]
                    except Exception:
                        prom = 0.0
    
                    excluded_table.append({
                        "波数 (cm⁻¹)": f"{x:.1f}",
                        "強度": f"{y:.3f}",
                        "卓立度": f"{prom:.3f}"
                    })
    
            st.write(f"**{file_key} の手動で除外された（除外マーク付き）ピーク:**")
            st.table(pd.DataFrame(excluded_table))
        else:
            st.info("手動で除外されたピークはありません。")
    
        # グリッドサーチ実行ボタン
        if st.button("🔁 最適閾値を探索", key=f"optimize_{file_key}"):
            # 手動追加ピークを (x, y) のタプルに変換
            manual_add = [
                (float(row["波数 (cm⁻¹)"]), float(row["強度"]))
                for row in manual_peak_table
            ]
        
            # 除外ピークインデックス
            manual_exclude = set()
            for row in excluded_table:
                x = float(row["波数 (cm⁻¹)"])
                idx4 = np.argmin(np.abs(wavenum - x))
                manual_exclude.add(idx4)
        
            smooth_list = list(range(3, 26, 2))
            result_opt = optimize_thresholds_via_gridsearch(
                wavenum=wavenum,
                spectrum=spectrum,
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

            # temp に保存
            st.session_state["second_deriv_smooth_temp"] = int(result_opt["second_deriv_smooth"])
            st.session_state["prominence_threshold_temp"] = float(result_opt["prominence_threshold"])
            st.session_state["second_deriv_threshold_temp"] = float(result_opt["second_deriv_threshold"])
            
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
                st.session_state[f"{file_key}_manual_peaks"] = []
                st.session_state[f"{file_key}_excluded_peaks"] = set()
            
                # セッションに一時的に保存
                st.session_state["prominence_threshold_temp"] = float(result_grid["prominence_threshold"])
                st.session_state["second_deriv_threshold_temp"] = float(result_grid["second_deriv_threshold"])
            
                st.session_state["peak_detection_triggered"] = True
            
                st.rerun()
