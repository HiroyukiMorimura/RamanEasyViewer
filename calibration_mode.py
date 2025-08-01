# -*- coding: utf-8 -*-
"""
検量線作成モジュール
単一ピーク面積またはPLS回帰による検量線作成機能
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from numpy import trapz
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
import io

# 共通ユーティリティ関数のインポート
from common_utils import (
    detect_file_type, read_csv_file, find_index, WhittakerSmooth, airPLS,
    remove_outliers_and_interpolate, process_spectrum_file
)

class CalibrationAnalyzer:
    def __init__(self):
        self.spectra_data = []
        self.concentrations = []
        self.wavenumbers = None
        self.calibration_model = None
        self.calibration_type = None
        self.wave_range = None
        self.fitted_params = None
        
    def lorentzian(self, x, amplitude, center, gamma, baseline=0):
        """ローレンツ関数 + ベースライン"""
        return baseline + amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    def fit_single_peak(self, x, y, initial_center=None):
        """単一ピークのローレンツフィッティング"""
        try:
            # 初期パラメータの推定
            if initial_center is None:
                center_idx = np.argmax(y)
                initial_center = x[center_idx]
            else:
                center_idx = find_index(x, initial_center)
            
            initial_amplitude = np.max(y) - np.min(y)
            initial_gamma = (x[-1] - x[0]) / 20
            initial_baseline = np.min(y)
            
            # パラメータの境界設定
            bounds_lower = [0, x[0], 0.1, -np.inf]
            bounds_upper = [initial_amplitude * 5, x[-1], (x[-1] - x[0]) / 2, np.inf]
            
            # フィッティング実行
            popt, pcov = curve_fit(
                self.lorentzian,
                x, y,
                p0=[initial_amplitude, initial_center, initial_gamma, initial_baseline],
                bounds=(bounds_lower, bounds_upper),
                maxfev=10000
            )
            
            return popt, pcov
            
        except Exception as e:
            st.error(f"フィッティングエラー: {str(e)}")
            return None, None
    
    def calculate_peak_area(self, amplitude, gamma):
        """ローレンツ関数の解析的面積計算"""
        return np.pi * amplitude * gamma
    
    def process_spectra_files(self, uploaded_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
        """複数のスペクトルファイルを処理"""
        self.spectra_data = []
        processed_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"ファイル処理中: {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            try:
                wavenum, raw_spectrum, corrected_spectrum, smoothed_spectrum, file_type, file_name = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                
                if wavenum is not None:
                    self.spectra_data.append({
                        'filename': file_name,
                        'wavenumbers': wavenum,
                        'raw_spectrum': raw_spectrum,
                        'corrected_spectrum': smoothed_spectrum,  # 移動平均後のスペクトルを使用
                        'file_type': file_type
                    })
                    processed_files.append(file_name)
                else:
                    st.warning(f"ファイル {file_name} の処理に失敗しました")
                    
            except Exception as e:
                st.error(f"ファイル処理エラー ({uploaded_file.name}): {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if self.spectra_data:
            self.wavenumbers = self.spectra_data[0]['wavenumbers']
            st.success(f"{len(self.spectra_data)}個のファイルを正常に処理しました")
            return processed_files
        else:
            st.error("処理可能なファイルがありませんでした")
            return []
    
    def create_peak_area_calibration(self, wave_start, wave_end, peak_center=None):
        """ピーク面積による検量線作成"""
        areas = []
        fitting_results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, spectrum_data in enumerate(self.spectra_data):
            status_text.text(f"ピークフィッティング中: {spectrum_data['filename']} ({i+1}/{len(self.spectra_data)})")
            progress_bar.progress((i + 1) / len(self.spectra_data))
            
            # 波数範囲の切り出し
            wavenum = spectrum_data['wavenumbers']
            spectrum = spectrum_data['corrected_spectrum']
            
            start_idx = find_index(wavenum, wave_start)
            end_idx = find_index(wavenum, wave_end)
            
            x_range = wavenum[start_idx:end_idx+1]
            y_range = spectrum[start_idx:end_idx+1]
            
            # ローレンツフィッティング
            popt, pcov = self.fit_single_peak(x_range, y_range, peak_center)
            
            if popt is not None:
                amplitude, center, gamma, baseline = popt
                area = self.calculate_peak_area(amplitude, gamma)
                areas.append(area)
                
                fitting_results.append({
                    'filename': spectrum_data['filename'],
                    'amplitude': amplitude,
                    'center': center,
                    'gamma': gamma,
                    'baseline': baseline,
                    'area': area,
                    'x_range': x_range,
                    'y_range': y_range,
                    'fitted_curve': self.lorentzian(x_range, *popt)
                })
            else:
                areas.append(0)
                fitting_results.append(None)
        
        progress_bar.empty()
        status_text.empty()
        
        self.calibration_type = 'peak_area'
        self.wave_range = [wave_start, wave_end]
        self.fitted_params = fitting_results
        
        return np.array(areas), fitting_results
    
    def create_multivariate_calibration(self, wave_start, wave_end, n_components=3):
        """多変量解析による検量線作成"""
        # 波数範囲の切り出し
        start_idx = find_index(self.wavenumbers, wave_start)
        end_idx = find_index(self.wavenumbers, wave_end)
        
        x_range = self.wavenumbers[start_idx:end_idx+1]
        
        # スペクトルデータの準備
        X = []
        for spectrum_data in self.spectra_data:
            spectrum = spectrum_data['corrected_spectrum']
            y_range = spectrum[start_idx:end_idx+1]
            X.append(y_range)
        
        X = np.array(X)
        y = np.array(self.concentrations)
        
        # PLS回帰
        self.calibration_model = PLSRegression(n_components=n_components)
        self.calibration_model.fit(X, y)
        
        # 予測値
        y_pred = self.calibration_model.predict(X).flatten()
        
        # クロスバリデーション
        loo = LeaveOneOut()
        cv_scores = cross_val_score(self.calibration_model, X, y, cv=loo, scoring='r2')
        
        self.calibration_type = 'multivariate'
        self.wave_range = [wave_start, wave_end]
        
        return y_pred, cv_scores, X, x_range
    
    def predict_concentration(self, new_spectrum_data, wave_start, wave_end):
        """新しいスペクトルの濃度予測"""
        if self.calibration_model is None and self.calibration_type != 'peak_area':
            return None
        
        wavenum = new_spectrum_data['wavenumbers']
        spectrum = new_spectrum_data['corrected_spectrum']
        
        start_idx = find_index(wavenum, wave_start)
        end_idx = find_index(wavenum, wave_end)
        
        if self.calibration_type == 'peak_area':
            # ピーク面積による予測
            x_range = wavenum[start_idx:end_idx+1]
            y_range = spectrum[start_idx:end_idx+1]
            
            popt, pcov = self.fit_single_peak(x_range, y_range)
            if popt is not None:
                amplitude, center, gamma, baseline = popt
                area = self.calculate_peak_area(amplitude, gamma)
                return area
            else:
                return None
                
        elif self.calibration_type == 'multivariate':
            # 多変量解析による予測
            y_range = spectrum[start_idx:end_idx+1]
            X_new = y_range.reshape(1, -1)
            prediction = self.calibration_model.predict(X_new)
            return prediction[0][0]
        
        return None

def calibration_mode():
    """検量線作成モード"""
    st.header("検量線作成")
    analyzer = CalibrationAnalyzer()
    
    # サイドバー設定
    st.sidebar.subheader("データ処理設定")
    
    # データ処理パラメータ
    start_wavenum = st.sidebar.number_input("波数（開始）:", value=400, min_value=0, max_value=4000)
    end_wavenum = st.sidebar.number_input("波数（終了）:", value=2000, min_value=start_wavenum+1, max_value=4000)
    dssn_th = st.sidebar.number_input("ベースラインパラメーター:", value=1000, min_value=1, max_value=10000) / 1e7
    savgol_wsize = st.sidebar.number_input("ウィンドウサイズ:", value=5, min_value=3, max_value=101, step=2)
    
    # 複数ファイルアップロード
    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（複数可）",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        key="calibration_uploader"
    )
    
    if uploaded_files:
        # ファイル処理
        processed_files = analyzer.process_spectra_files(
            uploaded_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize
        )
        
        if processed_files:
            # matplotlibでスペクトル表示
            st.subheader("スペクトル確認")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            colors = plt.cm.Set1(np.linspace(0, 1, len(analyzer.spectra_data)))
            
            for i, spectrum_data in enumerate(analyzer.spectra_data):
                ax.plot(spectrum_data['wavenumbers'], spectrum_data['corrected_spectrum'], 
                       color=colors[i], linewidth=1.5, 
                       label=f"{spectrum_data['filename']}")
            
            ax.set_xlabel('Raman Shift (cm⁻¹)')
            ax.set_ylabel('Intensity (a.u.)')
            # ax.set_title('アップロードされたスペクトル（移動平均処理済み）')
            ax.legend(loc='upper right') 
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # 濃度データ入力
            st.subheader("濃度データ入力")
            
            # 濃度データのテーブル作成
            if f"concentration_data_{len(processed_files)}" not in st.session_state:
                st.session_state[f"concentration_data_{len(processed_files)}"] = pd.DataFrame({
                    'ファイル名': processed_files,
                    '濃度': [0.0] * len(processed_files),
                    '単位': ['mg/L'] * len(processed_files)
                })
            
            concentration_df = st.data_editor(
                st.session_state[f"concentration_data_{len(processed_files)}"],
                use_container_width=True,
                num_rows="fixed",
                column_config={
                    "ファイル名": st.column_config.TextColumn(disabled=True),
                    "濃度": st.column_config.NumberColumn(
                        "濃度",
                        help="各サンプルの濃度を入力してください",
                        min_value=0.0,
                        step=0.1,
                        format="%.3f"
                    ),
                    "単位": st.column_config.TextColumn(
                        "単位",
                        help="濃度の単位を入力してください"
                    )
                },
                key=f"concentration_editor_{len(processed_files)}"
            )
            
            # 濃度データ確定ボタン
            col_button, col_status = st.columns([1, 2])
            with col_button:
                concentration_confirmed = st.button("濃度データ確定", type="secondary")
            
            with col_status:
                if concentration_confirmed:
                    # セッション状態を更新
                    st.session_state[f"concentration_data_{len(processed_files)}"] = concentration_df
                    analyzer.concentrations = concentration_df['濃度'].values
                    st.success("濃度データを確定しました！")
                    st.session_state.concentration_confirmed = True
                elif 'concentration_confirmed' in st.session_state and st.session_state.concentration_confirmed:
                    st.info("濃度データ確定済み")
                else:
                    st.warning("濃度データを入力して確定ボタンを押してください")
            
            # 検量線設定（下側に配置）
            if 'concentration_confirmed' in st.session_state and st.session_state.concentration_confirmed:
                analyzer.concentrations = st.session_state[f"concentration_data_{len(processed_files)}"]['濃度'].values
                
                st.subheader("検量線設定")
                
                # 検量線タイプ選択
                calibration_type = st.selectbox(
                    "検量線作成方法:",
                    ["ピーク面積", "多変量解析"],
                    help="ピーク面積: 単一ピークのローレンツフィッティング面積を使用\n多変量解析: 指定波数範囲の全スペクトルデータを使用したPLS回帰"
                )
                
                # 波数範囲設定
                col1, col2 = st.columns(2)
                
                with col1:
                    analysis_start = st.number_input(
                        "解析開始波数:", 
                        value=int(analyzer.wavenumbers.min()) if analyzer.wavenumbers is not None else start_wavenum,
                        min_value=int(analyzer.wavenumbers.min()) if analyzer.wavenumbers is not None else start_wavenum,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum
                    )
                
                with col2:
                    analysis_end = st.number_input(
                        "解析終了波数:", 
                        value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum,
                        min_value=analysis_start,
                        max_value=int(analyzer.wavenumbers.max()) if analyzer.wavenumbers is not None else end_wavenum
                    )
                
                if calibration_type == "ピーク面積":
                    col3, col4 = st.columns(2)
                    # ピーク中心波数の指定（オプション）
                    with col3:
                        peak_center = st.number_input(
                            "ピーク中心波数 (cm⁻¹):",
                            value=(analysis_start + analysis_end) / 2,
                            min_value=float(analysis_start),
                            max_value=float(analysis_end),
                            help="空欄の場合は自動検出"
                        )
                    with col4:
                        use_peak_center = st.checkbox("ピーク中心を固定", value=False)
                    
                elif calibration_type == "多変量解析":
                    # PLS成分数設定
                    n_components = st.number_input(
                        "成分数:",
                        value=3,
                        min_value=1,
                        max_value=min(10, len(processed_files)-1),
                        help="多変量解析（PLS回帰）の成分数を設定"
                    )
                
                # 検量線作成実行
                if st.button("検量線作成実行", type="primary"):
                    # 濃度データの確認
                    current_concentrations = st.session_state[f"concentration_data_{len(processed_files)}"]['濃度'].values
                    analyzer.concentrations = current_concentrations
                    
                    if len(set(analyzer.concentrations)) < 2:
                        st.error("少なくとも2つの異なる濃度が必要です")
                    else:
                        with st.spinner("検量線作成中..."):
                            if calibration_type == "ピーク面積":
                                # ピーク面積による検量線
                                center_param = peak_center if use_peak_center else None
                                areas, fitting_results = analyzer.create_peak_area_calibration(
                                    analysis_start, analysis_end, center_param
                                )
                                
                                if len(areas) > 0:
                                    # 検量線作成
                                    valid_indices = areas > 0
                                    valid_areas = areas[valid_indices]
                                    valid_concentrations = np.array(analyzer.concentrations)[valid_indices]
                                    
                                    if len(valid_areas) >= 2:
                                        # 線形回帰
                                        coeffs = np.polyfit(valid_areas, valid_concentrations, 1)
                                        slope, intercept = coeffs
                                        
                                        # 統計指標
                                        y_pred = slope * valid_areas + intercept
                                        r2 = r2_score(valid_concentrations, y_pred)
                                        rmse = np.sqrt(mean_squared_error(valid_concentrations, y_pred))
                                        
                                        st.success("ピーク面積検量線を作成しました！")
                                        
                                        # 結果保存
                                        st.session_state.calibration_results = {
                                            'type': 'peak_area',
                                            'areas': areas,
                                            'concentrations': analyzer.concentrations,
                                            'slope': slope,
                                            'intercept': intercept,
                                            'r2': r2,
                                            'rmse': rmse,
                                            'fitting_results': fitting_results,
                                            'wave_range': [analysis_start, analysis_end]
                                        }
                                    
                            elif calibration_type == "多変量解析":
                                # 多変量解析による検量線
                                y_pred, cv_scores, X, x_range = analyzer.create_multivariate_calibration(
                                    analysis_start, analysis_end, n_components
                                )
                                
                                # 統計指標
                                r2 = r2_score(analyzer.concentrations, y_pred)
                                rmse = np.sqrt(mean_squared_error(analyzer.concentrations, y_pred))
                                cv_r2_mean = np.mean(cv_scores)
                                cv_r2_std = np.std(cv_scores)
                                
                                st.success("多変量解析検量線を作成しました！")
                                
                                # 結果保存
                                st.session_state.calibration_results = {
                                    'type': 'multivariate',
                                    'y_pred': y_pred,
                                    'concentrations': analyzer.concentrations,
                                    'r2': r2,
                                    'rmse': rmse,
                                    'cv_r2_mean': cv_r2_mean,
                                    'cv_r2_std': cv_r2_std,
                                    'n_components': n_components,
                                    'wave_range': [analysis_start, analysis_end],
                                    'model': analyzer.calibration_model,
                                    'X': X,
                                    'x_range': x_range
                                }
                
                # 結果表示
                if 'calibration_results' in st.session_state:
                    results = st.session_state.calibration_results
                    
                    st.subheader("検量線結果")
                    
                    # 統計指標表示
                    col_r2, col_rmse = st.columns(2)
                    with col_r2:
                        st.metric("R²", f"{results['r2']:.4f}")
                    with col_rmse:
                        st.metric("RMSE", f"{results['rmse']:.4f}")
                    
                    if results['type'] == 'multivariate':
                        st.metric("CV R² (平均±標準偏差)", f"{results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
                    
                    # 検量線プロット
                    fig_cal = go.Figure()
                    
                    if results['type'] == 'peak_area':
                        # ピーク面積 vs 濃度
                        areas = results['areas']
                        concentrations = results['concentrations']
                        
                        # 有効なデータポイント
                        valid_indices = areas > 0
                        valid_areas = areas[valid_indices]
                        valid_concentrations = concentrations[valid_indices]
                        
                        fig_cal.add_trace(go.Scatter(
                            x=valid_areas,
                            y=valid_concentrations,
                            mode='markers',
                            name='データポイント',
                            marker=dict(size=8, color='blue')
                        ))
                        
                        # 回帰直線
                        x_line = np.linspace(valid_areas.min(), valid_areas.max(), 100)
                        y_line = results['slope'] * x_line + results['intercept']
                        
                        fig_cal.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            name=f'y = {results["slope"]:.4f}x + {results["intercept"]:.4f}',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_cal.update_layout(
                            title='ピーク面積検量線',
                            xaxis_title='ピーク面積',
                            yaxis_title='濃度'
                        )
                    
                    elif results['type'] == 'multivariate':
                        # 多変量解析予測値 vs 実測値
                        fig_cal.add_trace(go.Scatter(
                            x=results['concentrations'],
                            y=results['y_pred'],
                            mode='markers',
                            name='データポイント',
                            marker=dict(size=8, color='blue')
                        ))
                        
                        # 理想直線 (y=x)
                        min_val = min(min(results['concentrations']), min(results['y_pred']))
                        max_val = max(max(results['concentrations']), max(results['y_pred']))
                        
                        fig_cal.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='理想直線 (y=x)',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_cal.update_layout(
                            title=f'多変量解析検量線 (成分数: {results["n_components"]})',
                            xaxis_title='実測値',
                            yaxis_title='予測値'
                        )
                    
                    fig_cal.update_layout(height=400)
                    st.plotly_chart(fig_cal, use_container_width=True)
                    
                    # フィッティング結果表示（ピーク面積の場合）
                    if results['type'] == 'peak_area' and 'fitting_results' in results:
                        with st.expander("フィッティング結果詳細"):
                            fitting_results = results['fitting_results']
                            
                            # フィッティングパラメータテーブル
                            fit_data = []
                            for i, fit_result in enumerate(fitting_results):
                                if fit_result is not None:
                                    fit_data.append({
                                        'ファイル名': fit_result['filename'],
                                        '濃度': analyzer.concentrations[i],
                                        '振幅': fit_result['amplitude'],
                                        '中心波数': fit_result['center'],
                                        '半値幅': fit_result['gamma'],
                                        'ベースライン': fit_result['baseline'],
                                        'ピーク面積': fit_result['area']
                                    })
                            
                            if fit_data:
                                fit_df = pd.DataFrame(fit_data)
                                st.dataframe(fit_df, use_container_width=True)
                                
                                # フィッティング結果プロット
                                fig_fit = make_subplots(
                                    rows=(len(fitting_results) + 2) // 3,
                                    cols=3,
                                    subplot_titles=[f.get('filename', f'Sample {i+1}') if f else f'Sample {i+1}' 
                                                  for i, f in enumerate(fitting_results)]
                                )
                                
                                for i, fit_result in enumerate(fitting_results):
                                    if fit_result is not None:
                                        row = i // 3 + 1
                                        col = i % 3 + 1
                                        
                                        # 元データ
                                        fig_fit.add_trace(
                                            go.Scatter(
                                                x=fit_result['x_range'],
                                                y=fit_result['y_range'],
                                                mode='lines',
                                                name='元データ',
                                                line=dict(color='blue', width=2),
                                                showlegend=(i == 0)
                                            ),
                                            row=row, col=col
                                        )
                                        
                                        # フィッティング結果
                                        fig_fit.add_trace(
                                            go.Scatter(
                                                x=fit_result['x_range'],
                                                y=fit_result['fitted_curve'],
                                                mode='lines',
                                                name='フィッティング',
                                                line=dict(color='red', dash='dash', width=2),
                                                showlegend=(i == 0)
                                            ),
                                            row=row, col=col
                                        )
                                
                                fig_fit.update_layout(
                                    title='ローレンツフィッティング結果',
                                    height=300 * ((len(fitting_results) + 2) // 3)
                                )
                                st.plotly_chart(fig_fit, use_container_width=True)
                    
                    # データエクスポート
                    st.subheader("結果エクスポート")
                    
                    if results['type'] == 'peak_area':
                        export_data = {
                            'ファイル名': [data['filename'] for data in analyzer.spectra_data],
                            '濃度': analyzer.concentrations,
                            'ピーク面積': results['areas']
                        }
                        
                        # フィッティングパラメータを追加
                        if 'fitting_results' in results:
                            for param in ['amplitude', 'center', 'gamma', 'baseline']:
                                export_data[param] = [
                                    fit.get(param, 0) if fit else 0 
                                    for fit in results['fitting_results']
                                ]
                    
                    elif results['type'] == 'multivariate':
                        export_data = {
                            'ファイル名': [data['filename'] for data in analyzer.spectra_data],
                            '実測値': results['concentrations'],
                            '多変量解析予測値': results['y_pred']
                        }
                    
                    export_df = pd.DataFrame(export_data)
                    
                    # 統計情報を追加
                    stats_info = [
                        f"検量線タイプ: {results['type']}",
                        f"解析波数範囲: {results['wave_range'][0]}-{results['wave_range'][1]} cm⁻¹",
                        f"R²: {results['r2']:.4f}",
                        f"RMSE: {results['rmse']:.4f}"
                    ]
                    
                    if results['type'] == 'multivariate':
                        stats_info.append(f"成分数: {results['n_components']}")
                        stats_info.append(f"CV R²: {results['cv_r2_mean']:.4f} ± {results['cv_r2_std']:.4f}")
                    elif results['type'] == 'peak_area':
                        stats_info.append(f"回帰式: y = {results['slope']:.4f}x + {results['intercept']:.4f}")
                    
                    # CSVファイルにコメントとして統計情報を追加
                    csv_buffer = io.StringIO()
                    csv_buffer.write("# 検量線解析結果\n")
                    for info in stats_info:
                        csv_buffer.write(f"# {info}\n")
                    csv_buffer.write("#\n")
                    export_df.to_csv(csv_buffer, index=False)
                    csv_content = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="結果をCSVでダウンロード",
                        data=csv_content,
                        file_name=f"calibration_results_{results['type']}.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    calibration_mode()
