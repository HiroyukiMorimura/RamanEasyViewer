import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io

# 共通ユーティリティ関数のインポート
from common_utils import (
    detect_file_type, read_csv_file, find_index, WhittakerSmooth, airPLS,
    remove_outliers_and_interpolate, process_spectrum_file
)

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

def process_uploaded_file(uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
    """
    アップロードされたファイルを処理してスペクトルデータを取得
    共通ユーティリティ関数を使用
    """
    try:
        # 共通ユーティリティ関数を使用
        wavenum, raw_spectrum, corrected_spectrum, average_spectrum, file_type, file_name = process_spectrum_file(
            uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
        )
        
        if wavenum is None:
            st.error(f"{file_name}のファイルタイプを判別できません。")
            return None, None, None
        
        return wavenum, raw_spectrum, average_spectrum
        
    except Exception as e:
        st.error(f"ファイル処理エラー: {str(e)}")
        return None, None, None

class RamanDeconvolution:
    def __init__(self):
        self.x = None
        self.y = None
        self.fitted_params = None
        self.fitted_y = None
        self.individual_peaks = None
        self.optimization_scores = None
        self.peak_constraints = None
        
    def lorentzian(self, x, amplitude, center, gamma):
        """ローレンツ関数"""
        return amplitude * gamma**2 / ((x - center)**2 + gamma**2)
    
    def multi_lorentzian(self, x, *params):
        """複数のローレンツ関数の和"""
        n_peaks = len(params) // 3
        y = np.zeros_like(x, dtype=np.float64)
        for i in range(n_peaks):
            amplitude = params[3*i]
            center = params[3*i + 1]
            gamma = params[3*i + 2]
            y += self.lorentzian(x, amplitude, center, gamma)
        return y
    
    def multi_lorentzian_constrained(self, x, *params):
        """制約付き複数のローレンツ関数の和"""
        if self.peak_constraints is None:
            return self.multi_lorentzian(x, *params)
        
        # パラメータを再構成（固定波数を考慮）
        full_params = []
        param_idx = 0
        
        for i in range(len(self.peak_constraints)):
            # 振幅
            amplitude = params[param_idx]
            param_idx += 1
            
            # 中心波数（固定または可変）
            if pd.isna(self.peak_constraints[i]['center']) or self.peak_constraints[i]['center'] == '' or self.peak_constraints[i]['center'] is None:
                center = params[param_idx]
                param_idx += 1
            else:
                center = self.peak_constraints[i]['center']
            
            # 半値幅
            gamma = params[param_idx]
            param_idx += 1
            
            full_params.extend([amplitude, center, gamma])
        
        return self.multi_lorentzian(x, *full_params)
    
    def estimate_initial_params(self, x, y, n_peaks):
        """初期パラメータの推定"""
        # ピークを見つける
        peaks, _ = find_peaks(y, height=np.max(y) * 0.1, distance=len(x) // (n_peaks * 2))
        
        if len(peaks) < n_peaks:
            # ピークが足りない場合はランダムに配置
            additional_peaks = np.random.choice(
                np.arange(len(x) // 4, 3 * len(x) // 4), 
                n_peaks - len(peaks), 
                replace=False
            )
            peaks = np.concatenate([peaks, additional_peaks])
        
        # 最も高いn_peaks個のピークを選択
        peak_heights = y[peaks]
        top_peaks_idx = np.argsort(peak_heights)[-n_peaks:]
        peaks = peaks[top_peaks_idx]
        
        params = []
        for peak_idx in peaks:
            amplitude = y[peak_idx]
            center = x[peak_idx]
            gamma = (x[-1] - x[0]) / (n_peaks * 20)  # 適当な幅
            params.extend([amplitude, center, gamma])
        
        return params
    
    def estimate_initial_params_constrained(self, x, y, peak_constraints):
        """制約付き初期パラメータの推定"""
        params = []
        n_peaks = len(peak_constraints)
        
        # データ型を確認して変換
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # 自動検出されたピークを取得
        try:
            distance = max(int(len(x) // max(n_peaks * 2, 10)), 1)
            peaks, _ = find_peaks(y, height=float(np.max(y) * 0.1), distance=distance)
            # ピークインデックスを整数リストに変換
            peaks = [int(p) for p in peaks if 0 <= int(p) < len(x)]
        except Exception as e:
            peaks = []
        
        # フォールバック：ピークが見つからない場合は等間隔に配置
        if len(peaks) == 0:
            start_idx = int(len(x) // 4)
            end_idx = int(3 * len(x) // 4)
            peaks = np.linspace(start_idx, end_idx, n_peaks, dtype=int).tolist()
        
        for i, constraint in enumerate(peak_constraints):
            # 振幅の初期値
            if pd.isna(constraint['center']) or constraint['center'] == '' or constraint['center'] is None:
                # 波数が指定されていない場合は自動検出
                if i < len(peaks):
                    peak_idx = int(peaks[i])
                    peak_idx = max(0, min(len(y) - 1, peak_idx))  # 範囲チェック
                    amplitude = max(float(y[peak_idx]), float(np.max(y) * 0.1))  # 最小振幅を保証
                else:
                    amplitude = float(np.max(y) * 0.5)
            else:
                # 波数が指定されている場合は、その位置での強度を使用
                try:
                    center_idx = find_index(x, float(constraint['center']))
                    center_idx = max(0, min(len(y) - 1, center_idx))
                    amplitude = max(float(y[center_idx]), float(np.max(y) * 0.1))  # 最小振幅を保証
                except Exception:
                    amplitude = float(np.max(y) * 0.5)
            
            params.append(float(amplitude))
            
            # 中心波数（固定されていない場合のみ）
            if pd.isna(constraint['center']) or constraint['center'] == '' or constraint['center'] is None:
                if i < len(peaks):
                    peak_idx = int(peaks[i])
                    peak_idx = max(0, min(len(x) - 1, peak_idx))
                    center = float(x[peak_idx])
                else:
                    # 等間隔で配置
                    center = float(x[0] + (x[-1] - x[0]) * (i + 1) / (n_peaks + 1))
                
                # 中心波数が範囲内にあることを確認
                center = max(float(x[0]), min(float(x[-1]), center))
                params.append(float(center))
            
            # 半値幅の初期値
            gamma = max(0.1, float(x[-1] - x[0]) / (n_peaks * 20))  # 最小半値幅を保証
            params.append(float(gamma))
        
        return [float(p) for p in params]
    
    def setup_bounds_constrained(self, x, y, peak_constraints):
        """制約付きパラメータの境界設定"""
        bounds_lower = []
        bounds_upper = []
        
        for constraint in peak_constraints:
            # 振幅の境界
            bounds_lower.append(0.001)  # 0より大きい最小値
            bounds_upper.append(np.max(y) * 10)  # 現実的な最大値
            
            # 中心波数の境界（固定されていない場合のみ）
            if pd.isna(constraint['center']) or constraint['center'] == '' or constraint['center'] is None:
                # 波数範囲にマージンを追加
                margin = (x[-1] - x[0]) * 0.05
                bounds_lower.append(x[0] - margin)
                bounds_upper.append(x[-1] + margin)
            
            # 半値幅の境界
            min_gamma = 0.1  # 最小半値幅
            max_gamma = min((x[-1] - x[0]) / 4, 100)  # 最大半値幅
            bounds_lower.append(min_gamma)
            bounds_upper.append(max_gamma)
        
        return bounds_lower, bounds_upper
    
    def fit_peaks(self, x, y, n_peaks, x_range=None, peak_constraints=None):
        """ピークフィッティング（制約付き対応）- 互換性のため残す"""
        return self.fit_peaks_robust(x, y, n_peaks, x_range, peak_constraints, n_trials=1)
    
    def fit_peaks_robust(self, x, y, n_peaks, x_range=None, peak_constraints=None, n_trials=5):
        """ロバストなピークフィッティング（複数回試行）"""
        if x_range is not None:
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            x_fit = x
            y_fit = y
        
        best_result = None
        best_rss = np.inf
        best_params = None
        best_pcov = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for trial in range(n_trials):
            status_text.text(f"試行 {trial + 1}/{n_trials}")
            progress_bar.progress((trial + 1) / n_trials)
            
            try:
                # 異なるシードで初期パラメータを生成
                np.random.seed(trial * 42)  # 再現可能なランダム性
                
                result_params, result_pcov = self.fit_peaks_single_trial(
                    x_fit, y_fit, x, y, n_peaks, peak_constraints, trial
                )
                
                if result_params is not None:
                    # 残差平方和を計算
                    if peak_constraints is not None:
                        y_pred = self.multi_lorentzian(x_fit, *result_params)
                    else:
                        y_pred = self.multi_lorentzian(x_fit, *result_params)
                    
                    rss = np.sum((y_fit - y_pred) ** 2)
                    
                    # 最良結果を更新
                    if rss < best_rss:
                        best_rss = rss
                        best_result = result_params
                        best_params = result_params
                        best_pcov = result_pcov
                        
            except Exception as e:
                st.write(f"試行 {trial + 1} でエラー: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if best_result is not None:
            # 最良結果を保存
            self.fitted_params = best_params
            self.fitted_y = self.multi_lorentzian(x, *best_params)
            
            # 個別のピークを計算
            self.individual_peaks = []
            for i in range(n_peaks):
                amplitude = best_params[3*i]
                center = best_params[3*i + 1]
                gamma = best_params[3*i + 2]
                peak_y = self.lorentzian(x, amplitude, center, gamma)
                self.individual_peaks.append(peak_y)
            
            st.success(f"最良結果（RSS: {best_rss:.6f}）を採用")
            return best_params, best_pcov
        else:
            st.error("すべての試行が失敗しました")
            return None, None
    
    def fit_peaks_single_trial(self, x_fit, y_fit, x_full, y_full, n_peaks, peak_constraints, trial_seed):
        """単一試行でのフィッティング"""
        self.peak_constraints = peak_constraints
        
        if peak_constraints is not None:
            # 制約付きフィッティング
            initial_params = self.estimate_initial_params_constrained_robust(
                x_fit, y_fit, peak_constraints, trial_seed
            )
            bounds_lower, bounds_upper = self.setup_bounds_constrained(x_fit, y_fit, peak_constraints)
            
            # 初期パラメータが境界内にあることを確認
            initial_params = np.array(initial_params)
            bounds_lower = np.array(bounds_lower)
            bounds_upper = np.array(bounds_upper)
            
            # 初期パラメータを境界内に調整
            initial_params = np.clip(initial_params, bounds_lower, bounds_upper)
            
            popt, pcov = curve_fit(
                self.multi_lorentzian_constrained,
                x_fit,
                y_fit,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=20000,
                ftol=1e-12,
                xtol=1e-12
            )
            
            # パラメータを完全な形に展開
            full_params = []
            param_idx = 0
            
            for i in range(len(peak_constraints)):
                # 振幅
                amplitude = popt[param_idx]
                param_idx += 1
                
                # 中心波数
                if pd.isna(peak_constraints[i]['center']) or peak_constraints[i]['center'] == '' or peak_constraints[i]['center'] is None:
                    center = popt[param_idx]
                    param_idx += 1
                else:
                    center = peak_constraints[i]['center']
                
                # 半値幅
                gamma = popt[param_idx]
                param_idx += 1
                
                full_params.extend([amplitude, center, gamma])
            
            return np.array(full_params), pcov
        
        else:
            # 通常のフィッティング
            initial_params = self.estimate_initial_params_robust(x_fit, y_fit, n_peaks, trial_seed)
            
            # パラメータの境界設定
            bounds_lower = []
            bounds_upper = []
            
            for i in range(n_peaks):
                bounds_lower.extend([0.001, x_fit[0], 0.001])
                bounds_upper.extend([np.max(y_fit) * 10, x_fit[-1], (x_fit[-1] - x_fit[0]) / 2])
            
            initial_params = np.clip(initial_params, bounds_lower, bounds_upper)
            
            popt, pcov = curve_fit(
                self.multi_lorentzian,
                x_fit,
                y_fit,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=20000,
                ftol=1e-12,
                xtol=1e-12
            )
            
            return popt, pcov
    
    def estimate_initial_params_robust(self, x, y, n_peaks, seed):
        """ロバストな初期パラメータ推定"""
        np.random.seed(seed)
        
        # データ型を確認して変換
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # 複数の方法でピーク検出を試行
        methods = [
            {'height': float(np.max(y) * 0.1), 'distance': int(len(x) // max(n_peaks * 2, 5))},
            {'height': float(np.max(y) * 0.05), 'distance': int(len(x) // max(n_peaks * 3, 5))},
            {'height': float(np.max(y) * 0.2), 'distance': int(len(x) // max(n_peaks * 1.5, 5))},
        ]
        
        all_peaks = []
        for method in methods:
            try:
                peaks, _ = find_peaks(y, height=method['height'], distance=method['distance'])
                # ピークインデックスを整数に変換
                peaks = [int(p) for p in peaks if 0 <= int(p) < len(x)]
                all_peaks.extend(peaks)
            except Exception as e:
                continue
        
        # 重複除去とソート
        all_peaks = sorted(list(set(all_peaks)))
        
        if len(all_peaks) == 0:
            # フォールバック：等間隔配置
            start_idx = int(len(x) // 4)
            end_idx = int(3 * len(x) // 4)
            all_peaks = np.linspace(start_idx, end_idx, n_peaks * 2, dtype=int).tolist()
        
        # 最適なピーク選択
        if len(all_peaks) >= n_peaks:
            # 強度順でソート
            peak_heights = [float(y[p]) for p in all_peaks if 0 <= p < len(y)]
            if len(peak_heights) >= n_peaks:
                sorted_indices = np.argsort(peak_heights)[-n_peaks:]
                selected_peaks = [all_peaks[i] for i in sorted_indices]
            else:
                selected_peaks = all_peaks[:n_peaks]
        else:
            # ピークが足りない場合は補完
            selected_peaks = all_peaks.copy()
            while len(selected_peaks) < n_peaks:
                # ランダムだが再現可能な位置を追加
                additional_pos = int(len(x) * 0.2 + (len(x) * 0.6) * np.random.random())
                additional_pos = max(0, min(len(x) - 1, additional_pos))
                if additional_pos not in selected_peaks:
                    selected_peaks.append(additional_pos)
        
        # パラメータ生成
        params = []
        for peak_idx in selected_peaks:
            peak_idx = int(peak_idx)  # 確実に整数に変換
            peak_idx = max(0, min(len(y) - 1, peak_idx))  # 範囲チェック
            
            amplitude = max(float(y[peak_idx]), float(np.max(y) * 0.1))
            center = float(x[peak_idx])
            gamma = max(0.1, float(x[-1] - x[0]) / (n_peaks * 20))
            
            # 小さなランダム摂動を追加（再現可能）
            amplitude *= (0.8 + 0.4 * np.random.random())
            gamma *= (0.5 + 1.0 * np.random.random())
            
            params.extend([float(amplitude), float(center), float(gamma)])
        
        return [float(p) for p in params]
    
    def estimate_initial_params_constrained_robust(self, x, y, peak_constraints, seed):
        """ロバストな制約付き初期パラメータ推定"""
        np.random.seed(seed)
        
        # データ型を確認して変換
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # 通常の初期パラメータ推定を使用
        try:
            normal_params = self.estimate_initial_params_robust(x, y, len(peak_constraints), seed)
        except Exception:
            # フォールバック
            normal_params = []
        
        params = []
        param_idx = 0
        
        for i, constraint in enumerate(peak_constraints):
            # 振幅
            if param_idx < len(normal_params):
                amplitude = float(normal_params[param_idx])
            else:
                amplitude = float(np.max(y) * (0.3 + 0.7 * np.random.random()))
            
            params.append(float(amplitude))
            param_idx += 3  # 次のピークの振幅へ
            
            # 中心波数（固定されていない場合のみ）
            if pd.isna(constraint['center']) or constraint['center'] == '' or constraint['center'] is None:
                if param_idx - 2 < len(normal_params):
                    center = float(normal_params[param_idx - 2])
                else:
                    center = float(x[0] + (x[-1] - x[0]) * (i + 1) / (len(peak_constraints) + 1))
                
                center = max(float(x[0]), min(float(x[-1]), center))
                params.append(float(center))
            
            # 半値幅
            if param_idx - 1 < len(normal_params):
                gamma = float(normal_params[param_idx - 1])
            else:
                gamma = max(0.1, float(x[-1] - x[0]) / (len(peak_constraints) * 20))
            
            # ランダム摂動
            gamma *= (0.5 + 1.0 * np.random.random())
            params.append(float(gamma))
        
        return [float(p) for p in params]
        
    def calculate_optimization_score(self, x, y, n_peaks, x_range=None, peak_constraints=None):
        """統計的基準の計算（制約付き対応）"""
        if x_range is not None:
            mask = (x >= x_range[0]) & (x <= x_range[1])
            x_fit = x[mask]
            y_fit = y[mask]
        else:
            x_fit = x
            y_fit = y
        
        # 統計的基準計算では1回のフィッティングで十分
        # 波数固定制約がある場合は、それを維持してフィッティング実行
        popt, pcov = self.fit_peaks_robust(x, y, n_peaks, x_range, peak_constraints, n_trials=1)
        
        if popt is None:
            return np.inf
        
        # 残差平方和を計算
        y_pred = self.multi_lorentzian(x_fit, *popt)
        rss = np.sum((y_fit - y_pred) ** 2)
        
        # 統計的基準を計算
        n_data = len(x_fit)
        
        # パラメータ数の計算（制約を考慮）
        # 固定された波数はパラメータ数にカウントされない
        if peak_constraints is not None:
            n_params = 0
            for constraint in peak_constraints:
                n_params += 1  # 振幅（常にパラメータ）
                if pd.isna(constraint['center']) or constraint['center'] == '' or constraint['center'] is None:
                    n_params += 1  # 中心波数（可変の場合のみパラメータ）
                # 固定波数の場合は中心波数はパラメータにカウントしない
                n_params += 1  # 半値幅（常にパラメータ）
        else:
            n_params = 3 * n_peaks
        
        score = n_data * np.log(rss / n_data) + n_params * np.log(n_data)
        
        return score
    
    def find_optimal_peaks(self, x, y, max_peaks=10, x_range=None, peak_constraints=None):
        """最適なピーク数を統計的基準で判定（制約付き対応）"""
        optimization_scores = []
        
        # 制約がある場合の警告
        if peak_constraints is not None:
            has_fixed_peaks = any(
                not (pd.isna(c['center']) or c['center'] == '' or c['center'] is None) 
                for c in peak_constraints
            )
            if has_fixed_peaks:
                st.warning("⚠️ 波数固定制約がある状態で最適化を実行します。固定された波数はピーク数に関わらず維持されます。")
        
        for n_peaks in range(1, max_peaks + 1):
            # 制約がある場合は、現在のピーク数に対応する制約を適用
            current_constraints = None
            if peak_constraints is not None and n_peaks <= len(peak_constraints):
                current_constraints = peak_constraints[:n_peaks]  # 最初のn_peaks個の制約を使用
            
            score = self.calculate_optimization_score(x, y, n_peaks, x_range, current_constraints)
            optimization_scores.append(score)
        
        self.optimization_scores = optimization_scores
        optimal_n_peaks = np.argmin(optimization_scores) + 1
        
        return optimal_n_peaks, optimization_scores

def create_peak_constraints_table(n_peaks):
    """ピーク制約テーブルの作成"""
    if f"peak_constraints_{n_peaks}" not in st.session_state:
        st.session_state[f"peak_constraints_{n_peaks}"] = pd.DataFrame({
            'ピーク番号': [f"ピーク{i+1}" for i in range(n_peaks)],
            '固定波数 (cm⁻¹)': [None] * n_peaks,
            '備考': [''] * n_peaks
        })
    
    return st.session_state[f"peak_constraints_{n_peaks}"]

def process_peak_constraints(constraints_df):
    """ピーク制約の処理"""
    constraints = []
    for _, row in constraints_df.iterrows():
        center = row['固定波数 (cm⁻¹)']
        if pd.isna(center) or center == '' or center == 0:
            center = None
        else:
            try:
                center = float(center)
            except (ValueError, TypeError):
                center = None
        
        constraints.append({
            'center': center,
            'comment': row['備考']
        })
    
    return constraints

def peak_deconvolution_mode():
    st.header("ピーク分離機能")
    
    deconv = RamanDeconvolution()
    
    # 実データ設定
    st.sidebar.subheader("データ処理設定")
    
    # データ処理パラメータ
    start_wavenum = st.sidebar.number_input("波数（開始）:", value=400, min_value=0, max_value=4000)
    end_wavenum = st.sidebar.number_input("波数（終了）:", value=2000, min_value=start_wavenum+1, max_value=4000)
    dssn_th = st.sidebar.number_input("ベースラインパラメーター:", value=1000, min_value=1, max_value=10000) / 1e7
    savgol_wsize = st.sidebar.number_input("ウィンドウサイズ:", value=5, min_value=1, max_value=101, step=2)
    
    uploaded_file = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（単数）",
        type=['csv', 'txt'],
        accept_multiple_files=False,
        key="mv_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # ファイル処理
            wavenum, raw_spectrum, corrected_spectrum = process_uploaded_file(
                uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
            )
            
            if wavenum is not None:
                x = wavenum
                y = corrected_spectrum
                
                # データの確認
                # st.success(f"データを正常に読み込みました: {len(x)} データポイント")
                
                # 処理前後のスペクトル比較
                with st.expander("処理前後のスペクトル比較"):
                    fig_comparison = go.Figure()
                    
                    fig_comparison.add_trace(go.Scatter(
                        x=x, y=raw_spectrum,
                        mode='lines',
                        name='元データ',
                        line=dict(color='blue', width=1)
                    ))
                    
                    fig_comparison.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name='ベースライン補正後',
                        line=dict(color='red', width=1)
                    ))
                    
                    fig_comparison.update_layout(
                        title='スペクトル処理前後の比較',
                        xaxis_title='ラマンシフト (cm⁻¹)',
                        yaxis_title='強度',
                        height=400
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # フィッティング範囲の設定
                st.sidebar.subheader("フィッティング範囲設定")
                
                range_min = st.sidebar.number_input("範囲最小値:", value=int(x.min()), min_value=int(x.min()), max_value=int(x.max()))
                range_max = st.sidebar.number_input("範囲最大値:", value=int(x.max()), min_value=range_min, max_value=int(x.max()))
                peak_range = [range_min, range_max]
                
                # レイアウトを2列に分割
                col1, col2 = st.columns([2, 1])
                
                with col2:
                    # ピーク数最適化
                    st.subheader("ピーク数最適化")
                    
                    if st.button("ピーク数最適化"):
                        with st.spinner("最適化計算中..."):
                            optimal_n_peaks, optimization_scores = deconv.find_optimal_peaks(
                                x, y,
                                max_peaks=6,
                                x_range=peak_range,
                                peak_constraints=None
                            )
                            st.success(f"最適ピーク数: {optimal_n_peaks}")
                            
                            # 最適化スコアをプロット
                            fig_opt = go.Figure()
                            fig_opt.add_trace(go.Scatter(
                                x=list(range(1, len(optimization_scores) + 1)),
                                y=optimization_scores,
                                mode='lines+markers',
                                name='最適化スコア'
                            ))
                            fig_opt.update_layout(
                                title='最適化スコア vs ピーク数',
                                xaxis_title='ピーク数',
                                yaxis_title='最適化スコア',
                                height=300
                            )
                            st.plotly_chart(fig_opt, use_container_width=True)
                            
                            # セッション状態に保存
                            st.session_state.optimal_n_peaks = optimal_n_peaks
                    
                    # ピーク制約テーブル
                    st.subheader("ピークフィッティング")
                    
                    # ピーク数選択
                    if 'optimal_n_peaks' in st.session_state:
                        default_peaks = st.session_state.optimal_n_peaks
                    else:
                        default_peaks = 2
                        
                    n_peaks = st.number_input("ピーク数:", min_value=1, max_value=10, value=default_peaks)
                    
                    st.write("**波数固定設定**: 固定したい波数を入力してください（空欄の場合は自動最適化）")
                    
                    # 制約テーブルの作成と表示
                    constraints_df = create_peak_constraints_table(n_peaks)
                    
                    # データエディタで制約を編集
                    edited_constraints = st.data_editor(
                        constraints_df,
                        use_container_width=True,
                        num_rows="fixed",
                        column_config={
                            "ピーク番号": st.column_config.TextColumn(disabled=True),
                            "固定波数 (cm⁻¹)": st.column_config.NumberColumn(
                                "固定波数 (cm⁻¹)",
                                help="固定したい波数を入力。空欄の場合は自動最適化",
                                min_value=float(x.min()),
                                max_value=float(x.max()),
                                step=0.1,
                                format="%.1f"
                            ),
                            "備考": st.column_config.TextColumn(
                                "備考",
                                help="ピークの説明や化合物名など"
                            )
                        },
                        key=f"constraints_editor_{n_peaks}"
                    )
                    
                    # 制約の更新
                    st.session_state[f"peak_constraints_{n_peaks}"] = edited_constraints
                    
                    # 制約の処理
                    peak_constraints = process_peak_constraints(edited_constraints)
                    
                    # 制約の確認表示
                    with st.expander("制約確認"):
                        fixed_count = 0
                        free_count = 0
                        for i, constraint in enumerate(peak_constraints):
                            if constraint['center'] is not None:
                                st.write(f"ピーク{i+1}: 波数 {constraint['center']:.1f} cm⁻¹ で固定")
                                fixed_count += 1
                            else:
                                st.write(f"ピーク{i+1}: 波数は自動最適化")
                                free_count += 1
                        
                        st.write(f"**合計**: 固定ピーク {fixed_count}個, 自動最適化ピーク {free_count}個")
                        
                        # 想定パラメータ数の表示
                        expected_params = n_peaks * 2 + free_count  # 振幅×n + 半値幅×n + 中心波数×free_count
                        st.write(f"**想定パラメータ数**: {expected_params}")
                    
                    # フィッティング実行
                    if st.button("フィッティング実行"):
                        with st.spinner("フィッティング中..."):
                            # 試行回数の選択
                            n_trials = st.sidebar.selectbox("フィッティング試行回数:", [1, 3, 5, 10], index=2, 
                                                           help="回数が多いほど安定した結果が得られますが、時間がかかります")
                            
                            popt, pcov = deconv.fit_peaks_robust(x, y, n_peaks, peak_range, peak_constraints, n_trials)
                            
                            if popt is not None:
                                st.success("フィッティング完了!")
                                
                                # パラメータ表示
                                st.subheader("フィッティングパラメータ")
                                params_df = pd.DataFrame(
                                    np.array(popt).reshape(-1, 3),
                                    columns=['振幅', '中心', '半値幅'],
                                    index=[f'ピーク{i+1}' for i in range(n_peaks)]
                                )
                                
                                # 制約状態の表示
                                constraint_status = []
                                for i, constraint in enumerate(peak_constraints):
                                    if constraint['center'] is not None:
                                        constraint_status.append("固定")
                                    else:
                                        constraint_status.append("最適化")
                                
                                params_df['波数状態'] = constraint_status
                                params_df['備考'] = [c['comment'] for c in peak_constraints]
                                
                                st.dataframe(params_df)
                                
                                # フィッティング品質
                                if peak_range:
                                    mask = (x >= peak_range[0]) & (x <= peak_range[1])
                                    x_eval = x[mask]
                                    y_eval = y[mask]
                                else:
                                    x_eval = x
                                    y_eval = y
                                
                                y_pred = deconv.multi_lorentzian(x_eval, *popt)
                                r_squared = 1 - np.sum((y_eval - y_pred) ** 2) / np.sum((y_eval - np.mean(y_eval)) ** 2)
                                st.metric("R²", f"{r_squared:.4f}")
                
                with col1:
                    st.subheader("データとフィッティング結果")
                    
                    # プロット
                    fig = make_subplots(rows=1, cols=1)
                    
                    # 元データ
                    fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        name='元データ',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # フィッティング結果があれば表示
                    if deconv.fitted_y is not None:
                        fig.add_trace(go.Scatter(
                            x=x, y=deconv.fitted_y,
                            mode='lines',
                            name='フィッティング結果',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # 個別のピーク
                        if deconv.individual_peaks:
                            colors = px.colors.qualitative.Set2
                            for i, peak_y in enumerate(deconv.individual_peaks):
                                # 制約状態の表示
                                constraint_info = ""
                                if hasattr(deconv, 'peak_constraints') and deconv.peak_constraints and i < len(deconv.peak_constraints):
                                    if deconv.peak_constraints[i]['center'] is not None:
                                        constraint_info = f" (固定: {deconv.peak_constraints[i]['center']:.1f})"
                                
                                fig.add_trace(go.Scatter(
                                    x=x, y=peak_y,
                                    mode='lines',
                                    name=f'ピーク{i+1}{constraint_info}',
                                    line=dict(color=colors[i % len(colors)], width=2)
                                ))
                    
                    # 制約を視覚化（固定波数の位置に縦線）
                    if hasattr(deconv, 'peak_constraints') and deconv.peak_constraints:
                        for i, constraint in enumerate(deconv.peak_constraints):
                            if constraint['center'] is not None:
                                fig.add_vline(
                                    x=constraint['center'], 
                                    line_dash="dot", 
                                    line_color="red", 
                                    annotation_text=f"固定{i+1}",
                                    annotation_position="top"
                                )
                    
                    # ピーク範囲の表示
                    if peak_range:
                        fig.add_vline(x=peak_range[0], line_dash="dash", line_color="gray", annotation_text="範囲開始")
                        fig.add_vline(x=peak_range[1], line_dash="dash", line_color="gray", annotation_text="範囲終了")
                    
                    fig.update_layout(
                        title='ラマンピーク分離',
                        xaxis_title='ラマンシフト (cm⁻¹)',
                        yaxis_title='強度',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 残差プロット
                    if deconv.fitted_y is not None:
                        residuals = y - deconv.fitted_y
                        
                        fig_residuals = go.Figure()
                        fig_residuals.add_trace(go.Scatter(
                            x=x, y=residuals,
                            mode='lines',
                            name='残差',
                            line=dict(color='green', width=1)
                        ))
                        fig_residuals.add_hline(y=0, line_dash="dash", line_color="black")
                        fig_residuals.update_layout(
                            title='残差',
                            xaxis_title='ラマンシフト (cm⁻¹)',
                            yaxis_title='残差',
                            height=300
                        )
                        st.plotly_chart(fig_residuals, use_container_width=True)
                        
                        # データエクスポート
                        st.subheader("データエクスポート")
                        
                        # 結果をDataFrameに変換
                        export_df = pd.DataFrame({
                            'x': x,
                            'original_y': y,
                            'fitted_y': deconv.fitted_y,
                            'residuals': residuals
                        })
                        
                        # 個別のピークを追加
                        if deconv.individual_peaks:
                            for i, peak_y in enumerate(deconv.individual_peaks):
                                export_df[f'peak_{i+1}'] = peak_y
                        
                        # パラメータ情報を追加
                        if deconv.fitted_params is not None:
                            param_info = []
                            for i in range(n_peaks):
                                amplitude = deconv.fitted_params[3*i]
                                center = deconv.fitted_params[3*i + 1]
                                gamma = deconv.fitted_params[3*i + 2]
                                constraint_status = "固定" if hasattr(deconv, 'peak_constraints') and deconv.peak_constraints and i < len(deconv.peak_constraints) and deconv.peak_constraints[i]['center'] is not None else "最適化"
                                param_info.append(f"ピーク{i+1}: 振幅={amplitude:.4f}, 中心={center:.2f}, 半値幅={gamma:.2f} ({constraint_status})")
                            
                            # パラメータ情報をコメントとして追加
                            export_df.attrs['parameters'] = param_info
                        
                        csv = export_df.to_csv(index=False)
                        st.download_button(
                            label="結果をCSVでダウンロード",
                            data=csv,
                            file_name="raman_peak_separation_results.csv",
                            mime="text/csv"
                        )
                    
            else:
                st.error("データの処理に失敗しました。")
                return
                
        except Exception as e:
            st.error(f"ファイル処理エラー: {str(e)}")
            return
