# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 15:56:04 2025

@author: hiroy
"""

import numpy as np
import pandas as pd
import streamlit as st

import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import NMF, PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def create_features_labels(spectra, window_size=10):
    # 特徴量とラベルの配列を初期化
    X = []
    y = []
    # スペクトルデータの長さ
    n_points = len(spectra)
    # 人手によるピークラベル、または自動生成コードをここに配置
    peak_labels = np.zeros(n_points)

    # 特徴量とラベルの抽出
    for i in range(window_size, n_points - window_size):
        # 前後の窓サイズのデータを特徴量として使用
        features = spectra[i-window_size:i+window_size+1]
        X.append(features)
        y.append(peak_labels[i])

    return np.array(X), np.array(y)

def find_index(rs_array,  rs_focused):
    '''
    Convert the index of the proximate wavenumber by finding the absolute 
    minimum value of (rs_array - rs_focused)
    
    input
        rs_array: Raman wavenumber
        rs_focused: Index
    output
        index
    '''

    diff = [abs(element - rs_focused) for element in rs_array]
    index = np.argmin(diff)
    return index

def WhittakerSmooth(x,w,lambda_,differences=1):
    '''
    Penalized least squares algorithm for background fitting
    
    input
        x: input data (i.e. chromatogram of spectrum)
        w: binary masks (value of the mask is zero if a point belongs to peaks and one otherwise)
        lambda_: parameter that can be adjusted by user. The larger lambda is,  the smoother the resulting background
        differences: integer indicating the order of the difference of penalties
    
    output
        the fitted background vector
    '''
    X=np.array(x, dtype=np.float64)
    m=X.size
    E=eye(m,format='csc')
    for i in range(differences):
        E=E[1:]-E[:-1] # numpy.diff() does not work with sparse matrix. This is a workaround.
    W=diags(w,0,shape=(m,m))
    A=csc_matrix(W+(lambda_*E.T*E))
    B=csc_matrix(W*X.T).toarray().flatten()
    background=spsolve(A,B)
    return np.array(background)

def airPLS(x, dssn_th, lambda_, porder, itermax):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram or spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is, the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
    # マイナス値がある場合の処理
    min_value = np.min(x)
    print(np.min(x))
    offset = 0
    if min_value < 0:
        offset = abs(min_value) + 1  # 最小値を1にするためのオフセット
        x = x + offset  # 全体をシフト
    print(np.min(x))
    m = x.shape[0]
    w = np.ones(m, dtype=np.float64)  # 明示的に型を指定
    x = np.asarray(x, dtype=np.float64)  # xも明示的に型を指定
    
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        
        # dssn がゼロまたは非常に小さい場合を回避
        if dssn < 1e-10:
            dssn = 1e-10
        
        # 収束判定
        if (dssn < dssn_th * (np.abs(x)).sum()) or (i == itermax):
            if i == itermax:
                print('WARNING: max iteration reached!')
            break
        
        # 重みの更新
        w[d >= 0] = 0  # d > 0 はピークの一部として重みを無視
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        
        # 境界条件の調整
        if d[d < 0].size > 0:
            w[0] = np.exp(i * np.abs(d[d < 0]).max() / dssn)
        else:
            w[0] = 1.0  # 適切な初期値
        
        w[-1] = w[0]

    return z

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.asmatrix([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def find_peak_width(spectra, first_dev, peak_position, window_size=20):
    """
    Find the peak start/end close the peak position
    Parameters:
    spectra (ndarray): Original spectrum 
    first_dev (ndarray): First derivative of the spectrum 
    peak_position (int): Peak index 
    window_size (int): Window size to find the start/end of the peak 

    Returns:
    local_start_idx/local_end_idx: Start and end of the peaks 
    """

    start_idx = max(peak_position - window_size, 0)
    end_idx   = min(peak_position + window_size, len(first_dev) - 1)
    
    local_start_idx = np.argmax(first_dev[start_idx:end_idx+1]) + start_idx
    local_end_idx   = np.argmin(first_dev[start_idx:end_idx+1]) + start_idx
        
    return local_start_idx, local_end_idx

def find_peak_area(spectra, local_start_idx, local_end_idx):
    """
    Calculate the area of the peaks 

    Parameters:
    spectra (ndarray): Original spectrum 
    local_start_idx (int): Output of the find_peak_width
    local_end_idx (int): Output of the find_peak_width
    
    Returns:
    peak_area (float): Area of the peaks 
    """    
    
    peak_area = np.trapz(spectra[local_start_idx:local_end_idx+1], dx=1)
    
    return peak_area

def detect_file_type(data):
    """
    Determine the structure of the input data.
    """
    try:
        if data.columns[0].split(':')[0] == "# Laser Wavelength":
            return "ramaneye_new"
        elif data.columns[0] == "WaveNumber":
            return "ramaneye_old"
        elif data.columns[0] == "Pixels":
            return "eagle"
        elif data.columns[0] == "ENLIGHTEN Version":
            return "wasatch"
        return "unknown"
    except:
        return "unknown"

def read_csv_file(uploaded_file, file_extension):
    """
    Read a CSV or TXT file into a DataFrame based on file extension.
    """
    try:
        uploaded_file.seek(0)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', header=0, index_col=None, on_bad_lines='skip')
        return data
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

def remove_outliers_and_interpolate(spectrum, window_size=10, threshold_factor=3):
    """
    スペクトルからスパイク（外れ値）を検出し、補完する関数
    スパイクは、ウィンドウ内の標準偏差が一定の閾値を超える場合に検出される
    
    input:
        spectrum: numpy array, ラマンスペクトル
        window_size: ウィンドウのサイズ（デフォルトは20）
        threshold_factor: 標準偏差の閾値（デフォルトは5倍）
    
    output:
        cleaned_spectrum: numpy array, スパイクを取り除き補完したスペクトル
    """
    spectrum_len = len(spectrum)
    cleaned_spectrum = spectrum.copy()
    
    for i in range(spectrum_len):
        # 端点では、ウィンドウサイズが足りないので、ウィンドウを調整
        left_idx = max(i - window_size, 0)
        right_idx = min(i + window_size + 1, spectrum_len)
        
        # ウィンドウ内のデータを取得
        window = spectrum[left_idx:right_idx]
        
        # ウィンドウ内の中央値と標準偏差を計算
        window_median = np.median(window)
        window_std = np.std(window)
        
        # ウィンドウ内の値が標準偏差の閾値を超えるスパイクを検出
        if abs(spectrum[i] - window_median) > threshold_factor * window_std:
            # スパイクが見つかった場合、その値を両隣の中央値で補完
            if i > 0 and i < spectrum_len - 1:  # 両隣の値が存在する場合
                cleaned_spectrum[i] = (spectrum[i - 1] + spectrum[i + 1]) / 2
            elif i == 0:  # 左端の場合
                cleaned_spectrum[i] = spectrum[i + 1]
            elif i == spectrum_len - 1:  # 右端の場合
                cleaned_spectrum[i] = spectrum[i - 1] 
    return cleaned_spectrum

# Functions for multivariate analysis
def load_and_process_data_for_multivariate(uploaded_files, start_wavenum, end_wavenum, lambda_param, porder, savgol_wsize=5):
    """
    Load and process uploaded CSV files with support for multiple file types for multivariate analysis
    """
    grouped_data = {}
    all_baseline_spectra = []
    all_labels = []
    wavenumber = None
    dssn_th = 0.00001
    
    for uploaded_file in uploaded_files:
        try:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1].lower()
            
            st.write(f"Processing: {file_name}")
            
            # Read and detect file type
            data = read_csv_file(uploaded_file, file_extension)
            file_type = detect_file_type(data)
            uploaded_file.seek(0)
            
            if file_type == "unknown":
                st.error(f"{file_name}のファイルタイプを判別できません。")
                continue
            
            # Process each file type
            if file_type == "wasatch":
                st.write(f"ファイルタイプ: Wasatch ENLIGHTEN - {file_name}")
                lambda_ex = 785
                data = pd.read_csv(uploaded_file, skiprows=46)                
                pre_wavelength = np.array(data["Wavelength"].values)
                pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
                
                # Get number of available spectra
                number_of_rows = data.shape[1] - 3
                
                if number_of_rows > 0:
                    # Use the last available spectrum
                    number_line = number_of_rows - 1
                    if number_line == 0:
                        pre_spectrum = np.array(data["Processed"].values)
                    else:
                        pre_spectrum = np.array(data[f"Processed.{number_line}"].values)
                else:
                    pre_spectrum = np.array(data["Processed"].values)
                
            elif file_type == "ramaneye_old":
                st.write(f"ファイルタイプ: RamanEye Data(Old) - {file_name}")
                # number_of_rows = data.shape[1]
                
                df_transposed = data.set_index("WaveNumber").T

                # 列名（"acrylic board" など）を汎用化
                df_transposed.columns = ["intensity"]
                
                # 波数をfloatに変換し、インデックスに設定
                df_transposed.index = df_transposed.index.astype(float)
                df_transposed = df_transposed.sort_index()
                
                # 波数と強度をNumPy配列として取得
                pre_wavenum = df_transposed.index.to_numpy()
                pre_spectra = df_transposed["intensity"].to_numpy()
                
                if pre_wavenum[0] > pre_wavenum[1]:
                    # pre_wavenum と pre_spectra を反転
                    pre_wavenum = pre_wavenum[::-1]
                    pre_spectra = pre_spectra[::-1]
                        
            elif file_type == "ramaneye_new":
                st.write(f"ファイルタイプ: RamanEye Data(New) - {file_name}")
                data = pd.read_csv(uploaded_file, skiprows=9)
                number_of_rows = data.shape[1]
                
                # Use the last available column
                number_line = number_of_rows - 2
                pre_wavenum = data["WaveNumber"]
                pre_spectrum = np.array(data.iloc[:, number_line + 1])
                
                if pre_wavenum.iloc[0] >= pre_wavenum.iloc[1]:
                    # Reverse pre_wavenum and pre_spectrum
                    pre_wavenum = pre_wavenum[::-1]
                    pre_spectrum = pre_spectrum[::-1]
                    
            elif file_type == "eagle":
                st.write(f"ファイルタイプ: Eagle Data - {file_name}")
                data_transposed = data.transpose()
                header = data_transposed.iloc[:3]  # 最初の3行
                reversed_data = data_transposed.iloc[3:].iloc[::-1]
                data_transposed = pd.concat([header, reversed_data], ignore_index=True)
                pre_wavenum = np.array(data_transposed.iloc[3:, 0])
                pre_spectra = np.array(data_transposed.iloc[3:, 1])
            
            # Convert to numpy arrays if needed
            if isinstance(pre_wavenum, pd.Series):
                pre_wavenum = pre_wavenum.values
            if isinstance(pre_spectrum, pd.Series):
                pre_spectrum = pre_spectrum.values
            
            # Find indices for wavenumber range
            start_index = find_index(pre_wavenum, start_wavenum)
            end_index = find_index(pre_wavenum, end_wavenum)

            wavenum = np.array(pre_wavenum[start_index:end_index+1])
            spectrum = np.array(pre_spectrum[start_index:end_index+1])

            # Baseline and spike removal 
            spectrum_spikerm = remove_outliers_and_interpolate(spectrum)
            
            # Apply median filter
            mveAve_spectrum = signal.medfilt(spectrum_spikerm, savgol_wsize)
            
            # Baseline correction
            baseline = airPLS(mveAve_spectrum, dssn_th, lambda_param, porder, 30)
            BSremoval_spectrum = spectrum_spikerm - baseline
            BSremoval_spectrum_pos = BSremoval_spectrum + abs(np.minimum(spectrum_spikerm, 0))  # Correct negative values
        
            # Use the baseline corrected spectrum
            corrected_spectrum = BSremoval_spectrum_pos
            
            if wavenumber is None:
                wavenumber = wavenum
            
            # Extract group name from filename
            parts = file_name.split('_')
            if len(parts) >= 2:
                group_name = parts[1]  # Use second part as group name
            else:
                group_name = file_name.split('.')[0]  # Use filename without extension
            
            if group_name not in grouped_data:
                grouped_data[group_name] = []
            
            # Store processed data
            processed_data = np.column_stack([
                wavenum,
                spectrum,
                corrected_spectrum
            ])
            grouped_data[group_name].append(processed_data)
            
            # Add to combined data
            all_baseline_spectra.append(corrected_spectrum)
            all_labels.append(group_name)
            
            st.success(f"Successfully processed {file_name} - Group: {group_name}, Data points: {len(wavenum)}")
            
        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {e}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
    
    return grouped_data, np.array(all_baseline_spectra), all_labels, wavenumber

def plot_spectra_matplotlib(grouped_data, title="Baseline Corrected Spectra"):
    """
    Plot spectra using matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for i, (group, spectra_list) in enumerate(grouped_data.items()):
        color = colors[i % len(colors)]
        
        for j, spectrum in enumerate(spectra_list):
            label = group if j == 0 else None  # Only show legend for first spectrum of each group
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
    # Ensure non-negative values for NMF
    all_spectra[all_spectra < 0] = 0
    
    # Apply NMF
    nmf_model = NMF(n_components=n_components, init='random', random_state=42, max_iter=1000)
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

def spectrum_analysis_mode():
    """
    Spectrum analysis mode (original functionality)
    """
    st.header("📊 ラマンスペクトル解析")
    
    # パラメータ設定
    savgol_wsize         = 5    # Savitzky-Golayフィルタのウィンドウサイズ
    savgol_order         = 3    # Savitzky-Golayフィルタの次数
    pre_start_wavenum    = 400  # 波数の開始
    pre_end_wavenum      = 2000 # 波数の終了
    Fsize                = 14   # フォントサイズ
    
    # 波数範囲の設定
    start_wavenum = st.sidebar.number_input("波数（開始）を入力してください:", min_value=-200, max_value=4800, value=pre_start_wavenum, step=100)
    end_wavenum = st.sidebar.number_input("波数（終了）を入力してください:", min_value=-200, max_value=4800, value=pre_end_wavenum, step=100)

    dssn_th = st.sidebar.number_input("ベースラインパラメーターを入力してください:", min_value=1, max_value=10000, value=1000, step=1)
    dssn_th = dssn_th/10000000
    
    # 微分の平滑化パラメータ
    num_firstDev = st.sidebar.number_input(
        "1次微分の平滑化の数値を入力してください:",
        min_value=1,
        max_value=35,
        value=13,
        step=2,
        key='unique_number_firstDev_key'
    )

    num_secondDev = st.sidebar.number_input(
        "2次微分の平滑化の数値を入力してください:",
        min_value=1,
        max_value=35,
        value=5,
        step=2,
        key='unique_number_secondDev_key'
    )
    
    num_threshold = st.sidebar.number_input(
        "閾値を入力してください:",
        min_value=1,
        max_value=1000,
        value=10,
        step=10,
        key='unique_number_threshold_key'
    )

    # 複数ファイルのアップロード
    uploaded_files = st.file_uploader("ファイルを選択してください", accept_multiple_files=True)

    all_spectra = []  # すべてのスペクトルを格納するリスト
    all_bsremoval_spectra = []  # ベースライン補正後のスペクトルを格納するリスト
    all_averemoval_spectra = []  # ベースライン補正後移動平均を行ったスペクトルを格納するリスト
    file_labels = []  # 各ファイル名のリスト
    
    if uploaded_files:
        
        # すべてのファイルに対して処理
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1] if '.' in file_name else ''

            try:
                data = read_csv_file(uploaded_file, file_extension)
                file_type = detect_file_type(data)
                uploaded_file.seek(0)
                if file_type == "unknown":
                    st.error(f"{file_name}のファイルタイプを判別できません。")
                    continue

                # 各ファイルタイプに対する処理
                if file_type == "wasatch":
                    st.write(f"ファイルタイプ: Wasatch ENLIGHTEN - {file_name}")
                    lambda_ex = 785
                    data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=46)
                    pre_wavelength = np.array(data["Wavelength"].values)
                    pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
                    pre_spectra = np.array(data["Processed"].values)

                elif file_type == "ramaneye_old":
                    st.write(f"ファイルタイプ: RamanEye Data - {file_name}")
                    df_transposed = data.set_index("WaveNumber").T

                    # 列名（"acrylic board" など）を汎用化
                    df_transposed.columns = ["intensity"]
                    
                    # 波数をfloatに変換し、インデックスに設定
                    df_transposed.index = df_transposed.index.astype(float)
                    df_transposed = df_transposed.sort_index()
                    
                    # 波数と強度をNumPy配列として取得
                    pre_wavenum = df_transposed.index.to_numpy()
                    pre_spectra = df_transposed["intensity"].to_numpy()
                    
                    if pre_wavenum[0] > pre_wavenum[1]:
                        # pre_wavenum と pre_spectra を反転
                        pre_wavenum = pre_wavenum[::-1]
                        pre_spectra = pre_spectra[::-1]
                        
                elif file_type == "ramaneye_new":
                    st.write(f"ファイルタイプ: RamanEye Data - {file_name}")
                    
                    data = pd.read_csv(uploaded_file, skiprows=9)
                    pre_wavenum = data["WaveNumber"]
                    pre_spectra = np.array(data.iloc[:, -1])  # ユーザーの指定に基づく列を取得

                    if pre_wavenum[0] > pre_wavenum[1]:
                        # pre_wavenum と pre_spectra を反転
                        pre_wavenum = pre_wavenum[::-1]
                        pre_spectra = pre_spectra[::-1]
                        
                elif file_type == "eagle":
                    st.write(f"ファイルタイプ: Eagle Data - {file_name}")
                    data_transposed = data.transpose()
                    header = data_transposed.iloc[:3]  # 最初の3行
                    reversed_data = data_transposed.iloc[3:].iloc[::-1]
                    data_transposed = pd.concat([header, reversed_data], ignore_index=True)
                    pre_wavenum = np.array(data_transposed.iloc[3:, 0])
                    pre_spectra = np.array(data_transposed.iloc[3:, 1])
                
                start_index = find_index(pre_wavenum, start_wavenum)
                end_index = find_index(pre_wavenum, end_wavenum)

                wavenum = np.array(pre_wavenum[start_index:end_index+1])
                spectra = np.array(pre_spectra[start_index:end_index+1])

                # Baseline and spike removal 
                spectra_spikerm = remove_outliers_and_interpolate(spectra)
                spectra_spikerm = np.asarray(spectra_spikerm, dtype=np.float64)
                mveAve_spectra = signal.medfilt(spectra_spikerm, savgol_wsize)
                lambda_ = 10e2
                baseline = airPLS(mveAve_spectra, dssn_th, lambda_, 2, 30)
                BSremoval_specta = spectra_spikerm - baseline
                BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(spectra_spikerm, 0))  # 負値を補正

                # 移動平均後のスペクトル
                Averemoval_specta = mveAve_spectra  - baseline
                Averemoval_specta_pos = Averemoval_specta + abs(np.minimum(mveAve_spectra, 0))  # 負値を補正

                # 各スペクトルを格納
                file_labels.append(file_name)  # ファイル名を追加
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)
                all_averemoval_spectra.append(Averemoval_specta_pos)
                
            except Exception as e:
                st.error(f"{file_name}の処理中にエラーが発生しました: {e}")
    
        # すべてのファイルが処理された後に重ねてプロット
        fig, ax = plt.subplots(figsize=(10, 5))
        selected_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'black']
            
        # 元のスペクトルを重ねてプロット
        for i, spectrum in enumerate(all_spectra):
            ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i], label=f"{file_labels[i]}")
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Raw Spectra', fontsize=Fsize)
        ax.legend(title="Spectra")
        st.pyplot(fig)
        
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
            ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i], label=f"{file_labels[i]}")
        
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Baseline Removed', fontsize=Fsize)
        st.pyplot(fig)
        
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
            ax.plot(wavenum, spectrum, linestyle='-', color=selected_colors[i], label=f"{file_labels[i]}")
        
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Baseline Removed + Moving Average', fontsize=Fsize)
        st.pyplot(fig)
        
        # ベースライン補正後+スパイク修正後+移動平均スペクトルをCSVで出力
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
        
        # ピーク位置の検出
        if len(all_averemoval_spectra) > 0:  # データが存在する場合のみ実行
            firstDev_spectra = savitzky_golay(Averemoval_specta_pos, num_firstDev, savgol_order, 1)
            secondDev_spectra = savitzky_golay(Averemoval_specta_pos, num_secondDev, savgol_order, 2)
        
            peak_indices = np.where((firstDev_spectra[:-1] > 0) & (firstDev_spectra[1:] < 0) & 
                                      ((secondDev_spectra[:-1] / abs(np.min(secondDev_spectra[:-1]))) < -10/1000))[0]
            peaks = wavenum[peak_indices]
            
            peak_areas = []
            for peak_idx in peak_indices:
                start_idx, end_idx = find_peak_width(Averemoval_specta_pos, firstDev_spectra, peak_idx, window_size=20)
                area = find_peak_area(Averemoval_specta_pos, start_idx, end_idx)
                peak_areas.append(area)
            
            # Create a DataFrame to display peaks and their areas
            peak_data = {
                "ピーク位置 (cm⁻¹)": peaks,
                "ピーク面積": peak_areas
            }
            peak_df = pd.DataFrame(peak_data)
            
            # ピークの位置をプロット
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(wavenum, Averemoval_specta_pos, linestyle='-', color='b')
            for peak in peaks:
                ax.axvline(x=peak, color='r', linestyle='--', label=f'Peak at {peak}')
            ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
            ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
            ax.set_title('Peak Detection', fontsize=Fsize)
            st.pyplot(fig)

            # ピーク位置を表示
            st.write("ピーク位置:")
            st.table(peak_df)
             
        # Raman correlation table as a pandas DataFrame
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
        
        # Display Raman correlation table
        st.subheader("（参考）ラマン分光の帰属表")
        st.table(raman_df)

def multivariate_analysis_mode():
    """
    Multivariate analysis mode
    """
    st.header("📊 多変量解析ツール")
    
    # Parameters in sidebar
    start_wavenum = st.sidebar.number_input("Start Wavenumber", value=400, min_value=0, key="mv_start")
    end_wavenum = st.sidebar.number_input("End Wavenumber", value=1800, min_value=start_wavenum + 1, key="mv_end")
    
    st.sidebar.subheader("ベースライン削除")
    lambda_param = st.sidebar.number_input("ベースラインパラメーター", value=1000, min_value=10, step=10, key="mv_lambda")
    porder = st.sidebar.selectbox("指数", options=[1, 2, 3], index=1, key="mv_porder")
    
    st.sidebar.subheader("多変量解析")
    n_components = st.sidebar.selectbox("コンポーネント数", options=[2, 3, 4, 5], index=0, key="mv_components")
    
    # File upload
    st.header("📁 データアップロード")
    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type=['csv', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple CSV files with spectral data. Files should have Wavenumber, Raw, and Processed columns starting from row 46.",
        key="mv_uploader"
    )
    
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} files")
        
        # Show uploaded file names
        st.write("**Uploaded files:**")
        for file in uploaded_files:
            st.write(f"- {file.name}")
        
        # Process button
        if st.button("🔄 データプロセス実効", type="primary", key="mv_process"):
            with st.spinner("スペクトルデータを解析しています..."):
                
                # Load and process data
                grouped_data, all_baseline_spectra, all_labels, wavenumber = load_and_process_data_for_multivariate(
                    uploaded_files, start_wavenum, end_wavenum, lambda_param, porder
                )
                
                if len(all_baseline_spectra) > 0:
                    # Store data in session state
                    st.session_state.mv_grouped_data = grouped_data
                    st.session_state.mv_all_baseline_spectra = all_baseline_spectra
                    st.session_state.mv_all_labels = all_labels
                    st.session_state.mv_wavenumber = wavenumber
                    
                    st.success("データプロセス完了!")
                else:
                    st.error("アップロードされたファイルに有効なデータが見つかりませんでした")
    
    # Display results if data is processed
    if hasattr(st.session_state, 'mv_grouped_data') and st.session_state.mv_grouped_data:
        
        st.header("📈 スペクトルプロット")
        
        # Plot original spectra
        fig_spectra = plot_spectra_matplotlib(st.session_state.mv_grouped_data)
        st.pyplot(fig_spectra)
        
        # Display data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Spectra", len(st.session_state.mv_all_baseline_spectra))
        with col2:
            st.metric("Number of Groups", len(st.session_state.mv_grouped_data))
        with col3:
            st.metric("Data Points per Spectrum", len(st.session_state.mv_wavenumber))
        
        # Analysis
        st.header("🔬 多変量解析")
        
        if st.button("🚀 多変量解析実効", type="primary", key="mv_analyze"):
            with st.spinner("多変量解析を行っています..."):
                
                # Perform
                W, H, nmf_model = perform_nmf_analysis(st.session_state.mv_all_baseline_spectra, n_components)
                
                # Store NMF results
                st.session_state.mv_W = W
                st.session_state.mv_H = H
                st.session_state.mv_nmf_model = nmf_model
                
                # Calculate reconstruction error
                reconstruction_error = nmf_model.reconstruction_err_
                st.session_state.mv_reconstruction_error = reconstruction_error
                
                st.success(f"解析完了! Reconstruction error: {reconstruction_error:.4f}")
        
        # Display NMF results if available
        if hasattr(st.session_state, 'mv_W') and hasattr(st.session_state, 'mv_H'):
            
            # Display reconstruction error
            st.info(f"**Reconstruction Error:** {st.session_state.mv_reconstruction_error:.4f}")
            
            # Plot NMF components
            st.subheader("コンポーネントスペクトル")
            fig_components = plot_nmf_components(st.session_state.mv_H, st.session_state.mv_wavenumber)
            st.pyplot(fig_components)
            
            # Plot NMF scores
            st.subheader("スコアプロット")
            group_names = list(st.session_state.mv_grouped_data.keys())
            fig_scores = plot_nmf_scores(st.session_state.mv_W, st.session_state.mv_all_labels, group_names)
            if fig_scores:
                st.pyplot(fig_scores)
            
            # Display contribution ratios
            st.subheader("Component Contribution Ratios")
            contribution_ratios = np.sum(st.session_state.mv_W, axis=0)
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
                W_df = pd.DataFrame(st.session_state.mv_W, 
                                  columns=[f"Component_{i+1}" for i in range(st.session_state.mv_W.shape[1])])
                W_df['Sample'] = st.session_state.mv_all_labels
                csv_W = W_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Concentration Matrix",
                    data=csv_W,
                    file_name="nmf_concentration_matrix.csv",
                    mime="text/csv",
                    key="mv_download_W"
                )
            
            with col2:
                # Download spectral components
                H_df = pd.DataFrame(st.session_state.mv_H.T, 
                                  columns=[f"Component_{i+1}" for i in range(st.session_state.mv_H.shape[0])])
                H_df['Wavenumber'] = st.session_state.mv_wavenumber
                csv_H = H_df.to_csv(index=False)
                st.download_button(
                    label="📈 Download Spectral Components",
                    data=csv_H,
                    file_name="nmf_spectral_components.csv",
                    mime="text/csv",
                    key="mv_download_H"
                )
            
            # Advanced analysis section
            st.subheader("🔍 Advanced Analysis")
            
            # Show individual sample information
            if st.checkbox("Show Individual Sample Information", key="mv_sample_info"):
                sample_info_df = pd.DataFrame({
                    'Sample Index': range(len(st.session_state.mv_all_labels)),
                    'Group': st.session_state.mv_all_labels,
                    **{f'Component_{i+1}': st.session_state.mv_W[:, i] for i in range(st.session_state.mv_W.shape[1])}
                })
                st.dataframe(sample_info_df, use_container_width=True)

def main():
    st.set_page_config(page_title="統合ラマンスペクトル解析ツール", page_icon="📊", layout="wide")
    
    st.title("📊 統合ラマンスペクトル解析ツール")
    
    # Mode selection in sidebar
    st.sidebar.header("🔧 解析モード選択")
    analysis_mode = st.sidebar.selectbox(
        "解析モードを選択してください:",
        ["スペクトル解析", "多変量解析"],
        key="mode_selector"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📋 パラメータ設定")
    
    if analysis_mode == "スペクトル解析":
        spectrum_analysis_mode()
    else:
        multivariate_analysis_mode()
    
    # Instructions
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 使用方法")
    
    if analysis_mode == "スペクトル解析":
        st.sidebar.markdown("""
        **スペクトル解析モード:**
        1. 解析したいCSVファイルをアップロード
        2. パラメータを調整
        3. スペクトルの表示と解析結果を確認
        4. ピーク検出結果とラマン帰属表を参照
        5. 結果をCSVファイルでダウンロード
        """)
    else:
        st.sidebar.markdown("""
        **多変量解析モード:**
        1. 複数のCSVファイルをアップロード
        2. パラメータを調整
        3. 「データプロセス実効」をクリック
        4. 「多変量解析実効」をクリック
        5. 解析結果を確認・ダウンロード
        
        **ファイル命名規則:**
        GroupName_Number.csv
        """)

if __name__ == "__main__":
    main()
    
