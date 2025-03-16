# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 19:03:55 2024

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

def airPLS(x, dssn_th=0.00001, lambda_=100, porder=1, itermax=30):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    
    input
        x: input data (i.e. chromatogram or spectrum)
        lambda_: parameter that can be adjusted by user. The larger lambda is, the smoother the resulting background, z
        porder: adaptive iteratively reweighted penalized least squares for baseline fitting
    
    output
        the fitted background vector
    '''
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
        if data.columns[0] == "Timestamp":
            return "ramaneye"
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
    if file_extension == "csv":
        return pd.read_csv(uploaded_file, sep=',', header=0, index_col=None, on_bad_lines='skip')
    else:
        return pd.read_csv(uploaded_file, sep='\t', header=0, index_col=None, on_bad_lines='skip')

def main():
    # パラメータ設定
    savgol_wsize         = 5    # Savitzky-Golayフィルタのウィンドウサイズ
    savgol_order         = 3    # Savitzky-Golayフィルタの次数
    pre_start_wavenum    = 200  # 波数の開始
    pre_end_wavenum      = 3600 # 波数の終了
    wavenum_calibration  = -0   # 校正オフセット
    Designated_peak_wn   = 1700 # ピーク指定波数
    Fsize                = 14   # フォントサイズ
    
    st.title("Raman Spectrum Viewer")

    # 複数ファイルのアップロード
    uploaded_files = st.file_uploader("ファイルを選択してください", accept_multiple_files=True)

    all_spectra = []  # すべてのスペクトルを格納するリスト
    all_bsremoval_spectra = []  # ベースライン補正後のスペクトルを格納するリスト
    file_labels = []  # 各ファイル名のリスト
    
    if uploaded_files:
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan', 'yellow', 'black']
        
        # 波数範囲の設定
        start_wavenum = st.number_input("波数（開始）を入力してください:", min_value=100, max_value=3600, value=pre_start_wavenum, step=100)
        end_wavenum = st.number_input("波数（終了）を入力してください:", min_value=start_wavenum+100, max_value=3600, value=pre_end_wavenum, step=100)

        # すべてのファイルに対して処理
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name
            file_extension = file_name.split('.')[-1] if '.' in file_name else ''

            try:
                data = read_csv_file(uploaded_file, file_extension)
                file_type = detect_file_type(data)

                if file_type == "unknown":
                    st.error(f"{file_name}のファイルタイプを判別できません。")
                    continue

                # 各ファイルタイプに対する処理
                if file_type == "wasatch":
                    st.write(f"ファイルタイプ: Wasatch ENLIGHTEN - {file_name}")
                    lambda_ex = 1064
                    data = pd.read_csv(uploaded_file, skiprows=45)
                    pre_wavelength = np.array(data["Wavelength"].values)
                    pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
                    pre_spectra = np.array(data["Processed"].values)

                elif file_type == "ramaneye":
                    st.write(f"ファイルタイプ: RamanEye Data - {file_name}")
                    pre_wavenum = data["WaveNumber"]
                    pre_spectra = np.array(data.iloc[:, -1])  # ユーザーの指定に基づく列を取得

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

                # Baseline removal
                mveAve_spectra = signal.medfilt(spectra, savgol_wsize)
                baseline = airPLS(mveAve_spectra, 0.00001, 10e1, 2)
                BSremoval_specta = spectra - baseline
                BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(BSremoval_specta, 0))  # 負値を補正
                
                # 各スペクトルを格納
                file_labels.append(file_name)  # ファイル名を追加
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)

            except Exception as e:
                st.error(f"{file_name}の処理中にエラーが発生しました: {e}")
    
        # すべてのファイルが処理された後に重ねてプロット
        fig, ax = plt.subplots(figsize=(10, 5))

        # 元のスペクトルを重ねてプロット
        for i, spectrum in enumerate(all_spectra):
            ax.plot(wavenum, spectrum, linestyle='-', color=colors[i % len(colors)], label=f"{file_labels[i]}")
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Raw Spectra', fontsize=Fsize)
        ax.legend(title="Spectra")
        st.pyplot(fig)

        # ベースライン補正後のスペクトルを重ねてプロット
        fig, ax = plt.subplots(figsize=(10, 5))        
        for i, spectrum in enumerate(all_bsremoval_spectra):
            ax.plot(wavenum, spectrum, linestyle='-', color=colors[i % len(colors)], label=f"{file_labels[i]}")
        
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Baseline Removed', fontsize=Fsize)
        ax.legend(title="Spectra")
        st.pyplot(fig)

        # ピーク位置の検出
        firstDev_spectra = savitzky_golay(BSremoval_specta_pos, 13, savgol_order, 1)
        secondDev_spectra = savitzky_golay(BSremoval_specta_pos, 5, savgol_order, 2)
    
        peak_indices = np.where((firstDev_spectra[:-1] > 0) & (firstDev_spectra[1:] < 0) & 
                                  ((secondDev_spectra[:-1] / abs(np.min(secondDev_spectra[:-1]))) < -10/1000))[0]
        peaks = wavenum[peak_indices]

        # ピークの位置をプロット
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(wavenum, BSremoval_specta_pos, linestyle='-', color='b')
        for peak in peaks:
            ax.axvline(x=peak, color='r', linestyle='--', label=f'Peak at {peak}')
        ax.set_xlabel('WaveNumber / cm-1', fontsize=Fsize)
        ax.set_ylabel('Intensity / a.u.', fontsize=Fsize)
        ax.set_title('Peak Detection', fontsize=Fsize)
        st.pyplot(fig)

        # ピーク位置を表示
        st.write("ピーク位置:")
        st.table(peaks)
        
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
        
if __name__ == "__main__":
    main()
