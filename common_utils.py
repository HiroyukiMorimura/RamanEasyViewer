# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数（デバッグ機能付き）
全てのモジュールで使用される共通関数とインポート
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags
from pathlib import Path

def debug_print(message, data=None):
    """デバッグ用プリント関数"""
    print(f"[DEBUG] {message}")
    if data is not None:
        if hasattr(data, 'shape'):
            print(f"[DEBUG] データの形状: {data.shape}")
        if hasattr(data, 'dtype'):
            print(f"[DEBUG] データ型: {data.dtype}")
        if hasattr(data, '__len__'):
            print(f"[DEBUG] データ長: {len(data)}")
        if isinstance(data, (pd.DataFrame, pd.Series)):
            print(f"[DEBUG] データの最初の5行:")
            print(data.head())
            if isinstance(data, pd.DataFrame):
                print(f"[DEBUG] 列名: {list(data.columns)}")

def detect_file_type(data):
    """
    Determine the structure of the input data.
    """
    try:
        debug_print(f"ファイルタイプ検出開始", data)
        debug_print(f"最初の列名: {data.columns[0]}")
        
        if data.columns[0].split(':')[0] == "# Laser Wavelength":
            debug_print("ファイルタイプ: ramaneye_new")
            return "ramaneye_new"
        elif data.columns[0] == "WaveNumber":
            debug_print("ファイルタイプ: ramaneye_old")
            return "ramaneye_old"
        elif data.columns[0] == "Timestamp":
            debug_print("ファイルタイプ: ramaneye_old_old")
            return "ramaneye_old_old"
        elif data.columns[0] == "Pixels":
            debug_print("ファイルタイプ: eagle")
            return "eagle"
        elif data.columns[0] == "ENLIGHTEN Version":
            debug_print("ファイルタイプ: wasatch")
            return "wasatch"
        
        debug_print("ファイルタイプ: unknown")
        return "unknown"
    except Exception as e:
        debug_print(f"ファイルタイプ検出エラー: {e}")
        return "unknown"

def read_csv_file(uploaded_file, file_extension):
    """
    Read a CSV or TXT file into a DataFrame based on file extension.
    """
    try:
        uploaded_file.seek(0)
        debug_print(f"ファイル読み込み開始 - 拡張子: {file_extension}")
        
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', header=0, index_col=None, on_bad_lines='skip')
            
        debug_print("ファイル読み込み成功", data)
        return data
        
    except UnicodeDecodeError:
        debug_print("UTF-8での読み込み失敗、Shift_JISで再試行")
        uploaded_file.seek(0)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        debug_print("Shift_JISでの読み込み成功", data)
        return data
        
    except Exception as e:
        debug_print(f"ファイル読み込みエラー: {e}")
        return None

def find_index(rs_array, rs_focused):
    '''
    Convert the index of the proximate wavenumber by finding the absolute 
    minimum value of (rs_array - rs_focused)
    '''
    diff = [abs(element - rs_focused) for element in rs_array]
    index = np.argmin(diff)
    return index

def WhittakerSmooth(x, w, lambda_, differences=1):
    '''
    Penalized least squares algorithm for background fitting
    '''
    X = np.array(x, dtype=np.float64)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T).toarray().flatten()
    background = spsolve(A, B)
    return np.array(background)

def airPLS(x, dssn_th, lambda_, porder, itermax):
    '''
    Adaptive iteratively reweighted penalized least squares for baseline fitting
    '''
    debug_print(f"airPLS開始 - 入力データ", x)
    
    # マイナス値がある場合の処理
    min_value = np.min(x)
    offset = 0
    if min_value < 0:
        offset = abs(min_value) + 1
        x = x + offset
        debug_print(f"負の値を補正 - オフセット: {offset}")
    
    m = x.shape[0]
    w = np.ones(m, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    debug_print(f"airPLS処理開始 - データ長: {m}")
    
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        
        if dssn < 1e-10:
            dssn = 1e-10
        
        if (dssn < dssn_th * (np.abs(x)).sum()) or (i == itermax):
            if i == itermax:
                debug_print('WARNING: max iteration reached!')
            debug_print(f"airPLS完了 - 反復回数: {i}")
            break
        
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        
        if d[d < 0].size > 0:
            w[0] = np.exp(i * np.abs(d[d < 0]).max() / dssn)
        else:
            w[0] = 1.0
        
        w[-1] = w[0]

    return z

def remove_outliers_and_interpolate(spectrum, window_size=10, threshold_factor=3):
    """
    スペクトルからスパイク（外れ値）を検出し、補完する関数
    """
    debug_print("スパイク除去開始", spectrum)
    spectrum_len = len(spectrum)
    cleaned_spectrum = spectrum.copy()
    
    spike_count = 0
    for i in range(spectrum_len):
        left_idx = max(i - window_size, 0)
        right_idx = min(i + window_size + 1, spectrum_len)
        
        window = spectrum[left_idx:right_idx]
        window_median = np.median(window)
        window_std = np.std(window)
        
        if abs(spectrum[i] - window_median) > threshold_factor * window_std:
            spike_count += 1
            if i > 0 and i < spectrum_len - 1:
                cleaned_spectrum[i] = (spectrum[i - 1] + spectrum[i + 1]) / 2
            elif i == 0:
                cleaned_spectrum[i] = spectrum[i + 1]
            elif i == spectrum_len - 1:
                cleaned_spectrum[i] = spectrum[i - 1]
    
    debug_print(f"スパイク除去完了 - 除去したスパイク数: {spike_count}")
    return cleaned_spectrum

def find_peak_width(spectra, first_dev, peak_position, window_size=20):
    """
    Find the peak start/end close the peak position
    """
    start_idx = max(peak_position - window_size, 0)
    end_idx = min(peak_position + window_size, len(first_dev) - 1)
    
    local_start_idx = np.argmax(first_dev[start_idx:end_idx+1]) + start_idx
    local_end_idx = np.argmin(first_dev[start_idx:end_idx+1]) + start_idx
    
    return local_start_idx, local_end_idx

def find_peak_area(spectra, local_start_idx, local_end_idx):
    """
    Calculate the area of the peaks
    """
    peak_area = np.trapz(spectra[local_start_idx:local_end_idx+1], dx=1)
    return peak_area

def calculate_peak_width(spectrum, peak_idx, wavenum):
    """
    ピークの半値幅（FWHM）を計算する関数
    """
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return 0.0
    
    peak_intensity = spectrum[peak_idx]
    half_max = peak_intensity / 2.0
    
    # ピークから左側に向かって半値点を探す
    left_idx = peak_idx
    while left_idx > 0 and spectrum[left_idx] > half_max:
        left_idx -= 1
    
    # 線形補間で正確な半値点を求める
    if left_idx < peak_idx and spectrum[left_idx] <= half_max < spectrum[left_idx + 1]:
        ratio = (half_max - spectrum[left_idx]) / (spectrum[left_idx + 1] - spectrum[left_idx])
        left_wavenum = wavenum[left_idx] + ratio * (wavenum[left_idx + 1] - wavenum[left_idx])
    else:
        left_wavenum = wavenum[left_idx] if left_idx >= 0 else wavenum[0]
    
    # ピークから右側に向かって半値点を探す
    right_idx = peak_idx
    while right_idx < len(spectrum) - 1 and spectrum[right_idx] > half_max:
        right_idx += 1
    
    # 線形補間で正確な半値点を求める
    if right_idx > peak_idx and spectrum[right_idx] <= half_max < spectrum[right_idx - 1]:
        ratio = (half_max - spectrum[right_idx]) / (spectrum[right_idx - 1] - spectrum[right_idx])
        right_wavenum = wavenum[right_idx] + ratio * (wavenum[right_idx - 1] - wavenum[right_idx])
    else:
        right_wavenum = wavenum[right_idx] if right_idx < len(wavenum) else wavenum[-1]
    
    # 半値幅を計算
    fwhm = abs(right_wavenum - left_wavenum)
    return fwhm

def process_spectrum_file(uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
    """
    スペクトルファイルを処理する共通関数（デバッグ機能付き）
    """
    file_name = uploaded_file.name
    file_extension = file_name.split('.')[-1].lower()
    
    debug_print(f"ファイル処理開始: {file_name}")
    debug_print(f"パラメータ - start_wavenum: {start_wavenum}, end_wavenum: {end_wavenum}")
    debug_print(f"パラメータ - dssn_th: {dssn_th}, savgol_wsize: {savgol_wsize}")
    
    try:
        data = read_csv_file(uploaded_file, file_extension)
        if data is None:
            debug_print("データ読み込み失敗")
            return None, None, None, None, None, file_name
        
        file_type = detect_file_type(data)
        uploaded_file.seek(0)
        
        if file_type == "unknown":
            debug_print("未知のファイルタイプ")
            return None, None, None, None, None, file_name
        
        debug_print(f"検出されたファイルタイプ: {file_type}")
        
        # 各ファイルタイプに応じた処理
        if file_type == "wasatch":
            debug_print("wasatchファイル処理開始")
            lambda_ex = 785
            data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=46)
            pre_wavelength = np.array(data["Wavelength"].values)
            pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
            pre_spectra = np.array(data["Processed"].values)
            debug_print("wasatchファイル処理完了", pre_spectra)
            
        elif file_type == "ramaneye_old_old":
            debug_print("ramaneye_old_oldファイル処理開始")
            df_transposed = data.set_index("WaveNumber").T
            df_transposed.columns = ["intensity"]
            df_transposed.index = df_transposed.index.astype(float)
            df_transposed = df_transposed.sort_index()
            
            pre_wavenum = df_transposed.index.to_numpy()
            pre_spectra = df_transposed["intensity"].to_numpy()
            
            debug_print("変換前データ", pre_spectra)
            debug_print("変換前波数", pre_wavenum)
            
            if pre_wavenum[0] >= pre_wavenum[1]:
                debug_print("データを逆順に変換")
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                
        elif file_type == "ramaneye_old":
            debug_print("ramaneye_oldファイル処理開始")
            debug_print("元データの列名", list(data.columns))
            
            # WaveNumber列を取得（インデックスに設定する前に）
            pre_wavenum = np.array(data["WaveNumber"].values)
            debug_print("波数データ取得", pre_wavenum)
            
            # 最後の列（スペクトラムデータ）を取得
            pre_spectra = np.array(data.iloc[:, -1].values)
            debug_print("スペクトラムデータ取得", pre_spectra) 
            
            # データの長さチェック
            if len(pre_wavenum) != len(pre_spectra):
                debug_print(f"データ長不一致 - 波数: {len(pre_wavenum)}, スペクトラム: {len(pre_spectra)}")
                return None, None, None, None, None, file_name
            
            debug_print(f"データ方向チェック - 最初の波数: {pre_wavenum[0]}, 2番目の波数: {pre_wavenum[1]}")
            
            if len(pre_wavenum) > 1 and pre_wavenum[0] >= pre_wavenum[1]:
                debug_print("データを逆順に変換")
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                debug_print("逆順変換後 - 波数", pre_wavenum)
                debug_print("逆順変換後 - スペクトラム", pre_spectra)
                
        elif file_type == "ramaneye_new":
            debug_print("ramaneye_newファイル処理開始")
            data = pd.read_csv(uploaded_file, skiprows=9)
            pre_wavenum = np.array(data["WaveNumber"].values)
            pre_spectra = np.array(data.iloc[:, -1].values)
            
            debug_print("変換前データ", pre_spectra)
            debug_print("変換前波数", pre_wavenum)
            
            if len(pre_wavenum) > 1 and pre_wavenum[0] >= pre_wavenum[1]:
                debug_print("データを逆順に変換")
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                
        elif file_type == "eagle":
            debug_print("eagleファイル処理開始")
            data_transposed = data.transpose()
            header = data_transposed.iloc[:3]
            reversed_data = data_transposed.iloc[3:].iloc[::-1]
            data_transposed = pd.concat([header, reversed_data], ignore_index=True)
            pre_wavenum = np.array(data_transposed.iloc[3:, 0])
            pre_spectra = np.array(data_transposed.iloc[3:, 1])
            debug_print("eagleファイル処理完了", pre_spectra)
        
        debug_print("前処理完了 - 波数範囲の切り出し開始")
        debug_print(f"波数範囲: {pre_wavenum[0]} から {pre_wavenum[-1]}")
        
        # 波数範囲の切り出し
        start_index = find_index(pre_wavenum, start_wavenum)
        end_index = find_index(pre_wavenum, end_wavenum)
        
        debug_print(f"切り出しインデックス - 開始: {start_index}, 終了: {end_index}")
        
        wavenum = np.array(pre_wavenum[start_index:end_index+1])
        spectra = np.array(pre_spectra[start_index:end_index+1])
        
        debug_print("切り出し後データ", spectra)
        debug_print("切り出し後波数", wavenum)
        
        # スパイク除去とベースライン補正
        debug_print("スパイク除去開始")
        spectra_spikerm = remove_outliers_and_interpolate(spectra)
        
        debug_print("メディアンフィルタ適用")
        mveAve_spectra = signal.medfilt(spectra_spikerm, savgol_wsize)
        debug_print("メディアンフィルタ後", mveAve_spectra)
        
        debug_print("ベースライン補正開始")
        lambda_ = 10e2
        baseline = airPLS(mveAve_spectra, dssn_th, lambda_, 2, 30)
        debug_print("ベースライン", baseline)
        
        BSremoval_specta = spectra_spikerm - baseline
        BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(spectra_spikerm, 0))
        debug_print("ベースライン除去後", BSremoval_specta_pos)
        
        # 移動平均後のスペクトル
        Averemoval_specta = mveAve_spectra - baseline
        Averemoval_specta_pos = Averemoval_specta + abs(np.minimum(mveAve_spectra, 0))
        debug_print("最終処理後", Averemoval_specta_pos)
        
        debug_print("ファイル処理正常完了")
        return wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name
        
    except Exception as e:
        debug_print(f"処理中にエラーが発生: {e}")
        import traceback
        debug_print("エラーの詳細:")
        print(traceback.format_exc())
        return None, None, None, None, None, file_name
