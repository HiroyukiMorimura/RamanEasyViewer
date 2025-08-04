# -*- coding: utf-8 -*-
"""
共通ユーティリティ関数
全てのモジュールで使用される共通関数とインポート
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix, eye, diags
from pathlib import Path
import io

def detect_file_type(data):
    """
    Determine the structure of the input data.
    """
    try:
        if data.columns[0].split(':')[0] == "# Laser Wavelength":
            return "ramaneye_new"
        elif data.columns[0] == "WaveNumber":
            return "ramaneye_old"
        elif data.columns[0] == "Timestamp":
            return "ramaneye_old_old"
        elif data.columns[0] == "Pixels":
            return "eagle"
        elif data.columns[0] == "ENLIGHTEN Version":
            return "wasatch"
        return "unknown"
    except:
        return "unknown"

def safe_seek(file_obj):
    """
    ファイルオブジェクトを安全にseekする関数
    """
    try:
        if hasattr(file_obj, 'seek'):
            file_obj.seek(0)
    except Exception as e:
        print(f"Warning: seek操作に失敗しました: {e}")

def read_csv_file(uploaded_file, file_extension):
    """
    Read a CSV or TXT file into a DataFrame based on file extension.
    """
    try:
        safe_seek(uploaded_file)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', header=0, index_col=None, on_bad_lines='skip')
        return data
    except UnicodeDecodeError:
        safe_seek(uploaded_file)
        if file_extension == "csv":
            data = pd.read_csv(uploaded_file, sep=',', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        else:
            data = pd.read_csv(uploaded_file, sep='\t', encoding='shift_jis', header=0, index_col=None, on_bad_lines='skip')
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def get_file_name(uploaded_file):
    """
    アップロードされたファイルから安全にファイル名を取得する関数
    """
    try:
        if hasattr(uploaded_file, 'name'):
            return uploaded_file.name
        elif hasattr(uploaded_file, 'filename'):
            return uploaded_file.filename
        else:
            return "unknown_file"
    except Exception as e:
        print(f"Warning: ファイル名の取得に失敗しました: {e}")
        return "unknown_file"

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
    # マイナス値がある場合の処理
    min_value = np.min(x)
    offset = 0
    if min_value < 0:
        offset = abs(min_value) + 1
        x = x + offset
    
    m = x.shape[0]
    w = np.ones(m, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        
        if dssn < 1e-10:
            dssn = 1e-10
        
        if (dssn < dssn_th * (np.abs(x)).sum()) or (i == itermax):
            if i == itermax:
                print('WARNING: max iteration reached!')
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
    spectrum_len = len(spectrum)
    cleaned_spectrum = spectrum.copy()
    
    for i in range(spectrum_len):
        left_idx = max(i - window_size, 0)
        right_idx = min(i + window_size + 1, spectrum_len)
        
        window = spectrum[left_idx:right_idx]
        window_median = np.median(window)
        window_std = np.std(window)
        
        if abs(spectrum[i] - window_median) > threshold_factor * window_std:
            if i > 0 and i < spectrum_len - 1:
                cleaned_spectrum[i] = (spectrum[i - 1] + spectrum[i + 1]) / 2
            elif i == 0:
                cleaned_spectrum[i] = spectrum[i + 1]
            elif i == spectrum_len - 1:
                cleaned_spectrum[i] = spectrum[i - 1]
    
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

def try_read_wasatch_file(uploaded_file, skiprows_list=None):
    """
    Wasatchファイルを複数のskiprows値で試行錯誤して読み込む
    
    Parameters:
    uploaded_file: アップロードされたファイル
    skiprows_list: 試行するskiprows値のリスト
    
    Returns:
    data: 読み込み成功したDataFrame（失敗時はNone）
    successful_skiprows: 成功したskiprows値
    """
    if skiprows_list is None:
        # デフォルトの試行リスト（よく使われる値を優先）
        skiprows_list = [46, 47, 45, 44, 48, 49, 50]
    
    for skiprows in skiprows_list:
        try:
            safe_seek(uploaded_file)
            # UTF-8を先に試す
            try:
                data = pd.read_csv(uploaded_file, skiprows=skiprows, nrows=None)
            except UnicodeDecodeError:
                # UTF-8で失敗した場合はshift-jisを試す
                safe_seek(uploaded_file)
                data = pd.read_csv(uploaded_file, encoding='shift-jis', skiprows=skiprows, nrows=None)
            
            # Wavelength列とProcessed列の存在を確認
            if 'Wavelength' in data.columns and 'Processed' in data.columns:
                print(f"INFO: Wasatchファイルの読み込みに成功しました (skiprows={skiprows})")
                return data, skiprows
            
            # Wavelength列だけでもある場合は候補として保持
            elif 'Wavelength' in data.columns:
                # Processedがない場合、他の数値列を探す
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) > 1:  # Wavelength以外にも数値列がある
                    print(f"INFO: Wasatchファイルの読み込みに成功しました (skiprows={skiprows}, Processed列の代替使用)")
                    return data, skiprows
                    
        except Exception as e:
            # このskiprows値では読み込めないので次を試す
            continue
    
    # 全ての試行が失敗
    print("WARNING: Wasatchファイルの読み込みに失敗しました（全てのskiprows値で試行）")
    return None, None

def process_spectrum_file(uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize):
    """
    スペクトルファイルを処理する共通関数
    """
    # 安全にファイル名を取得
    file_name = get_file_name(uploaded_file)
    file_extension = file_name.split('.')[-1].lower()
    
    # ファイルオブジェクトの検証
    if uploaded_file is None:
        print("ERROR: uploaded_fileがNoneです")
        return None, None, None, None, None, file_name
    
    # bytesオブジェクトの場合はStringIOに変換
    if isinstance(uploaded_file, bytes):
        try:
            uploaded_file = io.StringIO(uploaded_file.decode('utf-8'))
        except UnicodeDecodeError:
            try:
                uploaded_file = io.StringIO(uploaded_file.decode('shift-jis'))
            except UnicodeDecodeError:
                print("ERROR: バイトデータのデコードに失敗しました")
                return None, None, None, None, None, file_name
    
    data = read_csv_file(uploaded_file, file_extension)
    if data is None:
        print(f"ERROR: ファイル {file_name} の読み込みに失敗しました")
        return None, None, None, None, None, file_name
    
    file_type = detect_file_type(data)
    safe_seek(uploaded_file)
    
    if file_type == "unknown":
        print(f"ERROR: 未知のファイル形式です: {file_name}")
        return None, None, None, None, None, file_name
    
    # 各ファイルタイプに応じた処理
    try:
        if file_type == "wasatch":
            lambda_ex = 785
            
            # 複数のskiprows値を試行錯誤
            data, successful_skiprows = try_read_wasatch_file(uploaded_file)
            
            if data is None:
                print(f"ERROR: Wasatchファイル {file_name} の読み込みに失敗しました")
                return None, None, None, None, None, file_name
            
            try:
                pre_wavelength = np.array(data["Wavelength"].values)
                pre_wavenum = (1e7 / lambda_ex) - (1e7 / pre_wavelength)
                
                # Processed列を優先的に使用、なければ他の数値列を使用
                if "Processed" in data.columns:
                    pre_spectra = np.array(data["Processed"].values)
                else:
                    # Processed列がない場合、Wavelength以外の数値列を使用
                    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                    spectrum_columns = [col for col in numeric_columns if col != 'Wavelength']
                    
                    if spectrum_columns:
                        # 最後の数値列をスペクトルデータとして使用
                        spectrum_column = spectrum_columns[-1]
                        pre_spectra = np.array(data[spectrum_column].values)
                        print(f"INFO: Processed列の代わりに{spectrum_column}列をスペクトルデータとして使用")
                    else:
                        print(f"ERROR: スペクトルデータに使用できる数値列が見つかりません")
                        return None, None, None, None, None, file_name
                        
            except KeyError as e:
                print(f"ERROR: 必要な列が見つかりません: {e}")
                print(f"利用可能な列: {list(data.columns)}")
                return None, None, None, None, None, file_name
            except Exception as e:
                print(f"ERROR: Wasatchファイルの処理中にエラーが発生しました: {e}")
                return None, None, None, None, None, file_name
            
        elif file_type == "ramaneye_old_old":
            df_transposed = data.set_index("WaveNumber").T
            df_transposed.columns = ["intensity"]
            df_transposed.index = df_transposed.index.astype(float)
            df_transposed = df_transposed.sort_index()
            
            pre_wavenum = df_transposed.index.to_numpy()
            pre_spectra = df_transposed["intensity"].to_numpy()
            
            if pre_wavenum[0] >= pre_wavenum[1]:
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                
        elif file_type == "ramaneye_old":
            # WaveNumber列を取得
            pre_wavenum = np.array(data["WaveNumber"].values)
            # 最後の列（スペクトラムデータ）を取得
            pre_spectra = np.array(data.iloc[:, -1].values)
            
            # データの長さチェック
            if len(pre_wavenum) != len(pre_spectra):
                print(f"ERROR: 波数とスペクトラムのデータ長が一致しません")
                return None, None, None, None, None, file_name
            
            if len(pre_wavenum) > 1 and pre_wavenum[0] >= pre_wavenum[1]:
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                
        elif file_type == "ramaneye_new":
            safe_seek(uploaded_file)
            data = pd.read_csv(uploaded_file, skiprows=9)
            pre_wavenum = np.array(data["WaveNumber"].values)
            pre_spectra = np.array(data.iloc[:, -1].values)
            
            if len(pre_wavenum) > 1 and pre_wavenum[0] >= pre_wavenum[1]:
                pre_wavenum = pre_wavenum[::-1]
                pre_spectra = pre_spectra[::-1]
                
        elif file_type == "eagle":
            data_transposed = data.transpose()
            header = data_transposed.iloc[:3]
            reversed_data = data_transposed.iloc[3:].iloc[::-1]
            data_transposed = pd.concat([header, reversed_data], ignore_index=True)
            pre_wavenum = np.array(data_transposed.iloc[3:, 0])
            pre_spectra = np.array(data_transposed.iloc[3:, 1])
        
        # 波数範囲の切り出し
        start_index = find_index(pre_wavenum, start_wavenum)
        end_index = find_index(pre_wavenum, end_wavenum)
        
        wavenum = np.array(pre_wavenum[start_index:end_index+1])
        spectra = np.array(pre_spectra[start_index:end_index+1])
        
        # スパイク除去とベースライン補正
        spectra_spikerm = remove_outliers_and_interpolate(spectra)
        mveAve_spectra = signal.medfilt(spectra_spikerm, savgol_wsize)
        lambda_ = 10e2
        baseline = airPLS(mveAve_spectra, dssn_th, lambda_, 2, 30)
        BSremoval_specta = spectra_spikerm - baseline
        BSremoval_specta_pos = BSremoval_specta + abs(np.minimum(spectra_spikerm, 0))
        
        # 移動平均後のスペクトル
        Averemoval_specta = mveAve_spectra - baseline
        Averemoval_specta_pos = Averemoval_specta + abs(np.minimum(mveAve_spectra, 0))
        
        return wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name
        
    except Exception as e:
        print(f"ERROR: ファイル処理中に予期しないエラーが発生しました: {e}")
        return None, None, None, None, None, file_name
