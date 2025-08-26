# -*- coding: utf-8 -*-
"""
スペクトル解析モジュール（堅牢版・legacy優先）
- common_utils.process_spectrum_file があれば必ずそれを使う
- 無い場合のみ堅牢ローダーで読みに行く
- 既定でユーザーの Downloads を監視（watchdog が無ければ監視はスキップ）
- フォルダが無い/空でもファイルアップロードで解析可能
"""

from __future__ import annotations

import os
import io
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- seaborn（任意）
try:
    import seaborn as sns  # type: ignore
    plt.style.use("default")
    sns.set_palette("husl")
except Exception:
    plt.style.use("default")

# --- watchdog（任意）
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False

# --- common_utils（任意）: legacy loader を最優先で使う
LEGACY_LOADER = None
try:
    from common_utils import process_spectrum_file as LEGACY_LOADER  # type: ignore
except Exception:
    LEGACY_LOADER = None


# =====================================================================
# パス関連
# =====================================================================
def get_default_downloads_dir() -> Path:
    home = Path.home()
    if os.name == "nt":
        up = os.environ.get("USERPROFILE")
        if up and (Path(up) / "Downloads").is_dir():
            return Path(up) / "Downloads"
    if (home / "Downloads").is_dir():
        return home / "Downloads"
    return home


# =====================================================================
# フォルダ監視
# =====================================================================
if WATCHDOG_AVAILABLE:
    class _SpectrumFileHandler(FileSystemEventHandler):
        def __init__(self, callback):
            self.callback = callback
            self.valid_ext = {".csv", ".txt"}

        def on_created(self, event):
            if not event.is_directory and Path(event.src_path).suffix.lower() in self.valid_ext:
                self.callback(Path(event.src_path), "created")

        def on_modified(self, event):
            if not event.is_directory and Path(event.src_path).suffix.lower() in self.valid_ext:
                self.callback(Path(event.src_path), "modified")

    def _start_observer(folder: Path, callback):
        h = _SpectrumFileHandler(callback)
        obs = Observer()
        obs.schedule(h, str(folder), recursive=False)
        obs.start()
        return obs
else:
    def _start_observer(folder: Path, callback):
        return None


def _file_change_callback(file_path: Path, event_type: str):
    st.session_state.setdefault("file_changes", [])
    st.session_state.file_changes.append(
        {"file_path": str(file_path), "event_type": event_type, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    )
    st.session_state["auto_update_trigger"] = time.time()


# =====================================================================
# 読み込み（フォールバック実装）
# =====================================================================
def _decode_and_read_csv(data_bytes: bytes) -> pd.DataFrame:
    """
    バイト列からロバストに DataFrame を作る。
    - 文字コードを順に試す
    - 区切り文字は sep=None で自動検出（失敗時にタブやセミコロンも試す）
    - 2列目までの数値行だけ抽出（ヘッダ/コメント除去）
    """
    encodings = ["utf-8", "utf-8-sig", "cp932", "shift_jis", "utf-16", "utf-16le", "utf-16be"]
    tried_errors = []

    for enc in encodings:
        try:
            text = data_bytes.decode(enc)
        except Exception as e:
            tried_errors.append(f"{enc}: {type(e).__name__}")
            continue

        for attempt in (
            dict(sep=None, engine="python"),  # sniff
            dict(sep="\t"),
            dict(sep=";"),
            dict(delim_whitespace=True),
        ):
            try:
                df = pd.read_csv(io.StringIO(text), **attempt)
                # 2列以上前提
                if df.shape[1] < 2:
                    continue

                # 先頭2列を数値化。数値でない行は落とす（ヘッダ・コメント除去）
                head2 = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce")
                mask = head2.notna().all(axis=1)
                df2 = df.loc[mask].copy()
                if df2.empty:
                    continue

                # 数値列として返す（1列目=波数, 2列目=強度）
                df2.iloc[:, 0] = pd.to_numeric(df2.iloc[:, 0], errors="coerce")
                df2.iloc[:, 1] = pd.to_numeric(df2.iloc[:, 1], errors="coerce")
                df2 = df2.dropna(subset=[df2.columns[0], df2.columns[1]])
                if df2.empty:
                    continue

                return df2.reset_index(drop=True)
            except Exception:
                continue

    raise ValueError("サポート外のフォーマット/エンコードです")


def _read_table_any(file_like_or_path) -> pd.DataFrame:
    # str/Path の場合はファイルから bytes を読み込む
    if isinstance(file_like_or_path, (str, Path)):
        with open(file_like_or_path, "rb") as f:
            data = f.read()
        return _decode_and_read_csv(data)

    # Streamlit UploadedFile or file-like の場合
    if hasattr(file_like_or_path, "getvalue"):
        data = file_like_or_path.getvalue()
        return _decode_and_read_csv(data)

    if hasattr(file_like_or_path, "read"):
        pos = file_like_or_path.tell() if hasattr(file_like_or_path, "tell") else None
        data = file_like_or_path.read()
        if pos is not None and hasattr(file_like_or_path, "seek"):
            file_like_or_path.seek(pos)
        return _decode_and_read_csv(data)

    raise ValueError("未知の入力タイプです")


def _process_file_to_arrays(
    file_like_or_path,
    start_wavenum: float,
    end_wavenum: float,
    movavg_window: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = _read_table_any(file_like_or_path)
    wavenum = df.iloc[:, 0].astype(float).to_numpy()
    y = df.iloc[:, 1].astype(float).to_numpy()

    # 範囲
    m = (wavenum >= start_wavenum) & (wavenum <= end_wavenum)
    wv = wavenum[m]
    ys = y[m]

    # ベースライン（min差し）
    y_bl = ys - np.min(ys)

    # 移動平均
    if movavg_window > 1:
        k = np.ones(movavg_window) / movavg_window
        y_ma = np.convolve(y_bl, k, mode="same")
    else:
        y_ma = y_bl
    return wv, ys, y_bl, y_ma


# =====================================================================
# 可視化
# =====================================================================
def _plot_overlaid(all_data: List[dict], key: str, title: str, Fsize: int = 14):
    fig, ax = plt.subplots(figsize=(10, 5))
    for d in all_data:
        ax.plot(d["wavenum"], d[key], label=d["label"])
    ax.set_xlabel("WaveNumber / cm-1", fontsize=Fsize)
    ax.set_ylabel("Intensity / a.u.", fontsize=Fsize)
    ax.set_title(title, fontsize=Fsize)
    ax.legend(title="Spectra")
    st.pyplot(fig)


def _build_interpolated_csv(all_data: List[dict], field: str) -> str:
    if not all_data:
        return ""
    min_w = min(d["wavenum"].min() for d in all_data)
    max_w = max(d["wavenum"].max() for d in all_data)
    npts = max(len(d["wavenum"]) for d in all_data)
    grid = np.linspace(min_w, max_w, npts)
    out = pd.DataFrame({"WaveNumber": grid})
    for d in all_data:
        out[d["label"]] = np.interp(grid, d["wavenum"], d[field])
    return out.to_csv(index=False, encoding="utf-8-sig")


def _show_reference_table():
    raman_data = {
        "ラマンシフト (cm⁻¹)": [
            "100–200", "150–450", "250–400", "290–330", "430–550", "450–550", "480–660", "500–700",
            "550–800", "630–790", "800–970", "1000–1250", "1300–1400", "1500–1600", "1600–1800",
            "2100–2250", "2800–3100", "3300–3500"
        ],
        "振動モード / 化学基": [
            "格子振動", "金属-酸素結合", "C-C アリファティック鎖", "Se-Se", "S-S", "Si-O-Si", "C-I",
            "C-Br", "C-Cl", "C-S", "C-O-C", "C=S", "CH₂/CH₃ 変角", "芳香族 C=C", "C=O",
            "C≡C / C≡N", "C-H (sp³, sp²)", "N-H / O-H"
        ],
        "強度": [
            "強い", "中〜弱", "強い", "強い", "強い", "強い", "強い", "強い", "強い", "中〜強",
            "中〜弱", "強い", "中〜弱", "強い", "中程度", "中〜強", "強い", "中程度"
        ],
    }
    st.subheader("（参考）ラマン分光の帰属表")
    st.table(pd.DataFrame(raman_data))


# =====================================================================
# メイン（legacy優先）
# =====================================================================
def spectrum_analysis_mode():
    st.header("ラマンスペクトル表示")

    # パラメータ
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    Fsize = 14

    start_wavenum = st.sidebar.number_input("波数（開始）", -200, 4800, pre_start_wavenum, 100)
    end_wavenum   = st.sidebar.number_input("波数（終了）", -200, 4800, pre_end_wavenum, 100)
    movavg_window = st.sidebar.number_input("移動平均ウィンドウ（奇数推奨）", 1, 101, 5, 2)

    # 既定フォルダ = Downloads
    default_folder = get_default_downloads_dir()
    folder_text = st.sidebar.text_input("フォルダパス（既定: Downloads）", value=str(default_folder))
    folder = Path(folder_text).expanduser()

    # フォルダ監視（ある場合のみ）
    if folder.is_dir() and WATCHDOG_AVAILABLE:
        if st.session_state.get("observer_started_for") != str(folder):
            try:
                obs = _start_observer(folder, _file_change_callback)
                st.session_state.observer_started_for = str(folder)
                st.session_state._observer = obs
                st.sidebar.info(f"監視中: {folder}")
            except Exception:
                st.sidebar.warning("フォルダ監視の開始に失敗しました")

    all_data: List[dict] = []

    # ===== 1) フォルダから読む =====
    if folder.is_dir():
        files = sorted(
            [p for p in folder.glob("*") if p.suffix.lower() in {".csv", ".txt"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            names = [p.name for p in files]
            selected = st.sidebar.multiselect("処理するファイルを選択", names, default=names, key="spectrum_selected_files")
            for name in selected:
                fpath = folder / name
                try:
                    if LEGACY_LOADER:
                        # common_utils と同じ挙動に寄せる（フェイクUploadedFile）
                        class _FakeUF:
                            def __init__(self, path: Path):
                                self.name = path.name
                                self._b = path.read_bytes()
                            def getvalue(self):
                                return self._b
                        fake = _FakeUF(fpath)
                        w, y, y_bl, y_ma, _ftype, fname = LEGACY_LOADER(fake, start_wavenum, end_wavenum, 1000/1e7, movavg_window)
                    else:
                        w, y, y_bl, y_ma = _process_file_to_arrays(str(fpath), start_wavenum, end_wavenum, movavg_window)
                        fname = fpath.stem
                    all_data.append({"wavenum": w, "raw": y, "baseline": y_bl, "moving": y_ma, "label": fname})
                except Exception as e:
                    st.warning(f"{name}: 読み込み失敗 — {e}")
        else:
            st.info("フォルダ内に処理可能なファイルがありません（.csv / .txt）")

    # ===== 2) アップローダ（フォルダが無い/空のときも使える）=====
    uploaded = st.file_uploader("CSV/TXT を選択（複数可）", type=["csv", "txt"], accept_multiple_files=True, key="spectrum_uploader")
    if uploaded:
        for uf in uploaded:
            try:
                if LEGACY_LOADER:
                    w, y, y_bl, y_ma, _ftype, fname = LEGACY_LOADER(uf, start_wavenum, end_wavenum, 1000/1e7, movavg_window)
                else:
                    buf = io.BytesIO(uf.getvalue())
                    w, y, y_bl, y_ma = _process_file_to_arrays(buf, start_wavenum, end_wavenum, movavg_window)
                    fname = Path(uf.name).stem
                all_data.append({"wavenum": w, "raw": y, "baseline": y_bl, "moving": y_ma, "label": fname})
            except Exception as e:
                st.warning(f"{uf.name}: 読み込み失敗 — {e}")

    # ===== 表示 =====
    if not all_data:
        return

    _plot_overlaid(all_data, "raw", "Raw Spectra", Fsize)
    st.download_button("Download Raw Spectra (CSV)", data=_build_interpolated_csv(all_data, "raw"),
                       file_name="raw_spectra.csv", mime="text/csv")

    _plot_overlaid(all_data, "baseline", "Baseline Removed", Fsize)
    st.download_button("Download Baseline Removed (CSV)", data=_build_interpolated_csv(all_data, "baseline"),
                       file_name="baseline_removed_spectra.csv", mime="text/csv")

    _plot_overlaid(all_data, "moving", "Baseline Removed + Moving Average", Fsize)
    st.download_button("Download Baseline+MovingAvg (CSV)", data=_build_interpolated_csv(all_data, "moving"),
                       file_name="baseline_removed_moving_avg_spectra.csv", mime="text/csv")

    _show_reference_table()
