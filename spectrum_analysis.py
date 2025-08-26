# -*- coding: utf-8 -*-
"""
スペクトル解析モジュール
"""

from __future__ import annotations

import os
import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 任意依存
try:
    import seaborn as sns  # type: ignore
    plt.style.use("default")
    sns.set_palette("husl")
except Exception:
    plt.style.use("default")

# watchdog は任意
try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
    WATCHDOG_AVAILABLE = True
except Exception:
    WATCHDOG_AVAILABLE = False

# 既存ユーティリティ（process_spectrum_file を使うため必須）
from common_utils import *  # noqa: F401,F403


# ===== Downloads 推定 =====
def _default_downloads() -> Path:
    home = Path.home()
    if os.name == "nt":
        up = os.environ.get("USERPROFILE")
        if up and (Path(up) / "Downloads").exists():
            return Path(up) / "Downloads"
    p = home / "Downloads"
    return p if p.exists() else home


# ===== 監視（任意） =====
if WATCHDOG_AVAILABLE:
    class _Handler(FileSystemEventHandler):
        def __init__(self, folder: Path):
            self.folder = folder

        def _is_target(self, p: str) -> bool:
            return Path(p).suffix.lower() in {".csv", ".txt"}

        def on_created(self, event):
            if not event.is_directory and self._is_target(event.src_path):
                st.session_state["__spectra_reload__"] = True

        def on_modified(self, event):
            if not event.is_directory and self._is_target(event.src_path):
                st.session_state["__spectra_reload__"] = True

    def _start_watch(folder: Path):
        try:
            obs = Observer()
            obs.schedule(_Handler(folder), str(folder), recursive=False)
            obs.start()
            st.session_state["__observer__"] = obs
            st.session_state["__watching__"] = str(folder)
        except Exception:
            st.sidebar.warning("フォルダ監視の開始に失敗しました")
else:
    def _start_watch(folder: Path):
        return None


# ===== 表示補助 =====
def _plot_overlaid(all_data: List[dict], key: str, title: str, Fsize: int = 14):
    fig, ax = plt.subplots(figsize=(10, 5))
    for d in all_data:
        ax.plot(d["wavenum"], d[key], label=d["file_name"])
    ax.set_xlabel("WaveNumber / cm-1", fontsize=Fsize)
    ax.set_ylabel("Intensity / a.u.", fontsize=Fsize)
    ax.set_title(title, fontsize=Fsize)
    ax.legend(title="Spectra")
    st.pyplot(fig)


def create_interpolated_csv(all_data, spectrum_type):
    if not all_data:
        return ""
    min_w = min(d["wavenum"].min() for d in all_data)
    max_w = max(d["wavenum"].max() for d in all_data)
    max_pts = max(len(d["wavenum"]) for d in all_data)
    grid = np.linspace(min_w, max_w, max_pts)
    df = pd.DataFrame({"WaveNumber": grid})
    for d in all_data:
        df[d["file_name"]] = np.interp(grid, d["wavenum"], d[spectrum_type])
    return df.to_csv(index=False, encoding="utf-8-sig")


def display_raman_correlation_table():
    raman_data = {
        "ラマンシフト (cm⁻¹)": [
            "100–200", "150–450", "250–400", "290–330", "430–550", "450–550", "480–660", "500–700",
            "550–800", "630–790", "800–970", "1000–1250", "1300–1400", "1500–1600", "1600–1800", "2100–2250",
            "2800–3100", "3300–3500"
        ],
        "振動モード / 化学基": [
            "格子振動", "金属-酸素結合", "C-C アリファティック鎖", "Se-Se", "S-S",
            "Si-O-Si", "C-I", "C-Br", "C-Cl", "C-S", "C-O-C", "C=S", "CH₂/CH₃(変角)",
            "芳香族 C=C", "C=O", "C≡C/C≡N", "C-H(sp³/sp²)", "N-H/O-H"
        ],
        "強度": [
            "強い", "中〜弱", "強い", "強い", "強い", "強い", "強い", "強い", "強い", "中〜強", "中〜弱", "強い",
            "中〜弱", "強い", "中程度", "中〜強", "強い", "中程度"
        ],
    }
    st.subheader("（参考）ラマン分光の帰属表")
    st.table(pd.DataFrame(raman_data))


# ===== メイン =====
def spectrum_analysis_mode():
    st.header("ラマンスペクトル表示")

    # UI パラメータ
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    Fsize = 14

    start_wavenum = st.sidebar.number_input("波数（開始）", -200, 4800, pre_start_wavenum, 100)
    end_wavenum   = st.sidebar.number_input("波数（終了）", -200, 4800, pre_end_wavenum, 100)

    dssn_th = st.sidebar.number_input("ベースラインパラメーター", 1, 10000, 1000, 1)
    dssn_th = dssn_th / 10000000  # 旧仕様踏襲

    savgol_wsize = st.sidebar.number_input("移動平均ウィンドウ（奇数推奨）", 1, 35, 5, 2, key="unique_savgol_wsize_key")

    # 既定: Downloads
    default_folder = _default_downloads()
    folder_text = st.sidebar.text_input("フォルダパス（既定: Downloads）", value=str(default_folder))
    folder = Path(folder_text).expanduser()

    all_data: List[dict] = []

    # ===== フォルダ監視＆一括処理（ファイル→ process_spectrum_file）=====
    if folder.exists() and folder.is_dir():
        # 監視は初回のみ
        if WATCHDOG_AVAILABLE:
            if st.session_state.get("__watching__") != str(folder):
                _start_watch(folder)

        # 対象列挙
        files = sorted(
            [p for p in folder.glob("*") if p.suffix.lower() in {".csv", ".txt"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        names = [p.name for p in files]
        if names:
            selected = st.sidebar.multiselect("処理するファイルを選択", names, default=names, key="selected_files_in_folder")
            for name in selected:
                fpath = folder / name
                try:
                    # ★ここがポイント：ローカルファイルも process_spectrum_file へ
                    with open(fpath, "rb") as fh:
                        result = process_spectrum_file(
                            fh, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                        )
                    wv, sp, bs, mv, ftype, fname = result
                    if wv is None:
                        st.warning(f"{name}: 読み込み失敗")
                        continue
                    all_data.append({
                        "wavenum": wv,
                        "raw_spectrum": sp,
                        "baseline_removed": bs,
                        "moving_avg": mv,
                        "file_name": fname,
                    })
                except Exception as e:
                    st.warning(f"{name}: 読み込み失敗 — {e}")
        else:
            st.info("フォルダ内に処理可能なファイルがありません（.csv / .txt）")
    else:
        st.warning("指定フォルダが存在しません。ファイルをアップロードしてください。")

    # ===== アップロード（UploadedFile → process_spectrum_file）=====
    uploaded = st.file_uploader("CSV/TXT を選択（複数可）", type=["csv", "txt"], accept_multiple_files=True, key="mv_uploader")
    if uploaded:
        for uf in uploaded:
            try:
                # UploadedFile はそのまま process_spectrum_file に渡す
                result = process_spectrum_file(
                    uf, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                wv, sp, bs, mv, ftype, fname = result
                if wv is None:
                    st.warning(f"{uf.name}: 読み込み失敗")
                    continue
                all_data.append({
                    "wavenum": wv,
                    "raw_spectrum": sp,
                    "baseline_removed": bs,
                    "moving_avg": mv,
                    "file_name": fname,
                })
            except Exception as e:
                st.warning(f"{uf.name}: 読み込み失敗 — {e}")

    # ===== 表示・エクスポート =====
    if not all_data:
        return

    # 色付けは Matplotlib デフォルトに任せる（順序で色が付く）
    def _rename(dkey):
        return {"raw_spectrum":"Raw Spectra", "baseline_removed":"Baseline Removed", "moving_avg":"Baseline Removed + Moving Average"}[dkey]

    for key in ("raw_spectrum", "baseline_removed", "moving_avg"):
        _plot_overlaid(all_data, key, _rename(key), Fsize)
        st.download_button(
            label=f"Download {_rename(key)} as CSV",
            data=create_interpolated_csv(all_data, key),
            file_name=f"{key}.csv",
            mime="text/csv",
        )

    display_raman_correlation_table()
