def render_interactive_plot(result, file_key, spectrum_type):
    """インタラクティブプロットを描画（peak_analysis_web.pyと同じ方式）"""
    st.subheader(f"📊 {file_key} - {spectrum_type}")
    
    # ---- 手動制御UI（peak_analysis_web.pyから移植） ----
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

    # ---- フィルタリング済みピーク配列（peak_analysis_web.pyと同じ） ----
    filtered_peaks = [
        i for i in result["detected_peaks"]
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result["detected_peaks"], result["detected_prominences"])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    
    # ---- 静的プロット描画（peak_analysis_web.pyから完全移植） ----
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

    # 手動ピーク（peak_analysis_web.pyと同じ処理）
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
    
    # PDFレポート用にPlotlyグラフを保存
    st.session_state[f"{file_key}_plotly_figure"] = fig
    
    # グラフ表示（peak_analysis_web.pyと同じ）
    st.plotly_chart(fig, use_container_width=True)
