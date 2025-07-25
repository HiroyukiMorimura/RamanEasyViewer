def render_interactive_plot(result, file_key, spectrum_type):
    """
    インタラクティブプロットの描画（修正版）
    """

    # ---- セッション初期化 ----
    if f"{file_key}_excluded_peaks" not in st.session_state:
        st.session_state[f"{file_key}_excluded_peaks"] = set()
    if f"{file_key}_manual_peaks" not in st.session_state:
        st.session_state[f"{file_key}_manual_peaks"] = []

    # ---- フィルタリング済みピーク配列 ----
    filtered_peaks = [
        i for i in result["detected_peaks"]
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result["detected_peaks"], result["detected_prominences"])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    
    # =========================================================
    # ① メインスペクトル用のFigure作成
    # =========================================================
    spec_x = np.asarray(result["wavenum"], dtype=float)
    spec_y = np.asarray(result["spectrum"], dtype=float)
    
    fig_main = go.Figure()

    # メインスペクトル
    fig_main.add_trace(
        go.Scatter(
            x=spec_x,
            y=spec_y,
            mode="lines+markers",
            name=spectrum_type,
            line=dict(color='blue', width=2),
            marker=dict(size=6, color='rgba(0,0,0,0)'),
            connectgaps=True
        )
    )
    
    # 自動検出ピーク（有効）
    if filtered_peaks:
        fig_main.add_trace(
            go.Scatter(
                x=spec_x[filtered_peaks],
                y=spec_y[filtered_peaks],
                mode="markers",
                name="検出ピーク（有効）",
                marker=dict(size=8, symbol="circle", color='red')
            )
        )
    
    # 除外ピーク
    excl = list(st.session_state[f"{file_key}_excluded_peaks"])
    if excl:
        fig_main.add_trace(
            go.Scatter(
                x=spec_x[excl],
                y=spec_y[excl],
                mode="markers",
                name="除外ピーク",
                marker=dict(symbol="x", size=8, color='gray')
            )
        )

    # 手動ピーク
    for x_manual, y_manual in st.session_state[f"{file_key}_manual_peaks"]:
        fig_main.add_trace(
            go.Scatter(
                x=[x_manual],
                y=[y_manual],
                mode="markers+text",
                text=["手動"],
                textposition="top center",
                name="手動ピーク",
                marker=dict(symbol="star", size=10, color='green'),
                showlegend=False
            )
        )
    
    fig_main.update_layout(
        height=360, 
        margin=dict(t=40, b=40),
        title=f"{file_key} - {spectrum_type}"
    )
    fig_main.update_xaxes(title_text="波数 (cm⁻¹)")
    fig_main.update_yaxes(title_text="Intensity (a.u.)")
    
    # =========================================================
    # ② インタラクティブ処理（修正版）
    # =========================================================
    if plotly_events is not None:
        event_key = f"{file_key}_click_event"
        
        # plotly_eventsを使用（これが実際のグラフを描画する）
        clicked_points = plotly_events(
            fig_main,
            click_event=True,
            select_event=False,  # selectは不要なのでFalseに
            hover_event=False,
            override_height=360,
            override_width="100%",  # 幅も指定
            key=event_key
        )
        
        # クリックイベント処理
        if clicked_points and len(clicked_points) > 0:
            pt = clicked_points[-1]  # 最新のクリック
            
            # デバウンス処理
            ev_id = f"{pt['curveNumber']}-{round(pt['x'], 3)}-{round(pt['y'], 3)}"
            if st.session_state.get(f"{event_key}_last") != ev_id:
                st.session_state[f"{event_key}_last"] = ev_id

                x_clicked = float(pt["x"])
                y_clicked = float(pt["y"])
                idx = int(np.argmin(np.abs(result["wavenum"] - x_clicked)))

                # トレース判定
                curve_num = pt["curveNumber"]
                if curve_num < len(fig_main.data):
                    trace_name = fig_main.data[curve_num].name
                    is_main_trace = (trace_name == spectrum_type)

                    if is_main_trace:
                        # 自動検出ピークならトグル
                        if idx in result["detected_peaks"]:
                            excl_set = st.session_state[f"{file_key}_excluded_peaks"]
                            if idx in excl_set:
                                excl_set.remove(idx)
                                st.success(f"ピーク {result['wavenum'][idx]:.1f} cm⁻¹ を有効にしました")
                            else:
                                excl_set.add(idx)
                                st.info(f"ピーク {result['wavenum'][idx]:.1f} cm⁻¹ を除外しました")
                            st.rerun()
                        else:
                            # 手動ピーク追加（重複チェック）
                            manual_peaks = st.session_state[f"{file_key}_manual_peaks"]
                            if not any(abs(px - x_clicked) < 5.0 for px, _ in manual_peaks):
                                manual_peaks.append((x_clicked, y_clicked))
                                st.success(f"手動ピーク {x_clicked:.1f} cm⁻¹ を追加しました")
                                st.rerun()

    else:
        # plotly_eventsが使用できない場合の代替表示
        st.plotly_chart(fig_main, use_container_width=True)
        st.warning("インタラクティブ機能を使用するには 'streamlit_plotly_events' をインストールしてください")
        
        # 手動でのピーク追加UI
        with st.expander("手動ピーク追加"):
            col1, col2 = st.columns(2)
            with col1:
                manual_x = st.number_input("追加する波数 (cm⁻¹)", 
                                         min_value=float(spec_x.min()), 
                                         max_value=float(spec_x.max()),
                                         key=f"manual_x_{file_key}")
            with col2:
                if st.button("ピーク追加", key=f"add_peak_{file_key}"):
                    idx = np.argmin(np.abs(spec_x - manual_x))
                    manual_y = spec_y[idx]
                    st.session_state[f"{file_key}_manual_peaks"].append((manual_x, manual_y))
                    st.rerun()
    
    # =========================================================
    # ③ 2次微分とProminenceの表示（変更なし）
    # =========================================================
    fig_sub = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=["微分スペクトル", "Prominence vs 波数"],
        vertical_spacing=0.08,
        row_heights=[0.45, 0.55]
    )

    # 2次微分
    fig_sub.add_trace(
        go.Scatter(
            x=result["wavenum"],
            y=result["second_derivative"],
            mode="lines",
            name="2次微分",
            line=dict(color='purple', width=1)
        ),
        row=1, col=1
    )
    fig_sub.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=1)

    # 全ピークの卓立度
    fig_sub.add_trace(
        go.Scatter(
            x=result["wavenum"][result["all_peaks"]],
            y=result["all_prominences"],
            mode="markers",
            name="全ピークの卓立度",
            marker=dict(size=4, color='orange')
        ),
        row=2, col=1
    )
    
    # 有効ピークの卓立度
    if filtered_peaks:
        fig_sub.add_trace(
            go.Scatter(
                x=result["wavenum"][filtered_peaks],
                y=filtered_prominences,
                mode="markers",
                name="有効な卓立度",
                marker=dict(symbol="circle", size=7, color='red')
            ),
            row=2, col=1
        )

    fig_sub.update_layout(height=470, margin=dict(t=60, b=60))
    fig_sub.update_xaxes(title_text="波数 (cm⁻¹)", row=2, col=1)
    fig_sub.update_yaxes(title_text="2次微分", row=1, col=1)
    fig_sub.update_yaxes(title_text="Prominence", row=2, col=1)
    
    st.plotly_chart(fig_sub, use_container_width=True)

    # 手動ピーク情報表示
    render_manual_peak_info(result, file_key)
