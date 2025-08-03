# -*- coding: utf-8 -*-
"""
ピークAI解析モジュール（PDFレポート機能付き）
RAG機能とOpenAI APIを使用したラマンスペクトルの高度な解析
Enhanced with PDF report generation

Created on Wed Jun 11 15:56:04 2025
@author: Enhanced System
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import json
import pickle
import requests
import ssl
import urllib3
import glob
import base64
import io
from datetime import datetime
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from pathlib import Path

# PDF生成関連のインポート
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    PDF_GENERATION_AVAILABLE = True
    print("PDF generation libraries loaded successfully")
except ImportError as e:
    PDF_GENERATION_AVAILABLE = False
    print(f"PDF generation not available: {e}")
    st.warning("PDFレポート機能を使用するには、以下のライブラリが必要です：reportlab, Pillow")

# 他の既存インポート...
from common_utils import *
from peak_analysis_web import optimize_thresholds_via_gridsearch

# セキュリティモジュールのインポート（既存コードから）
try:
    from security_manager import (
        SecurityManager, 
        get_security_manager, 
        SecurityConfig, 
        SecurityException
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False
    st.warning("セキュリティモジュールが利用できません。基本機能のみ動作します。")

# Interactive plotting
try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None

# AI/RAG関連のインポート（既存コードから）
try:
    import PyPDF2
    import docx
    import openai
    import faiss
    from sentence_transformers import SentenceTransformer
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    st.warning("AI analysis features require additional packages: PyPDF2, docx, openai, faiss, sentence-transformers")

# [既存のクラスをここに含める - LLMConnector, RamanRAGSystem, RamanSpectrumAnalyzer等]
# 以下は新しいPDFレポート生成クラス

class RamanPDFReportGenerator:
    """ラマンスペクトル解析PDFレポート生成クラス"""
    
    def __init__(self):
        self.temp_dir = "./temp_pdf_assets"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 日本語フォントの設定を試行
        self.setup_japanese_font()
        
        # レポートスタイルの設定
        self.setup_styles()
    
    def setup_japanese_font(self):
        """日本語フォントの設定"""
        self.japanese_font_available = False
        
        try:
            # システムにある日本語フォントを探す
            font_paths = [
                # Windows
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/meiryo.ttc", 
                "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
                # macOS
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/NotoSansCJK.ttc",
                # Linux
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('JapaneseFont', font_path))
                        self.japanese_font_available = True
                        self.japanese_font_name = 'JapaneseFont'
                        break
                    except:
                        continue
            
            if not self.japanese_font_available:
                # フォールバック：Helvetica使用
                self.japanese_font_name = 'Helvetica'
                st.warning("日本語フォントが見つかりませんでした。英数字のみ正常に表示されます。")
                
        except Exception as e:
            self.japanese_font_name = 'Helvetica'
            st.warning(f"フォント設定エラー: {e}")
    
    def setup_styles(self):
        """PDFスタイルの設定"""
        self.styles = getSampleStyleSheet()
        
        # カスタムスタイルの追加
        self.styles.add(ParagraphStyle(
            name='JapaneseTitle',
            parent=self.styles['Title'],
            fontName=self.japanese_font_name,
            fontSize=18,
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='JapaneseHeading',
            parent=self.styles['Heading1'],
            fontName=self.japanese_font_name,
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='JapaneseNormal',
            parent=self.styles['Normal'],
            fontName=self.japanese_font_name,
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='JapaneseCode',
            parent=self.styles['Code'],
            fontName='Courier',
            fontSize=8,
            textColor=colors.darkgreen
        ))
    
    def plotly_to_image(self, fig, filename, width=800, height=600, format='png'):
        """PlotlyグラフをPNG画像に変換"""
        try:
            # 画像として保存
            img_path = os.path.join(self.temp_dir, f"{filename}.{format}")
            
            # Plotlyの設定
            fig.update_layout(
                width=width,
                height=height,
                font=dict(size=10),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # 画像保存（kaleido使用を試行、フォールバックでorchestrator）
            try:
                pio.write_image(fig, img_path, format=format, width=width, height=height, scale=2)
            except Exception as e:
                st.warning(f"Kaleidoエンジン使用失敗。HTMLエクスポートを試行: {e}")
                # HTMLとして保存し、後でスクリーンショット的に処理
                html_path = os.path.join(self.temp_dir, f"{filename}.html")
                fig.write_html(html_path)
                # 簡易的なプレースホルダー画像を作成
                self._create_placeholder_image(img_path, width, height, f"Graph: {filename}")
            
            return img_path
            
        except Exception as e:
            st.error(f"グラフ画像変換エラー: {e}")
            # プレースホルダー画像を作成
            placeholder_path = os.path.join(self.temp_dir, f"{filename}_placeholder.png")
            self._create_placeholder_image(placeholder_path, width, height, f"Graph Error: {filename}")
            return placeholder_path
    
    def _create_placeholder_image(self, path, width, height, text):
        """プレースホルダー画像を作成"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 白背景の画像を作成
            img = PILImage.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # テキストを中央に描画
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill='black', font=font)
            
            # 枠線を描画
            draw.rectangle([0, 0, width-1, height-1], outline='gray')
            
            img.save(path)
            
        except Exception as e:
            st.warning(f"プレースホルダー画像作成エラー: {e}")
    
    def generate_comprehensive_pdf_report(
        self, 
        file_key: str,
        peak_data: List[Dict],
        analysis_result: str,
        peak_summary_df: pd.DataFrame,
        plotly_figure: go.Figure,
        relevant_docs: List[Dict] = None,
        user_hint: str = None,
        qa_history: List[Dict] = None
    ) -> bytes:
        """包括的なPDFレポートを生成"""
        
        if not PDF_GENERATION_AVAILABLE:
            raise Exception("PDF生成ライブラリが利用できません")
        
        # PDFファイルをメモリ上に作成
        pdf_buffer = io.BytesIO()
        
        try:
            # SimpleDocTemplateでPDF作成
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            # コンテンツリストを作成
            story = []
            
            # 1. タイトルページ
            story.extend(self._create_title_page(file_key))
            
            # 2. 実行サマリー
            story.extend(self._create_executive_summary(peak_data, analysis_result))
            
            # 3. グラフセクション
            if plotly_figure:
                story.extend(self._create_graph_section(plotly_figure, file_key))
            
            # 4. ピーク詳細テーブル
            story.extend(self._create_peak_details_section(peak_summary_df, peak_data))
            
            # 5. AI解析結果
            story.extend(self._create_ai_analysis_section(analysis_result))
            
            # 6. 参考文献（利用可能な場合）
            if relevant_docs:
                story.extend(self._create_references_section(relevant_docs))
            
            # 7. 補足情報
            if user_hint:
                story.extend(self._create_additional_info_section(user_hint))
            
            # 8. Q&A履歴（利用可能な場合）
            if qa_history:
                story.extend(self._create_qa_section(qa_history))
            
            # 9. 付録・メタデータ
            story.extend(self._create_appendix_section())
            
            # PDFを構築
            doc.build(story)
            
            # バイト配列として返す
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            st.error(f"PDFレポート生成エラー: {e}")
            raise e
    
    def _create_title_page(self, file_key: str) -> List:
        """タイトルページを作成"""
        content = []
        
        # メインタイトル
        title = Paragraph(
            "ラマンスペクトル解析レポート",
            self.styles['JapaneseTitle']
        )
        content.append(title)
        content.append(Spacer(1, 0.5*inch))
        
        # ファイル情報
        file_info = f"""
        <b>解析対象ファイル:</b> {file_key}<br/>
        <b>レポート生成日時:</b> {datetime.now().strftime('%Y年%m月%d日 %H時%M分')}<br/>
        <b>システム:</b> RamanEye AI Analysis System<br/>
        <b>バージョン:</b> 2.0 (Enhanced Security Edition)
        """
        
        content.append(Paragraph(file_info, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.5*inch))
        
        # 免責事項
        disclaimer = """
        <b>【重要】本レポートについて</b><br/>
        本レポートはAIによる自動解析結果を含んでいます。
        結果の解釈および活用については、専門家による検証を推奨します。
        測定条件、サンプル前処理、装置較正等の要因が結果に影響する可能性があります。
        """
        
        content.append(Paragraph(disclaimer, self.styles['JapaneseNormal']))
        content.append(PageBreak())
        
        return content
    
    def _create_executive_summary(self, peak_data: List[Dict], analysis_result: str) -> List:
        """実行サマリーを作成"""
        content = []
        
        content.append(Paragraph("実行サマリー", self.styles['JapaneseHeading']))
        
        # ピーク統計
        total_peaks = len(peak_data)
        auto_peaks = len([p for p in peak_data if p.get('type') == 'auto'])
        manual_peaks = len([p for p in peak_data if p.get('type') == 'manual'])
        
        summary_text = f"""
        <b>検出ピーク総数:</b> {total_peaks}<br/>
        <b>自動検出:</b> {auto_peaks} ピーク<br/>
        <b>手動追加:</b> {manual_peaks} ピーク<br/>
        <br/>
        <b>主要検出範囲:</b> {min([p['wavenumber'] for p in peak_data]):.0f} - {max([p['wavenumber'] for p in peak_data]):.0f} cm⁻¹
        """
        
        content.append(Paragraph(summary_text, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.3*inch))
        
        # AI解析結果の要約（最初の200文字）
        analysis_summary = analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
        content.append(Paragraph("<b>AI解析結果要約:</b>", self.styles['JapaneseNormal']))
        content.append(Paragraph(analysis_summary, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_graph_section(self, plotly_figure: go.Figure, file_key: str) -> List:
        """グラフセクションを作成"""
        content = []
        
        content.append(Paragraph("スペクトルおよびピーク検出結果", self.styles['JapaneseHeading']))
        
        try:
            # Plotlyグラフを画像に変換
            img_path = self.plotly_to_image(plotly_figure, f"spectrum_{file_key}", width=1000, height=800)
            
            # 画像をPDFに追加
            if os.path.exists(img_path):
                img = Image(img_path, width=7*inch, height=5.6*inch)
                content.append(img)
                content.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            error_text = f"グラフ表示エラー: {e}"
            content.append(Paragraph(error_text, self.styles['JapaneseNormal']))
        
        # グラフの説明
        graph_description = """
        上図はラマンスペクトルとピーク検出結果を示しています。
        赤い点は検出されたピーク、緑の星印は手動で追加されたピークを表示しています。
        下部のプロットは2次微分スペクトルとピークのProminence値を示しています。
        """
        
        content.append(Paragraph(graph_description, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_peak_details_section(self, peak_summary_df: pd.DataFrame, peak_data: List[Dict]) -> List:
        """ピーク詳細セクションを作成"""
        content = []
        
        content.append(Paragraph("検出ピーク詳細", self.styles['JapaneseHeading']))
        
        # DataFrameをテーブルに変換
        table_data = [peak_summary_df.columns.tolist()]  # ヘッダー
        for _, row in peak_summary_df.iterrows():
            table_data.append(row.tolist())
        
        # テーブルスタイルの設定
        table = Table(table_data, colWidths=[1*inch, 1.5*inch, 1.2*inch, 1.2*inch, 1.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), self.japanese_font_name),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), self.japanese_font_name),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        content.append(table)
        content.append(Spacer(1, 0.3*inch))
        
        return content
    
    def _create_ai_analysis_section(self, analysis_result: str) -> List:
        """AI解析結果セクションを作成"""
        content = []
        
        content.append(Paragraph("AI解析結果", self.styles['JapaneseHeading']))
        
        # 長いテキストを段落に分割
        paragraphs = analysis_result.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
                # マークダウン形式の簡単な変換
                para = para.replace('**', '<b>').replace('**', '</b>')
                para = para.replace('*', '<i>').replace('*', '</i>')
                
                content.append(Paragraph(para.strip(), self.styles['JapaneseNormal']))
                content.append(Spacer(1, 0.1*inch))
        
        return content
    
    def _create_references_section(self, relevant_docs: List[Dict]) -> List:
        """参考文献セクションを作成"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("参考文献", self.styles['JapaneseHeading']))
        
        for i, doc in enumerate(relevant_docs, 1):
            filename = doc.get('metadata', {}).get('filename', f'文献{i}')
            similarity = doc.get('similarity_score', 0.0)
            preview = doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', '')
            
            ref_text = f"""
            <b>{i}. {filename}</b><br/>
            類似度: {similarity:.3f}<br/>
            内容抜粋: {preview}<br/>
            """
            
            content.append(Paragraph(ref_text, self.styles['JapaneseNormal']))
            content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_additional_info_section(self, user_hint: str) -> List:
        """補足情報セクションを作成"""
        content = []
        
        content.append(Paragraph("補足情報", self.styles['JapaneseHeading']))
        content.append(Paragraph(f"ユーザー提供ヒント: {user_hint}", self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_qa_section(self, qa_history: List[Dict]) -> List:
        """Q&Aセクションを作成"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("質問応答履歴", self.styles['JapaneseHeading']))
        
        for i, qa in enumerate(qa_history, 1):
            qa_text = f"""
            <b>質問{i}:</b> {qa['question']}<br/>
            <b>回答{i}:</b> {qa['answer']}<br/>
            <i>日時: {qa['timestamp']}</i><br/>
            """
            
            content.append(Paragraph(qa_text, self.styles['JapaneseNormal']))
            content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_appendix_section(self) -> List:
        """付録セクションを作成"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("付録", self.styles['JapaneseHeading']))
        
        # システム情報
        system_info = f"""
        <b>システム情報:</b><br/>
        生成日時: {datetime.now().isoformat()}<br/>
        レポート形式: PDF (ReportLab生成)<br/>
        AI分析エンジン: OpenAI GPT Model<br/>
        セキュリティ機能: {'有効' if SECURITY_AVAILABLE else '無効'}<br/>
        """
        
        content.append(Paragraph(system_info, self.styles['JapaneseNormal']))
        
        return content
    
    def cleanup_temp_files(self):
        """一時ファイルをクリーンアップ"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"一時ファイルクリーンアップエラー: {e}")


# 既存の関数を修正: perform_ai_analysis
def perform_ai_analysis(file_key, final_peak_data, user_hint, llm_connector, peak_summary_df):
    """AI解析を実行（PDF機能付き）"""
    with st.spinner("AI解析中です。しばらくお待ちください..."):
        analysis_report = None
        start_time = time.time()

        try:
            analyzer = RamanSpectrumAnalyzer()

            # 関連文献を検索
            search_terms = ' '.join([f"{p['wavenumber']:.0f}cm-1" for p in final_peak_data[:5]])
            search_query = f"ラマンスペクトロスコピー ピーク {search_terms}"
            relevant_docs = st.session_state.rag_system.search_relevant_documents(search_query, top_k=5)

            # AIへのプロンプトを生成
            analysis_prompt = analyzer.generate_analysis_prompt(
                peak_data=final_peak_data,
                relevant_docs=relevant_docs,
                user_hint=user_hint
            )
            
            # OpenAI APIで解析を実行
            st.success("OpenAI APIの応答（リアルタイム表示）")
            full_response = llm_connector.generate_analysis(analysis_prompt)

            # 処理時間の表示
            elapsed = time.time() - start_time
            st.info(f"解析にかかった時間: {elapsed:.2f} 秒")

            # 解析結果をセッションに保存
            model_info = f"OpenAI ({llm_connector.selected_model})"
            st.session_state[f"{file_key}_ai_analysis"] = {
                'analysis': full_response,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': model_info,
                'analysis_context': full_response,
                'peak_data': final_peak_data,
                'peak_summary_df': peak_summary_df,
                'relevant_docs': relevant_docs,
                'user_hint': user_hint
            }

            # テキストレポート生成（既存）
            analysis_report = f"""ラマンスペクトル解析レポート
ファイル名: {file_key}
解析日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
使用モデル: {model_info}

=== 検出ピーク情報 ===
{peak_summary_df.to_string(index=False)}

=== AI解析結果 ===
{full_response}

=== 参照文献 ===
"""
            for i, doc in enumerate(relevant_docs, 1):
                analysis_report += f"{i}. {doc['metadata']['filename']}（類似度: {doc['similarity_score']:.3f}）\n"

            # ダウンロードボタンのセクション
            st.subheader("📥 レポートダウンロード")
            
            col1, col2 = st.columns(2)
            
            # テキストレポートダウンロード
            with col1:
                st.download_button(
                    label="📄 テキストレポートをダウンロード",
                    data=analysis_report,
                    file_name=f"raman_analysis_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key=f"download_text_report_{file_key}"
                )
            
            # PDFレポートダウンロード
            with col2:
                if PDF_GENERATION_AVAILABLE:
                    if st.button(f"📊 PDFレポートを生成", key=f"generate_pdf_{file_key}"):
                        generate_pdf_report(file_key, final_peak_data, full_response, peak_summary_df, relevant_docs, user_hint)
                else:
                    st.info("PDFレポート機能は利用できません（必要ライブラリ未インストール）")

        except Exception as e:
            st.error(f"AI解析中にエラーが発生しました: {str(e)}")
            st.info("OpenAI APIの接続を確認してください。有効なAPIキーが設定されていることを確認してください。")


def generate_pdf_report(file_key, final_peak_data, analysis_result, peak_summary_df, relevant_docs, user_hint):
    """PDFレポート生成の実行"""
    
    try:
        with st.spinner("PDFレポートを生成中..."):
            # PDFレポートジェネレーターを初期化
            pdf_generator = RamanPDFReportGenerator()
            
            # 現在表示されているPlotlyグラフを取得
            # 注意: 実際の実装では、グラフデータを適切に渡す必要があります
            plotly_figure = st.session_state.get(f"{file_key}_plotly_figure", None)
            
            # Q&A履歴を取得
            qa_history = st.session_state.get(f"{file_key}_qa_history", [])
            
            # PDFを生成
            pdf_bytes = pdf_generator.generate_comprehensive_pdf_report(
                file_key=file_key,
                peak_data=final_peak_data,
                analysis_result=analysis_result,
                peak_summary_df=peak_summary_df,
                plotly_figure=plotly_figure,
                relevant_docs=relevant_docs,
                user_hint=user_hint,
                qa_history=qa_history
            )
            
            # 一時ファイルをクリーンアップ
            pdf_generator.cleanup_temp_files()
            
            # ダウンロードボタンを表示
            st.download_button(
                label="📊 PDFレポートをダウンロード",
                data=pdf_bytes,
                file_name=f"raman_comprehensive_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                key=f"download_pdf_report_{file_key}"
            )
            
            st.success("✅ PDFレポートが正常に生成されました！")
            
    except Exception as e:
        st.error(f"PDFレポート生成エラー: {str(e)}")
        st.info("システム管理者に連絡してください。")


# 既存の関数にPlotlyグラフ保存機能を追加
def render_interactive_plot(result, file_key, spectrum_type):
    """インタラクティブプロットを描画（PDF用に保存機能付き）"""
    # [既存のプロット生成コード...]
    
    # 除外を反映したピークインデックス
    filtered_peaks = [
        i for i in result['detected_peaks']
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result['detected_peaks'], result['detected_prominences'])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            f'{file_key} - {spectrum_type}',
            f'{file_key} - 微分スペクトル比較',
            f'{file_key} - Prominence vs 波数'
        ],
        vertical_spacing=0.07,
        row_heights=[0.4, 0.3, 0.3]
    )

    # [既存のプロット描画コード...]
    # スペクトル描画
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'],
            y=result['spectrum'],
            mode='lines',
            name=spectrum_type,
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )

    # 有効なピーク
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

    # 手動ピーク
    for x, y in st.session_state[f"{file_key}_manual_peaks"]:
        fig.add_trace(
            go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(color='green', size=10, symbol='star'),
                text=["手動"],
                textposition='top center',
                name="手動ピーク",
                showlegend=False
            ),
            row=1, col=1
        )

    # 2次微分
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

    # Prominenceプロット
    fig.add_trace(
        go.Scatter(
            x=result['wavenum'][result['all_peaks']],
            y=result['all_prominences'],
            mode='markers',
            name='全ピークのProminence',
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
                name='有効なProminence',
                marker=dict(color='red', size=7, symbol='circle')
            ),
            row=3, col=1
        )

    fig.update_layout(height=800, margin=dict(t=80, b=40))
    fig.update_xaxes(title_text="波数 (cm⁻¹)", row=3, col=1)
    fig.update_yaxes(title_text="強度", row=1, col=1)
    fig.update_yaxes(title_text="微分値", row=2, col=1)
    
    # PDFレポート用にPlotlyグラフを保存
    st.session_state[f"{file_key}_plotly_figure"] = fig
    
    # [既存のクリック処理コード...]
    if plotly_events:
        # [既存のクリック処理コード...]
        pass
    else:
        st.plotly_chart(fig, use_container_width=True)


# 既存のメイン関数に変更なし（peak_ai_analysis_mode等）
