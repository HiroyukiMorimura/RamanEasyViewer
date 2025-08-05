# -*- coding: utf-8 -*-
"""
ピークAI解析モジュール
RAG機能とOpenAI APIを使用したラマンスペクトルの高度な解析
Enhanced with comprehensive features and PDF report generation

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
import warnings
from datetime import datetime
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from pathlib import Path
from common_utils import *
from peak_analysis_web import optimize_thresholds_via_gridsearch

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.signal._peak_finding')

# システムエラーハンドリング
def handle_system_warnings():
    """システム警告とエラーのハンドリング"""
    try:
        # inotify制限の確認
        import subprocess
        result = subprocess.run(['cat', '/proc/sys/fs/inotify/max_user_instances'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            max_instances = int(result.stdout.strip())
            if max_instances < 512:
                st.sidebar.warning(f"⚠️ システム制限: inotify instances = {max_instances}. "
                                 f"ファイル監視機能が制限される可能性があります。")
    except:
        pass  # システム情報取得失敗は無視

# Streamlit設定の最適化（inotify制限対策）
if hasattr(st, '_config'):
    try:
        # ファイル監視を最小限に抑制
        st._config.set_option('server.fileWatcherType', 'none')
        st._config.set_option('server.runOnSave', False)
    except:
        pass  # 設定変更に失敗しても続行

# 環境変数でのStreamlit設定（システムレベル対策の提案）
def suggest_system_optimization():
    """システム最適化の提案を表示"""
    if os.path.exists('/proc/sys/fs/inotify/max_user_instances'):
        with st.sidebar.expander("⚙️ システム最適化のヒント", expanded=False):
            st.markdown("""
            **Linux環境でのinotify制限対策:**
            
            ```bash
            # 一時的な増加
            echo 512 | sudo tee /proc/sys/fs/inotify/max_user_instances
            
            # 永続的な設定
            echo 'fs.inotify.max_user_instances=512' | sudo tee -a /etc/sysctl.conf
            sudo sysctl -p
            ```
            
            **Streamlit環境変数設定:**
            ```bash
            export STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
            export STREAMLIT_SERVER_RUN_ON_SAVE=false
            ```
            """)

# システム最適化提案を表示
suggest_system_optimization()

# PDF生成関連のインポート（新機能）
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from PIL import Image as PILImage
    import plotly.io as pio
    PDF_GENERATION_AVAILABLE = True
except ImportError as e:
    PDF_GENERATION_AVAILABLE = False
    st.warning("PDFレポート機能を使用するには、reportlab, Pillowライブラリが必要です")

# セキュリティモジュールのインポート
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

# AI/RAG関連のインポート
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

# OpenAI API Key（環境変数から取得を推奨）
openai_api_key = st.secrets["openai"]["openai_api_key"]

def check_internet_connection():
    """インターネット接続チェック"""
    try:
        # HTTPS接続のみを許可
        response = requests.get("https://www.google.com", timeout=5, verify=True)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def setup_ssl_context():
    """SSLコンテキストの設定"""
    try:
        # SSL証明書検証を強制
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        
        # TLS 1.2以上を強制
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        
        # 弱い暗号化を無効化
        ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
        return ssl_context
    except Exception as e:
        st.error(f"SSL設定エラー: {e}")
        return None

def save_original_spectrum_data_to_session(result, file_key):
    """元のスペクトルデータをセッションに保存"""
    try:
        # 有効なピークを計算
        filtered_peaks = [
            i for i in result["detected_peaks"]
            if i not in st.session_state.get(f"{file_key}_excluded_peaks", set())
        ]
        
        # 元データを辞書形式で保存
        original_data = {
            'wavenum': result['wavenum'].tolist() if hasattr(result['wavenum'], 'tolist') else list(result['wavenum']),
            'spectrum': result['spectrum'].tolist() if hasattr(result['spectrum'], 'tolist') else list(result['spectrum']),
            'second_derivative': result['second_derivative'].tolist() if hasattr(result['second_derivative'], 'tolist') else list(result['second_derivative']),
            'detected_peaks': list(result['detected_peaks']),
            'detected_prominences': list(result['detected_prominences']),
            'manual_peaks': st.session_state.get(f"{file_key}_manual_peaks", []),
            'excluded_peaks': st.session_state.get(f"{file_key}_excluded_peaks", set()),
            'all_peaks': list(result.get('all_peaks', [])),
            'all_prominences': list(result.get('all_prominences', [])),
            'filtered_peaks': filtered_peaks
        }
        
        st.session_state[f"{file_key}_original_spectrum_data"] = original_data
        st.success("✅ 元のスペクトルデータを保存しました")
        
    except Exception as e:
        st.warning(f"元データ保存警告: {e}")
class LLMConnector:
    """強化されたOpenAI LLM接続設定クラス"""
    def __init__(self):
        self.is_online = check_internet_connection()
        self.selected_model = "gpt-3.5-turbo"
        self.openai_client = None
        self.security_manager = get_security_manager() if SECURITY_AVAILABLE else None
        self.ssl_context = setup_ssl_context()
        self._setup_session()
        
    def _setup_session(self):
        """HTTPセッションの設定"""
        self.session = requests.Session()
        
        # ヘッダーの設定
        self.session.headers.update({
            'User-Agent': 'RamanEye-Client/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block'
        })
        
        # SSL設定
        if self.ssl_context:
            adapter = requests.adapters.HTTPAdapter()
            self.session.mount('https://', adapter)
        
    def setup_llm_connection(self):
        """強化されたOpenAI API接続設定"""
        # インターネット接続チェック
        if not self.is_online:
            st.sidebar.error("❌ インターネット接続が必要です")
            return False
        
        st.sidebar.success("🌐 インターネット接続: 正常")
        
        # モデル選択
        model_options = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]
        
        selected_model = st.sidebar.selectbox(
            "AI モデル選択",
            model_options,
            index=0,
            help="使用するAIモデルを選択してください"
        )
        
        try:
            # API設定
            openai.api_key = os.getenv("OPENAI_API_KEY", openai_api_key)
            
            # APIキーの妥当性検証
            if not self._validate_api_key(openai.api_key):
                st.sidebar.error("無効なAPIキーです")
                return False
            
            self.selected_model = selected_model
            self.openai_client = "openai"
            
            # セキュリティログ記録
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="API_CONNECTION_SETUP",
                    user_id=user_id,
                    details={
                        'model': selected_model,
                        'ssl_enabled': self.ssl_context is not None,
                        'timestamp': datetime.now().isoformat()
                    },
                    severity="INFO"
                )
            
            st.sidebar.success(f"✅ AI API接続設定完了 ({selected_model})")
            return True
            
        except Exception as e:
            st.sidebar.error(f"API設定エラー: {e}")
            
            # セキュリティログ記録
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="API_CONNECTION_ERROR",
                    user_id=user_id,
                    details={'error': str(e)},
                    severity="ERROR"
                )
            
            return False
    
    def _validate_api_key(self, api_key: str) -> bool:
        """APIキーの妥当性を検証"""
        if not api_key or len(api_key) < 20:
            return False
        
        # APIキーの形式チェック（OpenAI形式）
        if not api_key.startswith('sk-'):
            return False
        
        return True
    
    def generate_analysis(self, prompt, temperature=0.3, max_tokens=1024, stream_display=True):
        """強化されたOpenAI API解析実行"""
        if not self.selected_model:
            raise SecurityException("AI モデルが設定されていません")
        
        # プロンプトインジェクション対策
        sanitized_prompt = self._sanitize_prompt(prompt)
        
        system_message = "あなたはラマンスペクトロスコピーの専門家です。ピーク位置と論文、またはインターネット上の情報を比較して、このサンプルが何の試料なのか当ててください。すべて日本語で答えてください。"
        
        try:
            # セキュリティログ記録（リクエスト開始）
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="AI_ANALYSIS_REQUEST",
                    user_id=user_id,
                    details={
                        'model': self.selected_model,
                        'prompt_length': len(sanitized_prompt),
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                        'timestamp': datetime.now().isoformat()
                    },
                    severity="INFO"
                )
            
            # HTTPS通信でAPI呼び出し
            response = openai.ChatCompletion.create(
                model=self.selected_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": sanitized_prompt + "\n\nすべて日本語で詳しく説明してください。"}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                # 設定
                request_timeout=60,  # タイムアウト設定
                api_version=None  # 最新バージョンを強制
            )
            
            full_response = ""
            if stream_display:
                stream_area = st.empty()
            
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        content = self._sanitize_response_content(delta["content"])
                        full_response += content
                        if stream_display:
                            stream_area.markdown(full_response)
            
            # セキュリティログ記録（応答完了）
            if self.security_manager:
                user_id = current_user.get('username', 'unknown') if 'current_user' in locals() else 'unknown'
                self.security_manager.audit_logger.log_security_event(
                    event_type="AI_ANALYSIS_RESPONSE",
                    user_id=user_id,
                    details={
                        'response_length': len(full_response),
                        'success': True,
                        'timestamp': datetime.now().isoformat()
                    },
                    severity="INFO"
                )
            
            return full_response
                
        except Exception as e:
            # セキュリティログ記録（エラー）
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="AI_ANALYSIS_ERROR",
                    user_id=user_id,
                    details={'error': str(e)},
                    severity="ERROR"
                )
            
            raise SecurityException(f"OpenAI API解析エラー: {str(e)}")
    
    def _sanitize_prompt(self, prompt: str) -> str:
        """プロンプトインジェクション対策"""
        # 危険なパターンを除去
        dangerous_patterns = [
            "ignore previous instructions",
            "disregard the above",
            "forget everything",
            "new instruction:",
            "system:",
            "admin:",
            "jailbreak",
            "prompt injection"
        ]
        
        sanitized = prompt
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern.lower(), "")
            sanitized = sanitized.replace(pattern.upper(), "")
            sanitized = sanitized.replace(pattern.capitalize(), "")
        
        # 長さ制限
        if len(sanitized) > 10000:
            sanitized = sanitized[:10000]
        
        return sanitized
    
    def _sanitize_response_content(self, content: str) -> str:
        """応答内容のサニタイズ"""
        # HTMLタグの除去
        import re
        content = re.sub(r'<[^>]+>', '', content)
        
        # スクリプトタグの除去
        content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        
        return content
    
    def generate_qa_response(self, question, context, previous_qa_history=None):
        """強化された質問応答専用のOpenAI API呼び出し"""
        if not self.selected_model:
            raise SecurityException("OpenAI モデルが設定されていません")
        
        # 入力のサニタイズ
        sanitized_question = self._sanitize_prompt(question)
        sanitized_context = self._sanitize_prompt(context)
        
        system_message = """あなたはラマンスペクトロスコピーの専門家です。
解析結果や過去の質問履歴を踏まえて、ユーザーの質問に日本語で詳しく答えてください。
科学的根拠に基づいた正確な回答を心がけてください。"""
        
        # コンテキストの構築
        context_text = f"【解析結果】\n{sanitized_context}\n\n"
        
        if previous_qa_history:
            context_text += "【過去の質問履歴】\n"
            for i, qa in enumerate(previous_qa_history, 1):
                sanitized_prev_question = self._sanitize_prompt(qa['question'])
                sanitized_prev_answer = self._sanitize_prompt(qa['answer'])
                context_text += f"質問{i}: {sanitized_prev_question}\n回答{i}: {sanitized_prev_answer}\n\n"
        
        context_text += f"【新しい質問】\n{sanitized_question}"
        
        try:
            # セキュリティログ記録
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="QA_REQUEST",
                    user_id=user_id,
                    details={
                        'question_length': len(sanitized_question),
                        'context_length': len(sanitized_context),
                        'timestamp': datetime.now().isoformat()
                    },
                    severity="INFO"
                )
            
            response = openai.ChatCompletion.create(
                model=self.selected_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": context_text}
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=True,
                request_timeout=60
            )
            
            full_response = ""
            stream_area = st.empty()
            
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        content = self._sanitize_response_content(delta["content"])
                        full_response += content
                        stream_area.markdown(full_response)
            
            return full_response
                
        except Exception as e:
            raise SecurityException(f"質問応答エラー: {str(e)}")

class RamanRAGSystem:
    """強化されたRAG機能のクラス"""
    def __init__(
        self,
        embedding_model_name: str = 'all-MiniLM-L6-v2',
        use_openai_embeddings: bool = True,
        openai_embedding_model: str = "text-embedding-ada-002"
    ):
        self.use_openai = use_openai_embeddings and check_internet_connection()
        self.openai_embedding_model = openai_embedding_model
        self.security_manager = get_security_manager() if SECURITY_AVAILABLE else None
        
        if self.use_openai:
            self.embedding_model = None
        else:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        
        self.vector_db = None
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.embedding_dim: int = 0
        self.db_info: Dict = {}
    
    def build_vector_database(self, folder_path: str):
        """強化されたベクトルデータベース構築"""
        if not PDF_AVAILABLE:
            st.error("PDF処理ライブラリが利用できません")
            return
            
        if not os.path.exists(folder_path):
            st.error(f"指定されたフォルダが存在しません: {folder_path}")
            return

        # セキュリティチェック
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        # ファイル一覧取得（セキュリティ考慮）
        file_patterns = ['*.pdf', '*.docx', '*.txt']
        files = []
        for pat in file_patterns:
            potential_files = glob.glob(os.path.join(folder_path, pat))
            for file_path in potential_files:
                # ファイルアクセスのセキュリティチェック
                if self.security_manager:
                    access_result = self.security_manager.secure_file_access(
                        file_path, user_id, 'read'
                    )
                    if access_result['status'] == 'success':
                        files.append(file_path)
                    else:
                        st.warning(f"ファイルアクセス拒否: {file_path}")
                else:
                    files.append(file_path)
        
        if not files:
            st.warning("アクセス可能なファイルが見つかりません。")
            return

        # テキスト抽出とチャンク化（セキュリティ付き）
        all_chunks, all_metadata = [], []
        st.info(f"{len(files)} 件のファイルを安全に処理中…")
        pbar = st.progress(0)
        
        for idx, fp in enumerate(files):
            try:
                # ファイル完全性チェック
                if self.security_manager:
                    integrity_result = self.security_manager.integrity_manager.verify_file_integrity(Path(fp))
                    if integrity_result['status'] == 'corrupted':
                        st.error(f"ファイル完全性エラー: {fp}")
                        continue
                
                text = self._extract_text(fp)
                chunks = self.chunk_text(text)
                
                for c in chunks:
                    all_chunks.append(c)
                    all_metadata.append({
                        'filename': os.path.basename(fp),
                        'filepath': fp,
                        'preview': c[:100] + "…" if len(c) > 100 else c,
                        'processed_by': user_id,
                        'processed_at': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.error(f"ファイル処理エラー {fp}: {e}")
                if self.security_manager:
                    self.security_manager.audit_logger.log_security_event(
                        event_type="FILE_PROCESSING_ERROR",
                        user_id=user_id,
                        details={'file_path': fp, 'error': str(e)},
                        severity="ERROR"
                    )
                continue
                
            pbar.progress((idx + 1) / len(files))

        if not all_chunks:
            st.error("抽出できるテキストチャンクがありませんでした。")
            return

        # 埋め込みベクトルの生成
        st.info("埋め込みベクトルを生成中…")
        try:
            if self.use_openai:
                embeddings = self._create_openai_embeddings(all_chunks)
            else:
                embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)

            # FAISSインデックス構築
            self.embedding_dim = embeddings.shape[1]
            index = faiss.IndexFlatIP(self.embedding_dim)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)

            # 状態保存
            self.vector_db = index
            self.documents = all_chunks
            self.document_metadata = all_metadata
            self.db_info = {
                'created_at': datetime.now().isoformat(),
                'created_by': user_id,
                'n_docs': len(files),
                'n_chunks': len(all_chunks),
                'source_files': [os.path.basename(f) for f in files],
                'embedding_model': (
                    self.openai_embedding_model if self.use_openai 
                    else self.embedding_model.__class__.__name__
                ),
                'security_enabled': SECURITY_AVAILABLE
            }
            
            # セキュリティログ記録
            if self.security_manager:
                self.security_manager.audit_logger.log_security_event(
                    event_type="VECTOR_DB_CREATED",
                    user_id=user_id,
                    details={
                        'n_chunks': len(all_chunks),
                        'n_files': len(files),
                        'embedding_model': self.db_info['embedding_model']
                    },
                    severity="INFO"
                )
            
            st.success(f"ベクトルDB構築完了: {len(all_chunks)} チャンク")
            
        except Exception as e:
            st.error(f"ベクトルDB構築エラー: {e}")
            if self.security_manager:
                self.security_manager.audit_logger.log_security_event(
                    event_type="VECTOR_DB_ERROR",
                    user_id=user_id,
                    details={'error': str(e)},
                    severity="ERROR"
                )
    
    def _create_openai_embeddings(self, texts: List[str], batch_size: int = 200) -> np.ndarray:
        """OpenAI埋め込みAPIの使用"""
        all_embs = []
        
        # セキュリティログ記録
        if self.security_manager:
            current_user = st.session_state.get('current_user', {})
            user_id = current_user.get('username', 'unknown')
            
            self.security_manager.audit_logger.log_security_event(
                event_type="OPENAI_EMBEDDING_REQUEST",
                user_id=user_id,
                details={
                    'num_texts': len(texts),
                    'batch_size': batch_size,
                    'model': self.openai_embedding_model
                },
                severity="INFO"
            )
        
        try:
            for i in range(0, len(texts), batch_size):
                chunk = texts[i:i+batch_size]
                
                # テキストの前処理・サニタイズ
                sanitized_chunk = []
                for text in chunk:
                    # 長すぎるテキストのトランケート
                    if len(text) > 8000:  # OpenAI制限に合わせて調整
                        text = text[:8000]
                    sanitized_chunk.append(text)
                
                # HTTPS通信でAPI呼び出し
                resp = openai.Embedding.create(
                    model=self.openai_embedding_model,
                    input=sanitized_chunk,
                    timeout=60
                )
                
                embs = [d['embedding'] for d in resp['data']]
                all_embs.extend(embs)
                
                # 進捗表示
                if len(texts) > batch_size:
                    progress = min(i + batch_size, len(texts)) / len(texts)
                    st.progress(progress)
                    
        except Exception as e:
            if self.security_manager:
                user_id = current_user.get('username', 'unknown') if 'current_user' in locals() else 'unknown'
                self.security_manager.audit_logger.log_security_event(
                    event_type="OPENAI_EMBEDDING_ERROR",
                    user_id=user_id,
                    details={'error': str(e)},
                    severity="ERROR"
                )
            raise SecurityException(f"埋め込み生成エラー: {e}")
        
        return np.array(all_embs, dtype=np.float32)
    
    def _extract_text(self, file_path: str) -> str:
        """ファイルからのテキスト抽出"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # ファイルサイズチェック
        file_size = os.path.getsize(file_path)
        MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB制限
        if file_size > MAX_FILE_SIZE:
            raise SecurityException(f"ファイルサイズが制限を超えています: {file_path}")
        
        try:
            if ext == '.pdf':
                reader = PyPDF2.PdfReader(file_path)
                text_parts = []
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                        text_parts.append(page_text)
                    except Exception as e:
                        st.warning(f"PDF ページ {page_num} 読み込みエラー: {e}")
                return "\n".join(text_parts)
                
            elif ext == '.docx':
                doc = docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs)
                
            elif ext == '.txt':
                with open(file_path, encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                # テキストファイルのサイズ制限
                if len(content) > 1000000:  # 1MB制限
                    content = content[:1000000]
                return content
                
        except Exception as e:
            st.error(f"テキスト抽出エラー {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """テキストをチャンクに分割"""
        if not text or not text.strip():
            return []
        
        # テキストの前処理
        text = text.strip()
        
        # 危険なコンテンツのフィルタリング
        if self._contains_malicious_content(text):
            st.warning("潜在的に危険なコンテンツが検出されました。処理をスキップします。")
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip() and len(chunk) > 10:  # 短すぎるチャンクを除外
                chunks.append(chunk)
                
        return chunks
    
    def _contains_malicious_content(self, text: str) -> bool:
        """悪意のあるコンテンツの検出"""
        # 基本的なパターンマッチング
        malicious_patterns = [
            r'<script.*?>',
            r'javascript:',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
            r'document\.cookie',
            r'window\.location'
        ]
        
        import re
        text_lower = text.lower()
        
        for pattern in malicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False
    
    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """関連文書検索"""
        if self.vector_db is None:
            return []
        
        try:
            # クエリのサニタイズ
            sanitized_query = query.strip()
            if len(sanitized_query) > 1000:
                sanitized_query = sanitized_query[:1000]
            
            # セキュリティログ記録
            if self.security_manager:
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                self.security_manager.audit_logger.log_security_event(
                    event_type="DOCUMENT_SEARCH",
                    user_id=user_id,
                    details={
                        'query_length': len(sanitized_query),
                        'top_k': top_k
                    },
                    severity="INFO"
                )
    
            # DB作成時のモデル情報を確認
            model_used = self.db_info.get("embedding_model", "")
            if model_used == "text-embedding-ada-002":
                query_emb = self._create_openai_embeddings([sanitized_query])
            else:
                query_emb = self.embedding_model.encode([sanitized_query], show_progress_bar=False)
    
            query_emb = np.array(query_emb, dtype=np.float32)
            faiss.normalize_L2(query_emb)
            
            # 類似文書を検索
            scores, indices = self.vector_db.search(query_emb, top_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'text': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'similarity_score': float(score)
                    })
            
            return results
            
        except Exception as e:
            st.error(f"文書検索エラー: {e}")
            return []
    
    def save_database(self, save_path: str, db_name: str = "raman_rag_database"):
        """構築したデータベースを保存"""
        if self.vector_db is None:
            st.error("保存するデータベースが存在しません。")
            return False
        
        try:
            save_folder = Path(save_path)
            save_folder.mkdir(parents=True, exist_ok=True)
            
            # FAISSインデックスを保存
            faiss_path = save_folder / f"{db_name}_faiss.index"
            faiss.write_index(self.vector_db, str(faiss_path))
            
            # ドキュメントとメタデータを保存
            documents_path = save_folder / f"{db_name}_documents.pkl"
            with open(documents_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_metadata': self.document_metadata,
                    'embedding_dim': self.embedding_dim
                }, f)
            
            # データベース情報を保存
            info_path = save_folder / f"{db_name}_info.json"
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(self.db_info, f, ensure_ascii=False, indent=2)
            
            st.success(f"✅ データベースを保存しました: {save_folder}")
            st.info(f"📁 保存されたファイル:\n"
                   f"- {db_name}_faiss.index (FAISSベクトルインデックス)\n"
                   f"- {db_name}_documents.pkl (ドキュメントデータ)\n"
                   f"- {db_name}_info.json (データベース情報)")
            
            return True
            
        except Exception as e:
            st.error(f"データベース保存エラー: {e}")
            return False
    
    def get_database_info(self) -> Dict:
        """データベースの情報を取得"""
        if self.vector_db is None:
            return {"status": "データベースが構築されていません"}
        
        info = self.db_info.copy()
        info["status"] = "構築済み"
        info["current_chunks"] = len(self.documents)
        return info

class RamanSpectrumAnalyzer:
    """ラマンスペクトル解析クラス"""
    def generate_analysis_prompt(self, peak_data: List[Dict], relevant_docs: List[Dict], user_hint: Optional[str] = None) -> str:
        """ラマンスペクトル解析のためのプロンプトを生成"""
        
        def format_peaks(peaks: List[Dict]) -> str:
            header = "【検出ピーク一覧】"
            lines = [
                f"{i+1}. 波数: {p.get('wavenumber', 0):.1f} cm⁻¹, "
                f"強度: {p.get('intensity', 0):.3f}, "
                f"卓立度: {p.get('prominence', 0):.3f}, "
                f"種類: {'自動検出' if p.get('type') == 'auto' else '手動追加'}"
                for i, p in enumerate(peaks)
            ]
            return "\n".join([header] + lines)

        def format_reference_excerpts(docs: List[Dict]) -> str:
            header = "【引用文献の抜粋と要約】"
            lines = []
            for i, doc in enumerate(docs, 1):
                title = doc.get("metadata", {}).get("filename", f"文献{i}")
                page = doc.get("metadata", {}).get("page")
                summary = doc.get("page_content", "").strip()
                lines.append(f"\n--- 引用{i} ---")
                lines.append(f"出典ファイル: {title}")
                if page is not None:
                    lines.append(f"ページ番号: {page}")
                lines.append(f"抜粋内容:\n{summary}")
            return "\n".join([header] + lines)

        def format_doc_summaries(docs: List[Dict], preview_length: int = 300) -> str:
            header = "【文献の概要（類似度付き）】"
            lines = []
            for i, doc in enumerate(docs, 1):
                filename = doc.get("metadata", {}).get("filename", f"文献{i}")
                similarity = doc.get("similarity_score", 0.0)
                text = doc.get("text") or doc.get("page_content") or ""
                lines.append(
                    f"文献{i} (類似度: {similarity:.3f})\n"
                    f"ファイル名: {filename}\n"
                    f"冒頭抜粋: {text.strip()[:preview_length]}...\n"
                )
            return "\n".join([header] + lines)

        # プロンプト本文の構築
        sections = [
            "以下は、ラマンスペクトルで検出されたピーク情報です。",
            "これらのピークに基づき、試料の成分や特徴について推定してください。",
            "なお、文献との比較においてはピーク位置が±5cm⁻¹程度ずれることがあります。",
            "そのため、±5cm⁻¹以内の差であれば一致とみなして解析を行ってください。\n"
        ]

        if user_hint:
            sections.append(f"【ユーザーによる補足情報】\n{user_hint}\n")

        if peak_data:
            sections.append(format_peaks(peak_data))
        if relevant_docs:
            sections.append(format_reference_excerpts(relevant_docs))
            sections.append(format_doc_summaries(relevant_docs))

        sections.append(
            "これらを参考に、試料に含まれる可能性のある化合物や物質構造、特徴について詳しく説明してください。\n"
            "出力は日本語でお願いします。\n"
            "## 解析の観点:\n"
            "1. 各ピークの化学的帰属とその根拠\n"
            "2. 試料の可能な組成や構造\n"
            "3. 文献情報との比較・対照\n\n"
            "詳細で科学的根拠に基づいた考察を日本語で提供してください。"
        )

        return "\n".join(sections)

# === 新機能: PDFレポートジェネレータ ===
class RamanPDFReportGenerator:
    """ラマンスペクトル解析PDFレポート生成クラス"""
    
    def __init__(self):
        self.temp_dir = "./temp_pdf_assets"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # 日本語フォントの設定を試行
        self.setup_japanese_font()
        
        # レポートスタイルの設定
        self.setup_styles()
        
    def _sanitize_text_for_pdf(self, text: str) -> str:
        """PDFに安全なテキストに変換"""
        if text is None:
            return ""
        
        # HTMLエスケープ処理
        text = str(text)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#39;')
        
        # 改行の正規化
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 制御文字を除去
        import re
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text
    def setup_japanese_font(self):
        """日本語フォントの設定（フォールバック戦略強化）"""
        self.japanese_font_available = False
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))
        self.japanese_font_name = 'HeiseiKakuGo-W5'  # デフォルトを必ず定義

        try:
            # システムにある日本語フォントを探す
            font_paths = [
            # Windows - 日本語フォント
            "C:/Windows/Fonts/yumin.ttf",
            
            """
            "C:/Windows/Fonts/yumin.ttf",
            "C:/Windows/Fonts/msgothic.ttc",
            "C:/Windows/Fonts/meiryo.ttc", 
            "C:/Windows/Fonts/meiryob.ttc",
            "C:/Windows/Fonts/YuGothic.ttc",
            "C:/Windows/Fonts/YuGothM.ttc",
            "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
            "C:/Windows/Fonts/NotoSansJP-Regular.otf",
            
            # macOS - 日本語フォント
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴ ProN W3.otf",
            "/Library/Fonts/NotoSansCJK.ttc",
            "/Library/Fonts/NotoSansJP-Regular.otf",
            
            # Linux - 日本語フォント優先
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansJP-Regular.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansJP-Regular.otf",
            "/usr/share/fonts/truetype/vlgothic/VL-Gothic-Regular.ttf",
            "/usr/share/fonts/truetype/vlgothic/VL-PGothic-Regular.ttf",
            
            # 汎用フォント（最後の手段）
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            """
        ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('JapaneseFont', font_path))
                        self.japanese_font_available = True
                        self.japanese_font_name = 'JapaneseFont'
                        st.info(f"フォント使用: {os.path.basename(font_path)}")
                        break
                    except Exception as e:
                        continue
            """
            if not self.japanese_font_available:
                # より汎用的なフォールバック
                try:
                    # ReportLabの標準フォントを使用
                    self.japanese_font_name = 'Times-Roman'
                    st.info("標準フォント（Times-Roman）を使用します。日本語は文字化けする可能性があります。")
                except:
                    self.japanese_font_name = 'Helvetica'
                    st.info("基本フォント（Helvetica）を使用します。日本語は文字化けする可能性があります。")
            """
        except Exception as e:
            self.japanese_font_name = 'Helvetica'
            st.info(f"フォント設定警告: {e}. 基本フォントを使用します。")
    
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
        
    def plotly_to_image(self, fig, filename, width=800, height=600, format='png'):
        """PlotlyグラフをPNG画像に変換（改良版）"""
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
            
            # 画像保存（複数の方法を試行）
            success = False
            
            # 方法1: kaleido使用を試行
            try:
                pio.write_image(fig, img_path, format=format, width=width, height=height, scale=2)
                success = True
                st.info("Kaleidoエンジンでグラフを画像化しました")
            except Exception as kaleido_error:
                st.warning(f"Kaleidoエンジン使用失敗: {str(kaleido_error)}")
            
        except Exception as e:
            st.error(f"グラフ画像変換エラー: {e}")
            # 最終フォールバック
            placeholder_path = os.path.join(self.temp_dir, f"{filename}_fallback.png")
            self._create_enhanced_placeholder_image(placeholder_path, width, height, f"Graph Error: {filename}")
            return placeholder_path

    def _create_enhanced_placeholder_image(self, path, width, height, text):
        """高品質プレースホルダー画像を作成"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # 白背景の画像を作成
            img = PILImage.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # グラデーション背景を作成
            for y in range(height):
                color_value = int(255 - (y / height) * 20)  # 薄いグラデーション
                color = (color_value, color_value, color_value)
                draw.line([(0, y), (width, y)], fill=color)
            
            # フォントを設定
            try:
                # より大きなフォントを試行
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
            except:
                font_large = None
                font_small = None
            
            # タイトルテキスト
            title_text = "Raman Spectrum Analysis"
            if hasattr(draw, 'textbbox'):
                title_bbox = draw.textbbox((0, 0), title_text, font=font_large)
                title_width = title_bbox[2] - title_bbox[0]
                title_height = title_bbox[3] - title_bbox[1]
            else:
                title_width, title_height = 200, 20
            
            title_x = (width - title_width) // 2
            title_y = height // 4
            
            draw.text((title_x, title_y), title_text, fill='darkblue', font=font_large)
            
            # サブタイトル
            subtitle = text
            if hasattr(draw, 'textbbox'):
                sub_bbox = draw.textbbox((0, 0), subtitle, font=font_small)
                sub_width = sub_bbox[2] - sub_bbox[0]
            else:
                sub_width = len(subtitle) * 8
            
            sub_x = (width - sub_width) // 2
            sub_y = title_y + title_height + 20
            
            draw.text((sub_x, sub_y), subtitle, fill='black', font=font_small)
            
            # 簡単なグラフ風の装飾を追加
            # X軸
            draw.line([(width//8, height*3//4), (width*7//8, height*3//4)], fill='black', width=2)
            # Y軸
            draw.line([(width//8, height//8), (width//8, height*3//4)], fill='black', width=2)
            
            # サンプル波形
            points = []
            for i in range(width//8, width*7//8, 5):
                x = i
                y = height//2 + int(50 * np.sin((i - width//8) * 0.01)) + int(30 * np.sin((i - width//8) * 0.03))
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill='blue', width=2)
            
            # いくつかのピーク点を追加
            peak_points = [(width//3, height//2 - 20), (width*2//3, height//2 - 40)]
            for px, py in peak_points:
                draw.ellipse([px-4, py-4, px+4, py+4], fill='red')
            
            # 枠線を描画
            draw.rectangle([0, 0, width-1, height-1], outline='gray', width=2)
            
            # 注意書き
            note_text = "Note: Graph generated without Kaleido engine"
            note_y = height - 30
            if hasattr(draw, 'textbbox'):
                note_bbox = draw.textbbox((0, 0), note_text, font=font_small)
                note_width = note_bbox[2] - note_bbox[0]
            else:
                note_width = len(note_text) * 6
            
            note_x = (width - note_width) // 2
            draw.text((note_x, note_y), note_text, fill='gray', font=font_small)
            
            img.save(path)
            
        except Exception as e:
            # 最も基本的なプレースホルダー
            try:
                img = PILImage.new('RGB', (width, height), color='lightgray')
                draw = ImageDraw.Draw(img)
                
                simple_text = "Graph Placeholder"
                text_x = width // 2 - 50
                text_y = height // 2 - 10
                draw.text((text_x, text_y), simple_text, fill='black')
                
                img.save(path)
            except:
                st.warning(f"プレースホルダー画像作成最終エラー: {e}")
    
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
            
            # テキストのサイズを計算
            if hasattr(draw, 'textbbox'):
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
            else:
                # フォールバック
                text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (100, 20)
            
            x = (width - text_width) // 2
            y = (height - text_height) // 2
            
            draw.text((x, y), text, fill='black', font=font)
            
            # 枠線を描画
            draw.rectangle([0, 0, width-1, height-1], outline='gray')
            
            img.save(path)
            
        except Exception as e:
            st.warning(f"プレースホルダー画像作成エラー: {e}")
    
    def generate_pdf_report(
        self, 
        file_key: str,
        peak_data: List[Dict],
        analysis_result: str,
        peak_summary_df: pd.DataFrame,
        plotly_figure: go.Figure = None,
        relevant_docs: List[Dict] = None,
        user_hint: str = None,
        qa_history: List[Dict] = None,
        database_info: Dict = None,  # この行を追加
        database_files: List[str] = None,  # この行を追加
        original_spectrum_data: Dict = None,  # この行を追加
    ) -> bytes:

        """PDFレポートを生成"""
        
        if not PDF_GENERATION_AVAILABLE:
            raise Exception("PDF生成ライブラリが利用できません")
        
        # PDFファイルをメモリ上に作成
        import io
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
            if original_spectrum_data:
                story.extend(self._create_graph_section_from_original_data(original_spectrum_data, file_key))
        
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
    
    def _create_graph_section_from_original_data(self, spectrum_data: Dict, file_key: str) -> List:
        """元のスペクトルデータから直接グラフセクションを作成"""
        content = []
        
        heading_text = self._sanitize_text_for_pdf("スペクトルおよびピーク検出結果" if self.japanese_font_available else "Spectrum and Peak Detection Results")
        content.append(Paragraph(heading_text, self.styles['JapaneseHeading']))
        
        try:
            # 元データから直接画像を生成
            img_path = self._create_image_from_original_data(spectrum_data, file_key)
            
            if img_path and os.path.exists(img_path):
                try:
                    img = Image(img_path, width=7*inch, height=5.6*inch)
                    content.append(img)
                    content.append(Spacer(1, 0.2*inch))
                    st.success("✅ 元のスペクトルデータから直接グラフを生成しました")
                except Exception as img_error:
                    st.warning(f"画像追加エラー: {img_error}")
                    content.append(self._create_text_based_spectrum_info(spectrum_data))
            else:
                content.append(self._create_text_based_spectrum_info(spectrum_data))
            
        except Exception as e:
            st.error(f"元データグラフ作成エラー: {e}")
            content.append(self._create_text_based_spectrum_info(spectrum_data))
        
        # グラフの説明
        if self.japanese_font_available:
            description = """
            上図は元のスペクトルデータから直接生成されたラマンスペクトルとピーク検出結果です。
            赤い点は検出されたピーク、緑の星印は手動で追加されたピークを表示しています。
            """
        else:
            description = """
            The above figure shows the Raman spectrum and peak detection results generated directly from original data.
            Red points indicate detected peaks, green stars show manually added peaks.
            """
        
        description = self._sanitize_text_for_pdf(description)
        content.append(Paragraph(description, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_image_from_original_data(self, spectrum_data: Dict, file_key: str) -> str:
        """元のスペクトルデータから直接画像を生成"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 画像パス
            img_path = os.path.join(self.temp_dir, f"original_spectrum_{file_key}.png")
            
            # データを取得
            wavenum = spectrum_data.get('wavenum', [])
            spectrum = spectrum_data.get('spectrum', [])
            second_derivative = spectrum_data.get('second_derivative', [])
            detected_peaks = spectrum_data.get('detected_peaks', [])
            detected_prominences = spectrum_data.get('detected_prominences', [])
            manual_peaks = spectrum_data.get('manual_peaks', [])
            excluded_peaks = spectrum_data.get('excluded_peaks', set())
            all_peaks = spectrum_data.get('all_peaks', [])
            all_prominences = spectrum_data.get('all_prominences', [])
            
            if not wavenum or not spectrum:
                raise Exception("基本スペクトルデータが不足しています")
            
            # numpy配列に変換
            wavenum = np.array(wavenum)
            spectrum = np.array(spectrum)
            if len(second_derivative) > 0:
                second_derivative = np.array(second_derivative)
            
            # 図を作成
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), facecolor='white')
            fig.suptitle('Raman Spectrum Analysis', fontsize=14, y=0.95)
            
            # 1段目：メインスペクトル
            axes[0].plot(wavenum, spectrum, 'b-', linewidth=2, label='Spectrum')
            
            # 有効な検出ピーク
            valid_peaks = [i for i in detected_peaks if i not in excluded_peaks]
            if len(valid_peaks) > 0 and len(valid_peaks) <= len(wavenum):
                valid_indices = [i for i in valid_peaks if 0 <= i < len(wavenum)]
                if valid_indices:
                    axes[0].scatter(wavenum[valid_indices], spectrum[valid_indices], 
                                   c='red', s=50, label='Detected Peaks', zorder=5)
            
            # 除外されたピーク
            excluded_indices = [i for i in excluded_peaks if 0 <= i < len(wavenum)]
            if excluded_indices:
                axes[0].scatter(wavenum[excluded_indices], spectrum[excluded_indices], 
                               c='gray', s=50, marker='x', label='Excluded Peaks', zorder=5)
            
            # 手動ピーク
            if manual_peaks:
                manual_x = []
                manual_y = []
                for peak_wn in manual_peaks:
                    # 最も近い波数インデックスを見つける
                    idx = np.argmin(np.abs(wavenum - peak_wn))
                    if 0 <= idx < len(spectrum):
                        manual_x.append(peak_wn)
                        manual_y.append(spectrum[idx])
                
                if manual_x:
                    axes[0].scatter(manual_x, manual_y, c='green', s=80, marker='*', 
                                   label='Manual Peaks', zorder=6)
            
            axes[0].set_ylabel('Intensity (a.u.)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2段目：2次微分
            if len(second_derivative) > 0:
                axes[1].plot(wavenum, second_derivative, 'purple', linewidth=1, label='2nd Derivative')
                axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            else:
                # 2次微分データがない場合は簡易計算
                if len(spectrum) > 4:
                    simple_2nd_deriv = np.gradient(np.gradient(spectrum))
                    axes[1].plot(wavenum, simple_2nd_deriv, 'purple', linewidth=1, label='2nd Derivative (Calculated)')
                    axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            axes[1].set_ylabel('2nd Derivative')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # 3段目：Prominence
            if len(all_peaks) > 0 and len(all_prominences) > 0:
                # 全ピークのprominence
                valid_all_indices = [i for i in all_peaks if 0 <= i < len(wavenum)]
                if valid_all_indices and len(valid_all_indices) <= len(all_prominences):
                    axes[2].scatter(wavenum[valid_all_indices], all_prominences[:len(valid_all_indices)], 
                                   c='orange', s=20, alpha=0.6, label='All Peaks')
                
                # 有効なピークのprominence
                if len(valid_peaks) > 0 and len(detected_prominences) > 0:
                    valid_prom_indices = [i for i in valid_peaks if 0 <= i < len(wavenum)]
                    if valid_prom_indices:
                        # prominenceデータをマッピング
                        valid_prominences = []
                        for peak_idx in valid_prom_indices:
                            # detected_peaksでのインデックスを見つける
                            try:
                                orig_idx = list(detected_peaks).index(peak_idx)
                                if orig_idx < len(detected_prominences):
                                    valid_prominences.append(detected_prominences[orig_idx])
                                else:
                                    valid_prominences.append(0.1)  # デフォルト値
                            except ValueError:
                                valid_prominences.append(0.1)
                        
                        if valid_prominences:
                            axes[2].scatter(wavenum[valid_prom_indices], valid_prominences, 
                                           c='red', s=60, label='Valid Peaks', zorder=5)
            
            axes[2].set_xlabel('Wavenumber (cm⁻¹)')
            axes[2].set_ylabel('Prominence')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(img_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return img_path
            
        except Exception as e:
            st.error(f"元データ画像生成エラー: {e}")
            st.write(f"デバッグ情報: データキー = {list(spectrum_data.keys())}")
            return None
    
    def _create_text_based_spectrum_info(self, spectrum_data: Dict) -> Paragraph:
        """テキストベースのスペクトル情報を作成"""
        try:
            wavenum = spectrum_data.get('wavenum', [])
            spectrum = spectrum_data.get('spectrum', [])
            detected_peaks = spectrum_data.get('detected_peaks', [])
            manual_peaks = spectrum_data.get('manual_peaks', [])
            
            info_text = f"""
            スペクトル情報 (テキスト表示):
            • データポイント数: {len(wavenum)}
            • 波数範囲: {min(wavenum):.1f} - {max(wavenum):.1f} cm⁻¹
            • 強度範囲: {min(spectrum):.3f} - {max(spectrum):.3f}
            • 検出ピーク数: {len(detected_peaks)}
            • 手動ピーク数: {len(manual_peaks)}
            
            注: グラフィカル表示でエラーが発生したため、テキスト形式で表示しています。
            """
            
            info_text = self._sanitize_text_for_pdf(info_text)
            return Paragraph(info_text, self.styles['JapaneseNormal'])
            
        except Exception as e:
            error_text = self._sanitize_text_for_pdf(f"スペクトル情報表示エラー: {e}")
            return Paragraph(error_text, self.styles['JapaneseNormal'])
    
    
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

def render_qa_section(file_key, analysis_context, llm_connector):
    """AI解析結果の後に質問応答セクションを表示する関数"""
    qa_history_key = f"{file_key}_qa_history"
    if qa_history_key not in st.session_state:
        st.session_state[qa_history_key] = []
    
    st.markdown("---")
    st.subheader(f"💬 追加質問 - {file_key}")
    
    # 質問履歴の表示
    if st.session_state[qa_history_key]:
        with st.expander("📚 質問履歴を表示", expanded=False):
            for i, qa in enumerate(st.session_state[qa_history_key], 1):
                st.markdown(f"**質問{i}:** {qa['question']}")
                st.markdown(f"**回答{i}:** {qa['answer']}")
                st.markdown(f"*質問日時: {qa['timestamp']}*")
                st.markdown("---")
    
    # 質問入力フォーム
    with st.form(key=f"qa_form_{file_key}"):
        st.markdown("**解析結果について質問があれば、下記にご記入ください：**")
        
        st.markdown("""
        **質問例:**
        - このピークは何に由来しますか？
        - 他の可能性のある物質はありますか？
        - 測定条件で注意すべき点は？
        - 定量分析は可能ですか？
        """)
        
        user_question = st.text_area(
            "質問内容:",
            placeholder="例: 1500 cm⁻¹付近のピークについて詳しく教えてください",
            height=100
        )
        
        submit_button = st.form_submit_button("💬 質問する")
    
    # 質問処理
    if submit_button and user_question.strip():
        with st.spinner("AIが回答を考えています..."):
            try:
                answer = llm_connector.generate_qa_response(
                    question=user_question,
                    context=analysis_context,
                    previous_qa_history=st.session_state[qa_history_key]
                )
                
                new_qa = {
                    'question': user_question,
                    'answer': answer,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state[qa_history_key].append(new_qa)
                
                # st.success("✅ 回答が完了しました！")
                
            except Exception as e:
                st.error(f"質問処理中にエラーが発生しました: {str(e)}")
    
    elif submit_button and not user_question.strip():
        st.warning("質問内容を入力してください。")
        
def peak_ai_analysis_mode():
    """強化されたPeak AI analysis mode"""
    if not PDF_AVAILABLE:
        st.error("AI解析機能を使用するには、以下のライブラリが必要です：")
        st.code("pip install PyPDF2 python-docx openai faiss-cpu sentence-transformers")
        return
    
    st.header("ラマンピークAI解析")
    
    # LLM接続設定（セキュリティ強化版）
    llm_connector = LLMConnector()
    
    # OpenAI API設定
    llm_ready = llm_connector.setup_llm_connection()
    
    # RAG設定セクション（セキュリティ強化版）
    st.sidebar.subheader("📚 論文データベース設定")
    
    # データベース操作モードの選択
    db_mode = st.sidebar.radio(
        "データベース操作モード",
        ["新規作成", "既存データベース読み込み"],
        index=0
    )
     
    # 一時保存用ディレクトリ
    TEMP_DIR = "./tmp_uploads"
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # RAGシステムの初期化（セキュリティ強化版）
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RamanRAGSystem()
        st.session_state.rag_db_built = False
    
    if db_mode == "新規作成":
        setup_new_database(TEMP_DIR)
    elif db_mode == "既存データベース読み込み":
        load_existing_database()
    
    # データベース状態表示
    if st.session_state.rag_db_built:
        st.sidebar.success("✅ 論文データベース構築済み")
        
        if st.sidebar.button("📊 データベース情報を表示"):
            db_info = st.session_state.rag_system.get_database_info()
            st.sidebar.json(db_info)
    else:
        st.sidebar.info("ℹ️ 論文データベース未構築")
        
    # サイドバーに補足指示欄を追加
    user_hint = st.sidebar.text_area(
        "AIへの補足ヒント（任意）",
        placeholder="例：この試料はポリエチレン系高分子である可能性がある、など"
    )
    
    # ピーク解析部分の実行（セキュリティ強化版）
    perform_peak_analysis_with_ai(llm_connector, user_hint, llm_ready)

def setup_new_database(TEMP_DIR):
    """新規データベースの作成"""
    uploaded_files = st.sidebar.file_uploader(
        "📄 文献PDFを選択してください（複数可）",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if st.sidebar.button("📚 論文データベース構築"):
        if not uploaded_files:
            st.sidebar.warning("文献ファイルを選択してください。")
        else:
            with st.spinner("文献をアップロードし、データベースを構築中..."):
                security_manager = get_security_manager() if SECURITY_AVAILABLE else None
                current_user = st.session_state.get('current_user', {})
                user_id = current_user.get('username', 'unknown')
                
                uploaded_count = 0
                for uploaded_file in uploaded_files:
                    try:
                        # セキュリティ強化ファイルアップロード
                        if security_manager:
                            upload_result = security_manager.secure_file_upload(uploaded_file, user_id)
                            if upload_result['status'] == 'success':
                                uploaded_count += 1
                            else:
                                st.error(f"ファイルアップロードエラー: {upload_result['message']}")
                        else:
                            # フォールバック: 基本的なファイル保存
                            # uploaded_fileがstreamlitのUploadedFileオブジェクトなのでname属性を使用
                            save_path = os.path.join(TEMP_DIR, uploaded_file.name)
                            with open(save_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            uploaded_count += 1
                    except Exception as e:
                        st.error(f"ファイル処理エラー ({uploaded_file.name}): {e}")
                
                if uploaded_count > 0:
                    st.session_state.rag_system.build_vector_database(TEMP_DIR)
                    st.session_state.rag_db_built = True
                    st.sidebar.success(f"✅ {uploaded_count} 件のファイルからデータベースを構築しました。")

def load_existing_database():
    """既存データベースの読み込み"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 既存データベース読み込み")
    st.sidebar.info("セキュリティ機能により、アクセス権限のあるデータベースのみ読み込み可能です。")

def perform_peak_analysis_with_ai(llm_connector, user_hint, llm_ready):
    """セキュリティ強化されたAI機能を含むピーク解析の実行"""
    # パラメータ設定
    pre_start_wavenum = 400
    pre_end_wavenum = 2000
    
    # セッションステートの初期化
    for key, default in {
        "prominence_threshold": 100,
        "second_deriv_threshold": 100,
        "savgol_wsize": 5,
        "spectrum_type_select": "ベースライン削除",
        "second_deriv_smooth": 5,
        "manual_peak_keys": []
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
    # UIパネル（Sidebar）
    start_wavenum = st.sidebar.number_input("波数（開始）を入力してください:", -200, 4800, value=pre_start_wavenum, step=100)
    end_wavenum = st.sidebar.number_input("波数（終了）を入力してください:", -200, 4800, value=pre_end_wavenum, step=100)
    dssn_th = st.sidebar.number_input("ベースラインパラメーターを入力してください:", 1, 10000, value=1000, step=1) / 1e7
    savgol_wsize = st.sidebar.number_input("移動平均のウィンドウサイズを入力してください:", 5, 101, step=2, key="savgol_wsize")
    
    st.sidebar.subheader("ピーク検出設定")
    
    spectrum_type = st.sidebar.selectbox("解析スペクトル:", ["ベースライン削除", "移動平均後"], index=0, key="spectrum_type_select")
    
    second_deriv_smooth = st.sidebar.number_input(
        "2次微分平滑化:", 3, 35,
        step=2, key="second_deriv_smooth"
    )
    
    second_deriv_threshold = st.sidebar.number_input(
        "2次微分閾値:",
        min_value=0,
        max_value=1000,
        step=10,
        key="second_deriv_threshold"
    )
    
    peak_prominence_threshold = st.sidebar.number_input(
        "ピークProminence閾値:",
        min_value=0,
        max_value=1000,
        step=10,
        key="prominence_threshold"
    )

    # ファイルアップロード
    uploaded_files = st.file_uploader(
        "ラマンスペクトルをアップロードしてください（単数）", 
        accept_multiple_files=False, 
        key="file_uploader",
    )
    
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # st.sidebar.write("OPENAI_API_KEY is set? ", bool(os.getenv("OPENAI_API_KEY")))
    
    # アップロードファイル変更検出
    if uploaded_files:
        new_filenames = [uploaded_files.name]
    else:
        new_filenames = []
    prev_filenames = st.session_state.get("uploaded_filenames", [])

    # 設定変更検出
    config_keys = ["spectrum_type_select", "second_deriv_smooth", "second_deriv_threshold", "prominence_threshold"]
    config_changed = any(
        st.session_state.get(f"prev_{key}") != st.session_state[key] for key in config_keys
    )
    file_changed = new_filenames != prev_filenames

    # 手動ピーク初期化条件
    if config_changed or file_changed:
        for key in list(st.session_state.keys()):
            if key.endswith("_manual_peaks"):
                del st.session_state[key]
        st.session_state["manual_peak_keys"] = []
        st.session_state["uploaded_filenames"] = new_filenames
        for k in config_keys:
            st.session_state[f"prev_{k}"] = st.session_state[k]
            
    file_labels = []
    all_spectra = []
    all_bsremoval_spectra = []
    all_averemoval_spectra = []
    all_wavenum = []
    
    # セキュリティ状態表示
    if SECURITY_AVAILABLE:
        security_manager = get_security_manager()
        if security_manager:
            security_status = security_manager.get_security_status()
            
            with st.expander("🛡️ システム状態", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**データ保護機能:**")
                    st.write(f"🔐 暗号化: {'✅' if security_status['encryption_enabled'] else '❌'}")
                    st.write(f"🔍 完全性チェック: {'✅' if security_status['integrity_checking_enabled'] else '❌'}")
                    st.write(f"🛡️ アクセス制御: {'✅' if security_status['access_control_enabled'] else '❌'}")
                
                with col2:
                    st.write("**通信:**")
                    st.write(f"🌐 HTTPS強制: {'✅' if security_status['https_enforced'] else '❌'}")
                    st.write(f"📝 監査ログ: {'✅' if security_status['audit_logging_enabled'] else '❌'}")
                    st.write(f"🔑 キー: {'✅' if security_status['master_key_exists'] else '❌'}")
    else:
        st.warning("⚠️ セキュリティモジュールが無効です。基本機能のみ動作します。")
    
    # セキュリティログ記録
    if SECURITY_AVAILABLE:
        security_manager = get_security_manager()
        current_user = st.session_state.get('current_user', {})
        user_id = current_user.get('username', 'unknown')
        
        if security_manager:
            security_manager.audit_logger.log_security_event(
                event_type="PEAK_ANALYSIS_START",
                user_id=user_id,
                details={
                    'llm_ready': llm_ready,
                    'user_hint_provided': bool(user_hint),
                    'timestamp': datetime.now().isoformat()
                },
                severity="INFO"
            )
    
    if uploaded_files:
        # ファイル処理
        try:
            result = process_spectrum_file(
                uploaded_files, start_wavenum, end_wavenum, dssn_th, savgol_wsize
            )
            wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
            
            if wavenum is None:
                st.error(f"{file_name}の処理中にエラーが発生しました")
                return
            
            st.write(f"ファイルタイプ: {file_type} - {file_name}")
            
            file_labels.append(file_name)
            all_wavenum.append(wavenum)
            all_spectra.append(spectra)
            all_bsremoval_spectra.append(BSremoval_specta_pos)
            all_averemoval_spectra.append(Averemoval_specta_pos)
            
        except Exception as e:
            st.error(f"{uploaded_files.name}の処理中にエラーが発生しました: {e}")
        
        # ピーク検出の実行
        if 'peak_detection_triggered' not in st.session_state:
            st.session_state['peak_detection_triggered'] = False
    
        if st.button("ピーク検出を実行"):
            st.session_state['peak_detection_triggered'] = True
        
        if st.session_state['peak_detection_triggered']:
            perform_peak_detection_and_ai_analysis(
                file_labels, all_wavenum, all_bsremoval_spectra, all_averemoval_spectra,
                spectrum_type, second_deriv_smooth, second_deriv_threshold, peak_prominence_threshold,
                llm_connector, user_hint, llm_ready
            )
    
    # セキュリティ情報表示
    st.info("🔒 このモードでは、全てのファイル操作とAPI通信が安全に実行されます。")

def perform_peak_detection_and_ai_analysis(file_labels, all_wavenum, all_bsremoval_spectra, all_averemoval_spectra,
                                          spectrum_type, second_deriv_smooth, second_deriv_threshold, peak_prominence_threshold,
                                          llm_connector, user_hint, llm_ready):
    """ピーク検出とAI解析を実行"""
    st.subheader("ピーク検出結果")
    
    peak_results = []
    
    # 現在の設定を表示
    st.info(f"""
    **検出設定:**
    - スペクトルタイプ: {spectrum_type}
    - 2次微分平滑化: {second_deriv_smooth}, 閾値: {second_deriv_threshold} (ピーク検出用)
    - ピーク卓立度閾値: {peak_prominence_threshold}
    """)
    
    # ピーク検出の実行
    for i, file_name in enumerate(file_labels):
        if spectrum_type == "ベースライン削除":
            selected_spectrum = all_bsremoval_spectra[i]
        else:
            selected_spectrum = all_averemoval_spectra[i]
        
        wavenum = all_wavenum[i]
        
        # 2次微分計算
        if len(selected_spectrum) > second_deriv_smooth:
            second_derivative = savgol_filter(selected_spectrum, int(second_deriv_smooth), 2, deriv=2)
        else:
            second_derivative = np.gradient(np.gradient(selected_spectrum))
        
        # ピーク検出
        peaks, properties = find_peaks(-second_derivative, height=second_deriv_threshold)
        all_peaks, properties = find_peaks(-second_derivative)

        if len(peaks) > 0:
            prominences = peak_prominences(-second_derivative, peaks)[0]
            all_prominences = peak_prominences(-second_derivative, all_peaks)[0]

            # Prominence閾値でフィルタリング
            mask = prominences > peak_prominence_threshold
            filtered_peaks = peaks[mask]
            filtered_prominences = prominences[mask]
            
            # ピーク位置の補正
            corrected_peaks = []
            corrected_prominences = []
            
            for peak_idx, prom in zip(filtered_peaks, filtered_prominences):
                window_start = max(0, peak_idx - 2)
                window_end = min(len(selected_spectrum), peak_idx + 3)
                local_window = selected_spectrum[window_start:window_end]
                
                local_max_idx = np.argmax(local_window)
                corrected_idx = window_start + local_max_idx
            
                corrected_peaks.append(corrected_idx)
                
                # 警告を抑制してprominence計算
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        local_prom_values = peak_prominences(-second_derivative, [corrected_idx])
                        local_prom = local_prom_values[0][0] if len(local_prom_values[0]) > 0 else prom
                        # prominence値が0または負の場合は元の値を使用
                        if local_prom <= 0:
                            local_prom = max(0.001, prom)
                        corrected_prominences.append(local_prom)
                except Exception:
                    # エラー時は元のprominence値を使用
                    corrected_prominences.append(max(0.001, prom))
            
            filtered_peaks = np.array(corrected_peaks)
            filtered_prominences = np.array(corrected_prominences)
        else:
            filtered_peaks = np.array([])
            filtered_prominences = np.array([])
        
        # 結果を保存
        peak_data = {
            'file_name': file_name,
            'detected_peaks': filtered_peaks,
            'detected_prominences': filtered_prominences,
            'wavenum': wavenum,
            'spectrum': selected_spectrum,
            'second_derivative': second_derivative,
            'second_deriv_smooth': second_deriv_smooth,
            'second_deriv_threshold': second_deriv_threshold,
            'prominence_threshold': peak_prominence_threshold,
            'all_peaks': all_peaks,
            'all_prominences': all_prominences,
        }
        peak_results.append(peak_data)
        
        # 結果を表示
        # st.write(f"**{file_name}**")
        # st.write(f"検出されたピーク数: {len(filtered_peaks)} ")
        
        # ピーク情報をテーブルで表示
        if len(filtered_peaks) > 0:
            peak_wavenums = wavenum[filtered_peaks]
            peak_intensities = selected_spectrum[filtered_peaks]
            st.write("**検出されたピーク:**")
            peak_table = pd.DataFrame({
                'ピーク番号': range(1, len(peak_wavenums) + 1),
                '波数 (cm⁻¹)': [f"{wn:.1f}" for wn in peak_wavenums],
                '強度': [f"{intensity:.3f}" for intensity in peak_intensities],
                'Prominence': [f"{prom:.4f}" for prom in filtered_prominences]
            })
            st.table(peak_table)
        else:
            st.write("ピークが検出されませんでした")
    
    # ファイルごとの描画とAI解析
    for result in peak_results:
        render_peak_analysis_with_ai(result, spectrum_type, llm_connector, user_hint, llm_ready)

def render_peak_analysis_with_ai(result, spectrum_type, llm_connector, user_hint, llm_ready):
    """個別ファイルのピーク解析結果を描画してAI解析を実行"""
    file_key = result['file_name']

    # 初期化
    if f"{file_key}_excluded_peaks" not in st.session_state:
        st.session_state[f"{file_key}_excluded_peaks"] = set()
    if f"{file_key}_manual_peaks" not in st.session_state:
        st.session_state[f"{file_key}_manual_peaks"] = []

    # プロット描画
    render_interactive_plot(result, file_key, spectrum_type)
    
    # AI解析セクション
    render_ai_analysis_section(result, file_key, spectrum_type, llm_connector, user_hint, llm_ready)


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
        st.dataframe(manual_df, use_container_width=True, key=f"manual_peaks_table_{file_key}")
        
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
    
    # 元のスペクトルデータを保存（PDFレポート用）
    save_original_spectrum_data_to_session(result, file_key)
    
    # 【修正】一意のキーを追加してグラフ表示
    st.plotly_chart(fig, use_container_width=True, key=f"plotly_chart_{file_key}")

def render_ai_analysis_section(result, file_key, spectrum_type, llm_connector, user_hint, llm_ready):
    """AI解析セクションを描画"""
    st.markdown("---")
    st.subheader(f"AI解析 - {file_key}")
    
    # 最終的なピーク情報を収集
    final_peak_data = []
    
    # 有効な自動検出ピーク
    filtered_peaks = [
        i for i in result['detected_peaks']
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    filtered_prominences = [
        prom for i, prom in zip(result['detected_peaks'], result['detected_prominences'])
        if i not in st.session_state[f"{file_key}_excluded_peaks"]
    ]
    
    for idx, prom in zip(filtered_peaks, filtered_prominences):
        final_peak_data.append({
            'wavenumber': result['wavenum'][idx],
            'intensity': result['spectrum'][idx],
            'prominence': prom,
            'type': 'auto'
        })
    
    # 手動追加ピーク
    for x, y in st.session_state[f"{file_key}_manual_peaks"]:
        idx = np.argmin(np.abs(result['wavenum'] - x))
        try:
            # scipy警告を抑制してprominence計算
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prom_values = peak_prominences(-result['second_derivative'], [idx])
                prom = prom_values[0][0] if len(prom_values[0]) > 0 else 0.0
                # prominence値が0または負の場合のフォールバック
                if prom <= 0:
                    # 近傍の最大値を使用してprominenceを推定
                    window_start = max(0, idx - 5)
                    window_end = min(len(result['second_derivative']), idx + 6)
                    local_values = -result['second_derivative'][window_start:window_end]
                    if len(local_values) > 0:
                        prom = max(0.001, np.max(local_values) - np.min(local_values))
                    else:
                        prom = 0.001  # 最小値を設定
        except Exception as e:
            prom = 0.001  # エラー時のフォールバック値
        
        final_peak_data.append({
            'wavenumber': x,
            'intensity': y,
            'prominence': prom,
            'type': 'manual'
        })
    
    if final_peak_data:
        st.write(f"**最終確定ピーク数: {len(final_peak_data)}**")
        
        # ピーク表示
        peak_summary_df = pd.DataFrame([
            {
                'ピーク番号': i+1,
                '波数 (cm⁻¹)': f"{peak['wavenumber']:.1f}",
                '強度': f"{peak['intensity']:.3f}",
                'Prominence': f"{peak['prominence']:.3f}",
                'タイプ': '自動検出' if peak['type'] == 'auto' else '手動追加'
            }
            for i, peak in enumerate(final_peak_data)
        ])
        st.table(peak_summary_df)
        
        # AI解析実行ボタン
        ai_button_disabled = not (llm_ready and final_peak_data)
        if not llm_ready:
            st.warning("OpenAI APIが設定されていません。AI解析を実行するには、有効なAPIキーを入力してください。")
        
        if st.button(f"AI解析を実行 - {file_key}", key=f"ai_analysis_{file_key}", disabled=ai_button_disabled):
            perform_ai_analysis(file_key, final_peak_data, user_hint, llm_connector, peak_summary_df)
        
        # 過去の解析結果表示
        if f"{file_key}_ai_analysis" in st.session_state:
            with st.expander("📜 過去の解析結果を表示"):
                past_analysis = st.session_state[f"{file_key}_ai_analysis"]
                st.write(f"**解析日時:** {past_analysis['timestamp']}")
                st.write(f"**使用モデル:** {past_analysis['model']}")
                st.markdown("**解析結果:**")
                st.markdown(past_analysis['analysis'])
            
            
            # 質問応答セクションを表示
            if llm_ready:
                render_qa_section(
                    file_key=file_key,
                    analysis_context=st.session_state[f"{file_key}_ai_analysis"]['analysis_context'],
                    llm_connector=llm_connector
                )

            # レポートダウンロードセクション（追加質問の下）
            st.markdown("---")
            st.subheader("レポートダウンロード")
            
            # 過去の解析結果があるかチェック
            if f"{file_key}_ai_analysis" in st.session_state:
                # 解析結果から必要なデータを取得
                past_analysis = st.session_state[f"{file_key}_ai_analysis"]
                saved_peak_data = past_analysis.get('peak_data', [])
                saved_peak_summary_df = past_analysis.get('peak_summary_df', pd.DataFrame())
                saved_relevant_docs = past_analysis.get('relevant_docs', [])
                saved_user_hint = past_analysis.get('user_hint', '')
                
                # データベース情報を取得
                database_info = None
                database_files = []
                if hasattr(st.session_state, 'rag_system') and st.session_state.rag_system:
                    database_info = st.session_state.rag_system.get_database_info()
                    if database_info.get('source_files'):
                        database_files = database_info['source_files']
                
                col1, col2 = st.columns(2)
                
                # テキストレポートダウンロード
                with col1:
                    # テキストレポート生成
                    analysis_report = f"""ラマンスペクトル解析レポート
        ファイル名: {file_key}
        解析日時: {past_analysis['timestamp']}
        使用モデル: {past_analysis['model']}
        
        === データベース情報 ===
        """
                    if database_info:
                        analysis_report += f"""作成日時: {database_info.get('created_at', 'N/A')}
        作成者: {database_info.get('created_by', 'N/A')}
        総文献数: {database_info.get('n_docs', 0)}
        総チャンク数: {database_info.get('n_chunks', 0)}
        使用ファイル: {', '.join(database_files) if database_files else 'なし'}
        """
                    else:
                        analysis_report += "データベースは使用されていません。\n"
                    
                    if saved_user_hint:
                        analysis_report += f"\n=== AIへの補足ヒント ===\n{saved_user_hint}\n"
                    
                    analysis_report += f"""
        === 検出ピーク情報 ===
        {saved_peak_summary_df.to_string(index=False)}
        
        === AI解析結果 ===
        {past_analysis['analysis']}
        
        === 追加質問履歴 ===
        """
                    qa_history_key = f"{file_key}_qa_history"  # この行を追加
                    for i, qa in enumerate(st.session_state[qa_history_key], 1):
                        analysis_report += f"質問{i}: {qa['question']}\n回答{i}: {qa['answer']}\n質問日時: {qa['timestamp']}\n\n"
                    
                    analysis_report += "=== 参照文献 ===\n"
                    for i, doc in enumerate(saved_relevant_docs, 1):
                        analysis_report += f"{i}. {doc['metadata']['filename']}（類似度: {doc['similarity_score']:.3f}）\n"
                    
                    st.download_button(
                        label="テキストレポートをダウンロード",
                        data=analysis_report,
                        file_name=f"raman_analysis_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain",
                        key=f"download_text_report_{file_key}"
                    )
                
                # PDFレポートダウンロード
                with col2:
                    if PDF_GENERATION_AVAILABLE:
                        if st.button(f"📊 PDFレポートを生成", key=f"generate_comprehensive_pdf_{file_key}"):
                            try:
                                with st.spinner("PDFレポートを生成中..."):
                                    # PDFレポートジェネレーターを初期化
                                    pdf_generator = RamanPDFReportGenerator()
                                    
                                    # 現在表示されているPlotlyグラフを取得
                                    plotly_figure = st.session_state.get(f"{file_key}_plotly_figure", None)
                                    
                                    # Q&A履歴を取得
                                    qa_history_key = f"{file_key}_qa_history"
                                    qa_history = st.session_state[qa_history_key]
                                    
                                    # 元のスペクトルデータを取得
                                    original_spectrum_data = st.session_state.get(f"{file_key}_original_spectrum_data", None)
                                    
                                    # PDFを生成
                                    pdf_bytes = pdf_generator.generate_pdf_report(
                                        file_key=file_key,
                                        peak_data=saved_peak_data,
                                        analysis_result=past_analysis['analysis'],
                                        peak_summary_df=saved_peak_summary_df,
                                        plotly_figure=plotly_figure,
                                        relevant_docs=saved_relevant_docs,
                                        user_hint=saved_user_hint,
                                        qa_history=qa_history,
                                        database_info=database_info,
                                        database_files=database_files,
                                        original_spectrum_data=original_spectrum_data,
                                    )
                                    
                                    # 一時ファイルをクリーンアップ
                                    try:
                                        pdf_generator.cleanup_temp_files()
                                    except:
                                        pass
                                    
                                    # ダウンロードボタンを表示
                                    st.download_button(
                                        label="PDFレポートをダウンロード",
                                        data=pdf_bytes,
                                        file_name=f"raman_comprehensive_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                        mime="application/pdf",
                                        key=f"download_comprehensive_pdf_report_{file_key}"
                                    )
                                    
                                    st.success("✅ PDFレポートが正常に生成されました！")
                                    
                            except Exception as e:
                                st.error(f"PDFレポート生成エラー: {str(e)}")
                                # st.info("PDFレポート生成に必要なライブラリ（reportlab, Pillow）がインストールされていることを確認してください。")
                    else:
                        st.info("PDFレポート機能は利用できません（必要ライブラリ未インストール）")
            else:
                st.info("AI解析結果がないため、レポートを生成できません。先にAI解析を実行してください。")
            
    else:
        st.info("確定されたピークがありません。ピーク検出を実行するか、手動でピークを追加してください。")

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
            st.success("AIによる回答")
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

        except Exception as e:
            st.error(f"AI解析中にエラーが発生しました: {str(e)}")
            st.info("OpenAI APIの接続を確認してください。有効なAPIキーが設定されていることを確認してください。")
