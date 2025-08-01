# -*- coding: utf-8 -*-
"""
ピークAI解析モジュール
RAG機能とOpenAI APIを使用したラマンスペクトルの高度な解析
Enhanced with comprehensive features

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
from datetime import datetime
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from pathlib import Path
from common_utils import *
from peak_analysis_web import optimize_thresholds_via_gridsearch

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
            "OpenAI モデル選択",
            model_options,
            index=0,
            help="使用するOpenAIモデルを選択してください"
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
            
            st.sidebar.success(f"✅ OpenAI API接続設定完了 ({selected_model})")
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
            raise SecurityException("OpenAI モデルが設定されていません")
        
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
                
                st.success("✅ 回答が完了しました！")
                
            except Exception as e:
                st.error(f"質問処理中にエラーが発生しました: {str(e)}")
    
    elif submit_button and not user_question.strip():
        st.warning("質問内容を入力してください。")
    
    # 質問履歴のダウンロード
    if st.session_state[qa_history_key]:
        qa_report = generate_qa_report(file_key, st.session_state[qa_history_key])
        st.download_button(
            label="📥 質問履歴をダウンロード",
            data=qa_report,
            file_name=f"qa_history_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key=f"download_qa_{file_key}_{len(st.session_state[qa_history_key])}"
        )

def generate_qa_report(file_key, qa_history):
    """質問履歴レポートを生成する関数"""
    report_lines = [
        "ラマンスペクトル解析 - 質問履歴レポート",
        "=" * 50,
        f"ファイル名: {file_key}",
        f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"質問総数: {len(qa_history)}",
        "",
        "=" * 50,
        "質問履歴",
        "=" * 50,
        ""
    ]
    
    for i, qa in enumerate(qa_history, 1):
        report_lines.extend([
            f"質問{i}: {qa['question']}",
            f"回答{i}: {qa['answer']}",
            f"質問日時: {qa['timestamp']}",
            "-" * 30,
            ""
        ])
    
    return "\n".join(report_lines)

def peak_ai_analysis_mode():
    """強化されたPeak AI analysis mode"""
    if not PDF_AVAILABLE:
        st.error("AI解析機能を使用するには、以下のライブラリが必要です：")
        st.code("pip install PyPDF2 python-docx openai faiss-cpu sentence-transformers")
        return
    
    st.header("🔒 ラマンピークAI解析")
    
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
    
    # LLM接続設定（セキュリティ強化版）
    llm_connector = LLMConnector()
    
    # インターネット接続状態の表示
    if llm_connector.is_online:
        st.sidebar.success("🌐 インターネット接続: 正常")
        if llm_connector.ssl_context:
            st.sidebar.info("🔒 SSL/TLS暗号化: 有効")
    else:
        st.sidebar.error("❌ インターネット接続: 必要")
        st.error("この機能にはインターネット接続が必要です。")
        return
    
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
                    # セキュリティ強化ファイルアップロード
                    if security_manager:
                        upload_result = security_manager.secure_file_upload(uploaded_file, user_id)
                        if upload_result['status'] == 'success':
                            uploaded_count += 1
                        else:
                            st.error(f"ファイルアップロードエラー: {upload_result['message']}")
                    else:
                        # フォールバック: 基本的なファイル保存
                        save_path = os.path.join(TEMP_DIR, uploaded_file.name)
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        uploaded_count += 1
                
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
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    st.sidebar.write("OPENAI_API_KEY is set? ", bool(os.getenv("OPENAI_API_KEY")))
    
    # アップロードファイル変更検出
    new_filenames = [f.name for f in uploaded_files] if uploaded_files else []
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
    
    if uploaded_files:
        # ファイル処理
        for uploaded_file in uploaded_files:
            try:
                result = process_spectrum_file(
                    uploaded_file, start_wavenum, end_wavenum, dssn_th, savgol_wsize
                )
                wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
                
                if wavenum is None:
                    st.error(f"{file_name}の処理中にエラーが発生しました")
                    continue
                
                st.write(f"ファイルタイプ: {file_type} - {file_name}")
                
                file_labels.append(file_name)
                all_wavenum.append(wavenum)
                all_spectra.append(spectra)
                all_bsremoval_spectra.append(BSremoval_specta_pos)
                all_averemoval_spectra.append(Averemoval_specta_pos)
                
            except Exception as e:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました: {e}")
        
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
                
                local_prom = peak_prominences(-second_derivative, [corrected_idx])[0][0]
                corrected_prominences.append(local_prom)
            
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
        st.write(f"**{file_name}**")
        st.write(f"検出されたピーク数: {len(filtered_peaks)} (2次微分 + prominence判定)")
        
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
    """インタラクティブプロットを描画"""
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
    
    # クリック処理
    if plotly_events:
        event_key = f"{file_key}_click_event"
        clicked_points = plotly_events(
            fig,
            click_event=True,
            hover_event=False,
            select_event=False,
            override_height=800,
            key=event_key
        )
        
        clicked_main = [pt for pt in clicked_points if pt["curveNumber"] == 0]
        
        if clicked_main:
            pt = clicked_main[-1]
            click_id = str(pt['x']) + str(pt['y'])
        
            last_click_id = st.session_state.get(f"{event_key}_last", None)
            if click_id != last_click_id:
                st.session_state[f"{event_key}_last"] = click_id
        
                x = pt['x']
                y = pt['y']
                wavenum_arr = result['wavenum']
                idx = np.argmin(np.abs(wavenum_arr - x))
        
                # 自動検出ピークならトグル
                if idx in result['detected_peaks']:
                    if idx in st.session_state[f"{file_key}_excluded_peaks"]:
                        st.session_state[f"{file_key}_excluded_peaks"].remove(idx)
                    else:
                        st.session_state[f"{file_key}_excluded_peaks"].add(idx)
                else:
                    # 手動ピークの追加
                    is_duplicate = any(abs(existing_x - x) < 1.0 for existing_x, _ in st.session_state[f"{file_key}_manual_peaks"])
                    if not is_duplicate:
                        st.session_state[f"{file_key}_manual_peaks"].append((x, y))
    else:
        st.plotly_chart(fig, use_container_width=True)

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
            prom = peak_prominences(-result['second_derivative'], [idx])[0][0]
        except:
            prom = 0.0
        
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
    
    else:
        st.info("確定されたピークがありません。ピーク検出を実行するか、手動でピークを追加してください。")

def perform_ai_analysis(file_key, final_peak_data, user_hint, llm_connector, peak_summary_df):
    """AI解析を実行"""
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
                'analysis_context': full_response
            }

            # レポートテキスト生成
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

            # レポートダウンロードボタン
            st.download_button(
                label="解析レポートをダウンロード",
                data=analysis_report,
                file_name=f"raman_analysis_report_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"download_report_{file_key}"
            )

        except Exception as e:
            st.error(f"AI解析中にエラーが発生しました: {str(e)}")
            st.info("OpenAI APIの接続を確認してください。有効なAPIキーが設定されていることを確認してください。")
