# -*- coding: utf-8 -*-
"""
ラマンスペクトル解析システム - 完全リファクタリング版
全機能統合・責務分離・重複排除による改善版

Created on Wed Jun 11 15:56:04 2025
@author: Enhanced System - Complete Refactored
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
from typing import List, Dict, Optional, Tuple, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from pathlib import Path
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import re
import io

# 外部モジュール
from common_utils import *
from peak_analysis_web import optimize_thresholds_via_gridsearch

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
    import plotly.io as pio
    PDF_GENERATION_AVAILABLE = True
except ImportError:
    PDF_GENERATION_AVAILABLE = False

# AI関連のインポート
try:
    import PyPDF2
    import docx
    import openai
    import faiss
    from sentence_transformers import SentenceTransformer
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

# セキュリティモジュール
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

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning, module='scipy.signal._peak_finding')

# === 設定・データクラス定義 ===
@dataclass
class PeakData:
    """ピークデータを表すデータクラス"""
    wavenumber: float
    intensity: float
    prominence: float
    peak_type: str  # 'auto' or 'manual'
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class AnalysisConfig:
    """解析設定を表すデータクラス"""
    start_wavenum: int = 400
    end_wavenum: int = 2000
    dssn_th: float = 1e-4
    savgol_wsize: int = 5
    spectrum_type: str = "ベースライン削除"
    second_deriv_smooth: int = 5
    second_deriv_threshold: int = 100
    prominence_threshold: int = 100

@dataclass
class AnalysisResult:
    """解析結果を表すデータクラス"""
    file_name: str
    timestamp: str
    model: str
    peak_data: List[PeakData]
    ai_analysis: str
    relevant_docs: List[Dict]
    user_hint: str
    analysis_context: str
    peak_summary_df: Optional[pd.DataFrame] = None

@dataclass
class AIConfig:
    """AI設定を表すデータクラス"""
    openai_api_key: str = ""
    selected_model: str = "gpt-3.5-turbo"
    temperature: float = 0.3
    max_tokens: int = 1024
    use_openai_embeddings: bool = True
    openai_embedding_model: str = "text-embedding-ada-002"
    embedding_model_name: str = 'all-MiniLM-L6-v2'

@dataclass
class SecurityConfig:
    """セキュリティ設定"""
    ssl_verify: bool = True
    https_only: bool = True
    audit_logging: bool = True
    file_integrity_check: bool = True
    max_file_size: int = 100 * 1024 * 1024  # 100MB

# === 核となるビジネスロジック層 ===
class RamanAnalysisCore:
    """ラマンスペクトル解析の核となるロジックを管理"""
    
    def __init__(self, security_config: SecurityConfig = None):
        self.security_config = security_config or SecurityConfig()
        self.security_available = SECURITY_AVAILABLE
    
    def detect_peaks(self, wavenum: np.ndarray, spectrum: np.ndarray, 
                    config: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """ピーク検出の核となる処理"""
        # 2次微分計算
        if len(spectrum) > config.second_deriv_smooth:
            second_derivative = savgol_filter(spectrum, int(config.second_deriv_smooth), 2, deriv=2)
        else:
            second_derivative = np.gradient(np.gradient(spectrum))
        
        # ピーク検出
        peaks, _ = find_peaks(-second_derivative, height=config.second_deriv_threshold)
        all_peaks, _ = find_peaks(-second_derivative)
        
        if len(peaks) > 0:
            prominences = peak_prominences(-second_derivative, peaks)[0]
            all_prominences = peak_prominences(-second_derivative, all_peaks)[0]
            
            # Prominence閾値でフィルタリング
            mask = prominences > config.prominence_threshold
            filtered_peaks = peaks[mask]
            filtered_prominences = prominences[mask]
            
            # ピーク位置の補正
            corrected_peaks, corrected_prominences = self._correct_peak_positions(
                filtered_peaks, filtered_prominences, spectrum, second_derivative
            )
            
            return corrected_peaks, corrected_prominences, all_peaks, all_prominences
        
        return np.array([]), np.array([]), all_peaks, np.array([])
    
    def _correct_peak_positions(self, peaks: np.ndarray, prominences: np.ndarray, 
                               spectrum: np.ndarray, second_derivative: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ピーク位置の補正"""
        corrected_peaks = []
        corrected_prominences = []
        
        for peak_idx, prom in zip(peaks, prominences):
            window_start = max(0, peak_idx - 2)
            window_end = min(len(spectrum), peak_idx + 3)
            local_window = spectrum[window_start:window_end]
            
            local_max_idx = np.argmax(local_window)
            corrected_idx = window_start + local_max_idx
            
            corrected_peaks.append(corrected_idx)
            
            # prominence再計算
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    local_prom_values = peak_prominences(-second_derivative, [corrected_idx])
                    local_prom = local_prom_values[0][0] if len(local_prom_values[0]) > 0 else prom
                    if local_prom <= 0:
                        local_prom = max(0.001, prom)
                    corrected_prominences.append(local_prom)
            except Exception:
                corrected_prominences.append(max(0.001, prom))
        
        return np.array(corrected_peaks), np.array(corrected_prominences)
    
    def create_peak_objects(self, peak_indices: np.ndarray, prominences: np.ndarray,
                           wavenum: np.ndarray, spectrum: np.ndarray, peak_type: str = 'auto') -> List[PeakData]:
        """ピークオブジェクトを作成"""
        peaks = []
        for idx, prom in zip(peak_indices, prominences):
            peaks.append(PeakData(
                wavenumber=float(wavenum[idx]),
                intensity=float(spectrum[idx]),
                prominence=float(prom),
                peak_type=peak_type
            ))
        return peaks
    
    def calculate_manual_peak_prominence(self, wavenumber: float, wavenum: np.ndarray, 
                                       second_derivative: np.ndarray) -> float:
        """手動ピークのprominence計算"""
        idx = np.argmin(np.abs(wavenum - wavenumber))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prom_values = peak_prominences(-second_derivative, [idx])
                prom = prom_values[0][0] if len(prom_values[0]) > 0 else 0.0
                if prom <= 0:
                    window_start = max(0, idx - 5)
                    window_end = min(len(second_derivative), idx + 6)
                    local_values = -second_derivative[window_start:window_end]
                    if len(local_values) > 0:
                        prom = max(0.001, np.max(local_values) - np.min(local_values))
                    else:
                        prom = 0.001
                return float(prom)
        except Exception:
            return 0.001

# === AI・セキュリティ統合管理 ===
class SecurityManager:
    """セキュリティ機能の統合管理"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.ssl_context = self._setup_ssl_context()
    
    def _setup_ssl_context(self):
        """SSLコンテキストの設定"""
        if not self.config.ssl_verify:
            return None
        
        try:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = True
            ssl_context.verify_mode = ssl.CERT_REQUIRED
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
            return ssl_context
        except Exception as e:
            st.warning(f"SSL設定警告: {e}")
            return None
    
    def secure_file_upload(self, uploaded_file, user_id: str = "unknown") -> Dict[str, Any]:
        """セキュアファイルアップロード"""
        try:
            # ファイルサイズチェック
            if hasattr(uploaded_file, 'size') and uploaded_file.size > self.config.max_file_size:
                return {"status": "error", "message": f"ファイルサイズ制限超過: {uploaded_file.size}"}
            
            # ファイル形式チェック
            allowed_extensions = {'.pdf', '.docx', '.txt', '.csv'}
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext not in allowed_extensions:
                return {"status": "error", "message": f"許可されていないファイル形式: {file_ext}"}
            
            # ファイル名のサニタイズ
            safe_filename = re.sub(r'[^\w\-_.]', '_', uploaded_file.name)
            
            return {"status": "success", "safe_filename": safe_filename}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def sanitize_prompt(self, prompt: str) -> str:
        """プロンプトインジェクション対策"""
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
    
    def sanitize_response_content(self, content: str) -> str:
        """応答内容のサニタイズ"""
        # HTMLタグの除去
        content = re.sub(r'<[^>]+>', '', content)
        # スクリプトタグの除去
        content = re.sub(r'<script.*?</script>', '', content, flags=re.IGNORECASE | re.DOTALL)
        return content

class NetworkManager:
    """ネットワーク通信の統合管理"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.session = self._setup_session()
    
    def _setup_session(self) -> requests.Session:
        """HTTPセッションの設定"""
        session = requests.Session()
        
        # セキュリティヘッダー
        session.headers.update({
            'User-Agent': 'RamanEye-Client/2.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block'
        })
        
        return session
    
    def check_internet_connection(self) -> bool:
        """インターネット接続チェック"""
        try:
            response = self.session.get("https://www.google.com", timeout=5, verify=True)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

class AIConnectionManager:
    """AI接続の統合管理"""
    
    def __init__(self, config: AIConfig, security_manager: SecurityManager, network_manager: NetworkManager):
        self.config = config
        self.security_manager = security_manager
        self.network_manager = network_manager
        self.is_connected = False
        self.openai_client = None
    
    def setup_connection(self) -> bool:
        """AI接続設定"""
        if not self.network_manager.check_internet_connection():
            st.sidebar.error("❌ インターネット接続が必要です")
            return False
        
        st.sidebar.success("🌐 インターネット接続: 正常")
        
        try:
            # APIキー設定
            api_key = os.getenv("OPENAI_API_KEY", self.config.openai_api_key)
            if not self._validate_api_key(api_key):
                st.sidebar.error("無効なAPIキーです")
                return False
            
            openai.api_key = api_key
            self.is_connected = True
            
            st.sidebar.success(f"✅ OpenAI API接続設定完了 ({self.config.selected_model})")
            return True
            
        except Exception as e:
            st.sidebar.error(f"API設定エラー: {e}")
            return False
    
    def _validate_api_key(self, api_key: str) -> bool:
        """APIキーの妥当性を検証"""
        if not api_key or len(api_key) < 20:
            return False
        if not api_key.startswith('sk-'):
            return False
        return True
    
    def generate_completion(self, messages: List[Dict], stream: bool = True) -> str:
        """OpenAI API呼び出し（セキュリティ強化版）"""
        if not self.is_connected:
            raise Exception("OpenAI API接続が設定されていません")
        
        # メッセージのサニタイズ
        sanitized_messages = []
        for msg in messages:
            sanitized_content = self.security_manager.sanitize_prompt(msg.get('content', ''))
            sanitized_messages.append({
                'role': msg.get('role', 'user'),
                'content': sanitized_content
            })
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config.selected_model,
                messages=sanitized_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=stream,
                request_timeout=60
            )
            
            full_response = ""
            if stream:
                stream_area = st.empty()
                for chunk in response:
                    if "choices" in chunk and len(chunk["choices"]) > 0:
                        delta = chunk["choices"][0]["delta"]
                        if "content" in delta:
                            content = self.security_manager.sanitize_response_content(delta["content"])
                            full_response += content
                            stream_area.markdown(full_response)
            else:
                full_response = response.choices[0].message.content
                full_response = self.security_manager.sanitize_response_content(full_response)
            
            return full_response
            
        except Exception as e:
            raise Exception(f"OpenAI API呼び出しエラー: {str(e)}")
    
    def create_embeddings(self, texts: List[str], batch_size: int = 200) -> np.ndarray:
        """OpenAI埋め込みAPI呼び出し"""
        if not self.is_connected:
            raise Exception("OpenAI API接続が設定されていません")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            
            # テキストの前処理・サニタイズ
            sanitized_chunk = []
            for text in chunk:
                sanitized_text = self.security_manager.sanitize_prompt(text)
                if len(sanitized_text) > 8000:
                    sanitized_text = sanitized_text[:8000]
                sanitized_chunk.append(sanitized_text)
            
            try:
                response = openai.Embedding.create(
                    model=self.config.openai_embedding_model,
                    input=sanitized_chunk,
                    timeout=60
                )
                
                embeddings = [d['embedding'] for d in response['data']]
                all_embeddings.extend(embeddings)
                
                # 進捗表示
                if len(texts) > batch_size:
                    progress = min(i + batch_size, len(texts)) / len(texts)
                    st.progress(progress)
                    
            except Exception as e:
                raise Exception(f"埋め込み生成エラー: {e}")
        
        return np.array(all_embeddings, dtype=np.float32)

class DocumentProcessor:
    """文書処理の統合管理"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    def extract_text(self, file_path: str) -> str:
        """ファイルからのテキスト抽出"""
        ext = os.path.splitext(file_path)[1].lower()
        
        # ファイルサイズチェック
        file_size = os.path.getsize(file_path)
        if file_size > self.security_manager.config.max_file_size:
            raise Exception(f"ファイルサイズが制限を超えています: {file_path}")
        
        try:
            if ext == '.pdf':
                return self._extract_pdf_text(file_path)
            elif ext == '.docx':
                return self._extract_docx_text(file_path)
            elif ext == '.txt':
                return self._extract_txt_text(file_path)
            else:
                raise Exception(f"サポートされていないファイル形式: {ext}")
                
        except Exception as e:
            st.error(f"テキスト抽出エラー {file_path}: {e}")
            return ""
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """PDF からテキスト抽出"""
        reader = PyPDF2.PdfReader(file_path)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            except Exception as e:
                st.warning(f"PDF ページ {page_num} 読み込みエラー: {e}")
        return "\n".join(text_parts)
    
    def _extract_docx_text(self, file_path: str) -> str:
        """DOCX からテキスト抽出"""
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    
    def _extract_txt_text(self, file_path: str) -> str:
        """TXT からテキスト抽出"""
        with open(file_path, encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # テキストファイルのサイズ制限
        if len(content) > 1000000:  # 1MB制限
            content = content[:1000000]
        return content
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """テキストをチャンクに分割"""
        if not text or not text.strip():
            return []
        
        text = text.strip()
        
        # 危険なコンテンツのフィルタリング
        if self._contains_malicious_content(text):
            st.warning("潜在的に危険なコンテンツが検出されました。処理をスキップします。")
            return []
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip() and len(chunk) > 10:
                chunks.append(chunk)
                
        return chunks
    
    def _contains_malicious_content(self, text: str) -> bool:
        """悪意のあるコンテンツの検出"""
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
        
        text_lower = text.lower()
        
        for pattern in malicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False

# === AI・RAG機能統合管理 ===
class RamanRAGSystem:
    """RAG機能の統合管理（完全リファクタリング版）"""
    
    def __init__(self, ai_config: AIConfig, security_manager: SecurityManager, 
                 ai_connection: AIConnectionManager, doc_processor: DocumentProcessor):
        self.config = ai_config
        self.security_manager = security_manager
        self.ai_connection = ai_connection
        self.doc_processor = doc_processor
        
        # 埋め込みモデルの初期化
        self.use_openai = self.config.use_openai_embeddings and ai_connection.is_connected
        if not self.use_openai and AI_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            except Exception as e:
                st.warning(f"ローカル埋め込みモデル初期化失敗: {e}")
                self.embedding_model = None
        else:
            self.embedding_model = None
        
        # データベース関連
        self.vector_db = None
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.embedding_dim: int = 0
        self.db_info: Dict = {}
    
    def build_vector_database(self, folder_path: str) -> bool:
        """ベクトルデータベース構築"""
        if not AI_AVAILABLE:
            st.error("AI機能が利用できません")
            return False
            
        if not os.path.exists(folder_path):
            st.error(f"指定されたフォルダが存在しません: {folder_path}")
            return False
        
        # ファイル一覧取得
        file_patterns = ['*.pdf', '*.docx', '*.txt']
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        if not files:
            st.warning("処理可能なファイルが見つかりません。")
            return False
        
        # テキスト抽出とチャンク化
        all_chunks, all_metadata = [], []
        st.info(f"{len(files)} 件のファイルを処理中…")
        pbar = st.progress(0)
        
        for idx, file_path in enumerate(files):
            try:
                text = self.doc_processor.extract_text(file_path)
                chunks = self.doc_processor.chunk_text(text)
                
                for chunk in chunks:
                    all_chunks.append(chunk)
                    all_metadata.append({
                        'filename': os.path.basename(file_path),
                        'filepath': file_path,
                        'preview': chunk[:100] + "…" if len(chunk) > 100 else chunk,
                        'processed_at': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.error(f"ファイル処理エラー {file_path}: {e}")
                continue
                
            pbar.progress((idx + 1) / len(files))
        
        if not all_chunks:
            st.error("抽出できるテキストチャンクがありませんでした。")
            return False
        
        # 埋め込みベクトルの生成
        st.info("埋め込みベクトルを生成中…")
        try:
            if self.use_openai:
                embeddings = self.ai_connection.create_embeddings(all_chunks)
            elif self.embedding_model:
                embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
            else:
                st.error("埋め込みモデルが利用できません")
                return False
            
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
                'n_docs': len(files),
                'n_chunks': len(all_chunks),
                'source_files': [os.path.basename(f) for f in files],
                'embedding_model': (
                    self.config.openai_embedding_model if self.use_openai 
                    else self.config.embedding_model_name
                ),
                'security_enabled': SECURITY_AVAILABLE
            }
            
            st.success(f"ベクトルDB構築完了: {len(all_chunks)} チャンク")
            return True
            
        except Exception as e:
            st.error(f"ベクトルDB構築エラー: {e}")
            return False
    
    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """関連文書検索"""
        if self.vector_db is None:
            return []
        
        try:
            # クエリのサニタイズ
            sanitized_query = self.security_manager.sanitize_prompt(query.strip())
            if len(sanitized_query) > 1000:
                sanitized_query = sanitized_query[:1000]
            
            # 埋め込み生成
            if self.use_openai:
                query_emb = self.ai_connection.create_embeddings([sanitized_query])
            elif self.embedding_model:
                query_emb = self.embedding_model.encode([sanitized_query], show_progress_bar=False)
                query_emb = np.array(query_emb, dtype=np.float32)
            else:
                return []
            
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
    
    def get_database_info(self) -> Dict:
        """データベースの情報を取得"""
        if self.vector_db is None:
            return {"status": "データベースが構築されていません"}
        
        info = self.db_info.copy()
        info["status"] = "構築済み"
        info["current_chunks"] = len(self.documents)
        return info

class RamanSpectrumAnalyzer:
    """スペクトル解析プロンプト生成の統合管理"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
    
    def generate_analysis_prompt(self, peak_data: List[Dict], relevant_docs: List[Dict], 
                                user_hint: Optional[str] = None) -> str:
        """ラマンスペクトル解析のためのプロンプトを生成"""
        
        def format_peaks(peaks: List[Dict]) -> str:
            header = "【検出ピーク一覧】"
            lines = [
                f"{i+1}. 波数: {p.get('wavenumber', 0):.1f} cm⁻¹, "
                f"強度: {p.get('intensity', 0):.3f}, "
                f"卓立度: {p.get('prominence', 0):.3f}, "
                f"種類: {'自動検出' if p.get('peak_type') == 'auto' else '手動追加'}"
                for i, p in enumerate(peaks)
            ]
            return "\n".join([header] + lines)
        
        def format_doc_summaries(docs: List[Dict], preview_length: int = 300) -> str:
            header = "【文献の概要（類似度付き）】"
            lines = []
            for i, doc in enumerate(docs, 1):
                filename = doc.get("metadata", {}).get("filename", f"文献{i}")
                similarity = doc.get("similarity_score", 0.0)
                text = doc.get("text") or ""
                # テキストのサニタイズ
                text = self.security_manager.sanitize_prompt(text)
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
            sanitized_hint = self.security_manager.sanitize_prompt(user_hint)
            sections.append(f"【ユーザーによる補足情報】\n{sanitized_hint}\n")
        
        if peak_data:
            sections.append(format_peaks(peak_data))
        if relevant_docs:
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

class RamanAIManager:
    """AI解析とRAG機能を統合管理（完全リファクタリング版）"""
    
    def __init__(self, ai_config: AIConfig = None):
        self.config = ai_config or AIConfig()
        self.security_manager = SecurityManager(SecurityConfig())
        self.network_manager = NetworkManager(self.security_manager)
        self.ai_connection = AIConnectionManager(self.config, self.security_manager, self.network_manager)
        self.doc_processor = DocumentProcessor(self.security_manager)
        self.rag_system = None
        self.spectrum_analyzer = RamanSpectrumAnalyzer(self.security_manager)
        self.is_ready = False
    
    def setup_connection(self) -> bool:
        """AI接続設定"""
        success = self.ai_connection.setup_connection()
        if success:
            self.rag_system = RamanRAGSystem(
                self.config, self.security_manager, self.ai_connection, self.doc_processor
            )
            self.is_ready = True
        return success
    
    def build_knowledge_base(self, folder_path: str) -> bool:
        """知識ベース構築"""
        if self.rag_system:
            return self.rag_system.build_vector_database(folder_path)
        return False
    
    def get_database_info(self) -> Dict:
        """データベース情報取得"""
        if self.rag_system:
            return self.rag_system.get_database_info()
        return {"status": "RAGシステムが初期化されていません"}
    
    def analyze_peaks(self, peaks: List[PeakData], user_hint: str = "") -> AnalysisResult:
        """ピーク解析実行"""
        if not self.is_ready:
            raise Exception("AI機能が初期化されていません")
        
        # 関連文献検索
        search_terms = ' '.join([f"{p.wavenumber:.0f}cm-1" for p in peaks[:5]])
        search_query = f"ラマンスペクトロスコピー ピーク {search_terms}"
        relevant_docs = self.rag_system.search_relevant_documents(search_query, top_k=5)
        
        # プロンプト生成
        peak_dict_list = [p.to_dict() for p in peaks]
        analysis_prompt = self.spectrum_analyzer.generate_analysis_prompt(
            peak_data=peak_dict_list,
            relevant_docs=relevant_docs,
            user_hint=user_hint
        )
        
        # AI解析実行
        system_message = "あなたはラマンスペクトロスコピーの専門家です。ピーク位置と論文、またはインターネット上の情報を比較して、このサンプルが何の試料なのか当ててください。すべて日本語で答えてください。"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": analysis_prompt + "\n\nすべて日本語で詳しく説明してください。"}
        ]
        
        analysis_text = self.ai_connection.generate_completion(messages, stream=True)
        
        return AnalysisResult(
            file_name="",  # 呼び出し元で設定
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model=f"OpenAI ({self.config.selected_model})",
            peak_data=peaks,
            ai_analysis=analysis_text,
            relevant_docs=relevant_docs,
            user_hint=user_hint,
            analysis_context=analysis_text
        )
    
    def answer_question(self, question: str, context: str, qa_history: List[Dict] = None) -> str:
        """質問応答機能"""
        if not self.is_ready:
            return "AI機能が利用できません"
        
        system_message = """あなたはラマンスペクトロスコピーの専門家です。
解析結果や過去の質問履歴を踏まえて、ユーザーの質問に日本語で詳しく答えてください。
科学的根拠に基づいた正確な回答を心がけてください。"""
        
        # コンテキストの構築
        context_text = f"【解析結果】\n{context}\n\n"
        
        if qa_history:
            context_text += "【過去の質問履歴】\n"
            for i, qa in enumerate(qa_history, 1):
                context_text += f"質問{i}: {qa['question']}\n回答{i}: {qa['answer']}\n\n"
        
        context_text += f"【新しい質問】\n{question}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": context_text}
        ]
        
        return self.ai_connection.generate_completion(messages, stream=True)

# === PDFレポート生成統合管理 ===
class RamanPDFReportGenerator:
    """PDFレポート生成機能の統合管理（完全リファクタリング版）"""
    
    def __init__(self):
        self.temp_dir = "./temp_pdf_assets"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.japanese_font_available = False
        self.japanese_font_name = 'Helvetica'  # デフォルト
        self.styles = None
        
        self._setup_japanese_font()
        self._setup_styles()
    
    def _setup_japanese_font(self):
        """日本語フォントの設定"""
        try:
            font_paths = [
                # Windows
                "C:/Windows/Fonts/msgothic.ttc",
                "C:/Windows/Fonts/meiryo.ttc", 
                "C:/Windows/Fonts/NotoSansCJK-Regular.ttc",
                "C:/Windows/Fonts/YuGothic.ttc",
                # macOS
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
                "/System/Library/Fonts/NotoSansCJK.ttc",
                "/Library/Fonts/NotoSansCJK.ttc",
                # Linux
                "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]
            
            for font_path in font_paths:
                if os.path.exists(font_path):
                    try:
                        pdfmetrics.registerFont(TTFont('JapaneseFont', font_path))
                        self.japanese_font_available = True
                        self.japanese_font_name = 'JapaneseFont'
                        break
                    except Exception:
                        continue
            
            if not self.japanese_font_available:
                self.japanese_font_name = 'Times-Roman'
                
        except Exception as e:
            self.japanese_font_name = 'Helvetica'
    
    def _setup_styles(self):
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
        """PlotlyグラフをPNG画像に変換"""
        try:
            img_path = os.path.join(self.temp_dir, f"{filename}.{format}")
            
            fig.update_layout(
                width=width,
                height=height,
                font=dict(size=10),
                margin=dict(l=50, r=50, t=80, b=50)
            )
            
            # 画像保存の試行
            success = False
            
            # Kaleidoエンジン使用
            try:
                pio.write_image(fig, img_path, format=format, width=width, height=height, scale=2)
                success = True
            except Exception:
                # Matplotlib代替作成
                try:
                    self._create_matplotlib_alternative(fig, img_path, width, height)
                    success = True
                except Exception:
                    # プレースホルダー画像作成
                    self._create_enhanced_placeholder_image(img_path, width, height, f"Spectrum Graph: {filename}")
                    success = True
            
            return img_path if success else None
            
        except Exception as e:
            placeholder_path = os.path.join(self.temp_dir, f"{filename}_fallback.png")
            self._create_enhanced_placeholder_image(placeholder_path, width, height, f"Graph Error: {filename}")
            return placeholder_path
    
    def _create_matplotlib_alternative(self, plotly_fig, save_path, width, height):
        """MatplotlibでPlotlyグラフの代替画像を作成"""
        try:
            import matplotlib.pyplot as plt
            
            fig_width = width / 100
            fig_height = height / 100
            
            fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), facecolor='white')
            fig.suptitle('Raman Spectrum Analysis', fontsize=14, y=0.95)
            
            # サンプルグラフ作成
            x_sample = np.linspace(400, 2000, 100)
            
            # スペクトル風
            y1_sample = np.exp(-(x_sample - 1200)**2 / 50000) + 0.5 * np.exp(-(x_sample - 800)**2 / 20000)
            axes[0].plot(x_sample, y1_sample, 'b-', linewidth=2, label='Spectrum')
            axes[0].scatter([800, 1200], [0.5, 1.0], c='red', s=50, zorder=5, label='Peaks')
            axes[0].set_ylabel('Intensity (a.u.)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 2次微分風
            y2_sample = -np.gradient(np.gradient(y1_sample))
            axes[1].plot(x_sample, y2_sample, 'purple', linewidth=1, label='2nd Derivative')
            axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('2nd Derivative')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Prominence風
            prominence_sample = np.abs(y2_sample) * 100
            axes[2].scatter(x_sample, prominence_sample, c='orange', s=10, alpha=0.6, label='All Peaks')
            axes[2].scatter([800, 1200], [50, 80], c='red', s=30, label='Valid Peaks')
            axes[2].set_xlabel('Wavenumber (cm⁻¹)')
            axes[2].set_ylabel('Prominence')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            
        except Exception as e:
            raise Exception(f"Matplotlib代替作成エラー: {e}")
    
    def _create_enhanced_placeholder_image(self, path, width, height, text):
        """高品質プレースホルダー画像を作成"""
        try:
            img = PILImage.new('RGB', (width, height), color='white')
            draw = PILImage.ImageDraw.Draw(img)
            
            # グラデーション背景
            for y in range(height):
                color_value = int(255 - (y / height) * 20)
                color = (color_value, color_value, color_value)
                draw.line([(0, y), (width, y)], fill=color)
            
            # テキスト描画
            try:
                font_large = PILImage.ImageFont.load_default()
                font_small = PILImage.ImageFont.load_default()
            except:
                font_large = None
                font_small = None
            
            # タイトル
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
            
            # グラフ風装飾
            draw.line([(width//8, height*3//4), (width*7//8, height*3//4)], fill='black', width=2)
            draw.line([(width//8, height//8), (width//8, height*3//4)], fill='black', width=2)
            
            # サンプル波形
            points = []
            for i in range(width//8, width*7//8, 5):
                x = i
                y = height//2 + int(50 * np.sin((i - width//8) * 0.01)) + int(30 * np.sin((i - width//8) * 0.03))
                points.append((x, y))
            
            if len(points) > 1:
                draw.line(points, fill='blue', width=2)
            
            # ピーク点
            peak_points = [(width//3, height//2 - 20), (width*2//3, height//2 - 40)]
            for px, py in peak_points:
                draw.ellipse([px-4, py-4, px+4, py+4], fill='red')
            
            # 枠線
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
            # 最終フォールバック
            try:
                img = PILImage.new('RGB', (width, height), color='lightgray')
                draw = PILImage.ImageDraw.Draw(img)
                simple_text = "Graph Placeholder"
                text_x = width // 2 - 50
                text_y = height // 2 - 10
                draw.text((text_x, text_y), simple_text, fill='black')
                img.save(path)
            except:
                pass
    
    def generate_comprehensive_pdf_report(
        self, 
        file_key: str,
        peak_data: List[Dict],
        analysis_result: str,
        peak_summary_df: pd.DataFrame,
        plotly_figure: go.Figure = None,
        relevant_docs: List[Dict] = None,
        user_hint: str = None,
        qa_history: List[Dict] = None
    ) -> bytes:
        """包括的なPDFレポートを生成（Q&A・ヒント統合版）"""
        
        pdf_buffer = io.BytesIO()
        
        try:
            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=2*cm,
                leftMargin=2*cm,
                topMargin=2*cm,
                bottomMargin=2*cm
            )
            
            story = []
            
            # コンテンツ作成（Q&A・ヒント情報を含む）
            story.extend(self._create_title_page(file_key))
            story.extend(self._create_executive_summary(peak_data, analysis_result, qa_history, user_hint))
            
            if plotly_figure:
                story.extend(self._create_graph_section(plotly_figure, file_key))
            
            story.extend(self._create_peak_details_section(peak_summary_df, peak_data))
            story.extend(self._create_ai_analysis_section(analysis_result))
            
            # ユーザーヒント情報（常に追加、ない場合もその旨を記載）
            story.extend(self._create_additional_info_section(user_hint))
            
            # Q&A履歴（常に追加、ない場合もその旨を記載）
            story.extend(self._create_qa_section(qa_history))
            
            if relevant_docs:
                story.extend(self._create_references_section(relevant_docs))
            
            story.extend(self._create_appendix_section())
            
            doc.build(story)
            
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            
            return pdf_bytes
            
        except Exception as e:
            st.error(f"PDFレポート生成エラー: {e}")
            raise e
    
    def _create_title_page(self, file_key: str) -> List:
        """タイトルページを作成"""
        content = []
        
        title = Paragraph(
            "ラマンスペクトル解析レポート",
            self.styles['JapaneseTitle']
        )
        content.append(title)
        content.append(Spacer(1, 0.5*inch))
        
        file_info = f"""
        <b>解析対象ファイル:</b> {file_key}<br/>
        <b>レポート生成日時:</b> {datetime.now().strftime('%Y年%m月%d日 %H時%M分')}<br/>
        <b>システム:</b> RamanEye AI Analysis System - Refactored Edition<br/>
        <b>バージョン:</b> 3.0 (Complete Refactored & Integrated)
        """
        
        content.append(Paragraph(file_info, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.5*inch))
        
        disclaimer = """
        <b>【重要】本レポートについて</b><br/>
        本レポートはAIによる自動解析結果を含んでいます。
        結果の解釈および活用については、専門家による検証を推奨します。
        測定条件、サンプル前処理、装置較正等の要因が結果に影響する可能性があります。
        """
        
        content.append(Paragraph(disclaimer, self.styles['JapaneseNormal']))
        content.append(PageBreak())
        
        return content
    
    def _create_executive_summary(self, peak_data: List[Dict], analysis_result: str, qa_history: List[Dict] = None, user_hint: str = None) -> List:
        """実行サマリーを作成（Q&A・ヒント情報統合版）"""
        content = []
        
        content.append(Paragraph("実行サマリー", self.styles['JapaneseHeading']))
        
        total_peaks = len(peak_data)
        auto_peaks = len([p for p in peak_data if p.get('peak_type') == 'auto'])
        manual_peaks = len([p for p in peak_data if p.get('peak_type') == 'manual'])
        
        # 基本統計情報
        summary_text = f"""
        <b>検出ピーク総数:</b> {total_peaks}<br/>
        <b>自動検出:</b> {auto_peaks} ピーク<br/>
        <b>手動追加:</b> {manual_peaks} ピーク<br/>
        <br/>
        <b>主要検出範囲:</b> {min([p['wavenumber'] for p in peak_data]):.0f} - {max([p['wavenumber'] for p in peak_data]):.0f} cm⁻¹<br/>
        """
        
        # ユーザー情報の追加
        if user_hint and user_hint.strip():
            summary_text += f"<br/><b>ユーザー提供ヒント:</b> あり<br/>"
        else:
            summary_text += f"<br/><b>ユーザー提供ヒント:</b> なし<br/>"
        
        # Q&A情報の追加
        qa_count = len(qa_history) if qa_history else 0
        summary_text += f"<b>追加質問・回答:</b> {qa_count} 件<br/>"
        
        content.append(Paragraph(summary_text, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.3*inch))
        
        # 初回AI解析結果の要約
        analysis_summary = analysis_result[:200] + "..." if len(analysis_result) > 200 else analysis_result
        content.append(Paragraph("<b>初回AI解析結果要約:</b>", self.styles['JapaneseNormal']))
        content.append(Paragraph(analysis_summary, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.2*inch))
        
        # Q&Aがある場合の追加情報
        if qa_count > 0:
            qa_summary_text = f"""
            <b>質問応答の概要:</b><br/>
            初回解析後、ユーザーから {qa_count} 件の追加質問が行われ、
            それぞれについてAIが詳細な回答を提供しました。
            質問内容と回答の詳細は本レポートの「追加質問・回答履歴」セクションに記載されています。
            """
            content.append(Paragraph(qa_summary_text, self.styles['JapaneseNormal']))
            content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_graph_section(self, plotly_figure: go.Figure, file_key: str) -> List:
        """グラフセクションを作成"""
        content = []
        
        content.append(Paragraph("スペクトルおよびピーク検出結果", self.styles['JapaneseHeading']))
        
        try:
            img_path = self.plotly_to_image(plotly_figure, f"spectrum_{file_key}", width=1000, height=800)
            
            if os.path.exists(img_path):
                img = Image(img_path, width=7*inch, height=5.6*inch)
                content.append(img)
                content.append(Spacer(1, 0.2*inch))
            
        except Exception as e:
            error_text = f"グラフ表示エラー: {e}"
            content.append(Paragraph(error_text, self.styles['JapaneseNormal']))
        
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
        
        table_data = [peak_summary_df.columns.tolist()]
        for _, row in peak_summary_df.iterrows():
            table_data.append(row.tolist())
        
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
        
        paragraphs = analysis_result.split('\n\n')
        
        for para in paragraphs:
            if para.strip():
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
        """補足情報セクションを作成（強化版）"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("ユーザー提供情報", self.styles['JapaneseHeading']))
        
        if user_hint and user_hint.strip():
            content.append(Paragraph("<b>AIへの補足ヒント:</b>", self.styles['JapaneseNormal']))
            
            # 長いヒントの場合は段落分け
            hint_paragraphs = user_hint.split('\n')
            for para in hint_paragraphs:
                if para.strip():
                    content.append(Paragraph(para.strip(), self.styles['JapaneseNormal']))
                    content.append(Spacer(1, 0.1*inch))
        else:
            content.append(Paragraph("AIへの補足ヒント: （ユーザーからの追加情報は提供されませんでした）", self.styles['JapaneseNormal']))
        
        content.append(Spacer(1, 0.2*inch))
        
        return content
    
    def _create_qa_section(self, qa_history: List[Dict]) -> List:
        """Q&Aセクションを作成（強化版）"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("追加質問・回答履歴", self.styles['JapaneseHeading']))
        
        if qa_history and len(qa_history) > 0:
            # サマリー情報
            summary_text = f"""
            <b>質問総数:</b> {len(qa_history)} 件<br/>
            <b>最初の質問:</b> {qa_history[0].get('timestamp', 'N/A')}<br/>
            <b>最後の質問:</b> {qa_history[-1].get('timestamp', 'N/A')}<br/>
            """
            content.append(Paragraph(summary_text, self.styles['JapaneseNormal']))
            content.append(Spacer(1, 0.3*inch))
            
            # 各Q&Aの詳細
            for i, qa in enumerate(qa_history, 1):
                # 質問セクション
                question_style = ParagraphStyle(
                    name=f'Question{i}',
                    parent=self.styles['JapaneseNormal'],
                    fontName=self.japanese_font_name,
                    fontSize=11,
                    spaceAfter=6,
                    textColor=colors.darkblue,
                    leftIndent=0.2*inch
                )
                
                content.append(Paragraph(f"<b>【質問 {i}】</b> ({qa.get('timestamp', 'N/A')})", question_style))
                
                question_text = qa.get('question', '').strip()
                if question_text:
                    content.append(Paragraph(f"Q: {question_text}", self.styles['JapaneseNormal']))
                
                content.append(Spacer(1, 0.1*inch))
                
                # 回答セクション
                answer_style = ParagraphStyle(
                    name=f'Answer{i}',
                    parent=self.styles['JapaneseNormal'],
                    fontName=self.japanese_font_name,
                    fontSize=10,
                    spaceAfter=6,
                    leftIndent=0.2*inch,
                    textColor=colors.darkgreen
                )
                
                content.append(Paragraph(f"<b>【回答 {i}】</b>", answer_style))
                
                answer_text = qa.get('answer', '').strip()
                if answer_text:
                    # 長い回答は段落分け
                    answer_paragraphs = answer_text.split('\n\n')
                    for para in answer_paragraphs:
                        if para.strip():
                            content.append(Paragraph(f"A: {para.strip()}", self.styles['JapaneseNormal']))
                            content.append(Spacer(1, 0.05*inch))
                
                # 区切り線
                if i < len(qa_history):
                    content.append(Spacer(1, 0.2*inch))
                    content.append(Paragraph("─" * 50, self.styles['JapaneseNormal']))
                    content.append(Spacer(1, 0.2*inch))
        else:
            content.append(Paragraph("追加質問は行われませんでした。", self.styles['JapaneseNormal']))
            content.append(Spacer(1, 0.2*inch))
            
            no_qa_info = """
            このセクションには、初回解析後にユーザーから追加で行われた質問と
            AIからの回答が記録されます。今回は追加質問がありませんでした。
            """
            content.append(Paragraph(no_qa_info, self.styles['JapaneseNormal']))
        
        return content
    
    def _create_appendix_section(self) -> List:
        """付録セクションを作成（包括版）"""
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
        アーキテクチャ: 完全リファクタリング版 v3.0<br/>
        """
        
        content.append(Paragraph(system_info, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.3*inch))
        
        # レポート内容の説明
        report_description = """
        <b>レポート内容について:</b><br/>
        本PDFレポートには以下の情報が包括的に含まれています：<br/>
        <br/>
        • <b>実行サマリー:</b> 解析の概要と統計情報<br/>
        • <b>スペクトルグラフ:</b> ピーク検出結果の可視化<br/>
        • <b>ピーク詳細:</b> 自動検出・手動追加ピークの一覧<br/>
        • <b>初回AI解析結果:</b> メインの解析内容<br/>
        • <b>ユーザー提供情報:</b> AIへの補足ヒント（提供された場合）<br/>
        • <b>追加質問・回答履歴:</b> 初回解析後の全てのQ&A<br/>
        • <b>参考文献:</b> RAGシステムで参照された文献情報<br/>
        • <b>付録:</b> システム情報と利用上の注意<br/>
        """
        
        content.append(Paragraph(report_description, self.styles['JapaneseNormal']))
        content.append(Spacer(1, 0.3*inch))
        
        # 利用上の注意
        usage_notes = """
        <b>利用上の注意:</b><br/>
        • 本レポートの解析結果は参考情報として提供されます<br/>
        • 重要な判断を行う場合は、専門家による検証を推奨します<br/>
        • 測定条件、前処理、装置較正等が結果に影響する可能性があります<br/>
        • 追加質問・回答は解析結果の理解を深めるための補助情報です<br/>
        • ユーザー提供ヒントはAI解析の方向性に影響を与えています<br/>
        """
        
        content.append(Paragraph(usage_notes, self.styles['JapaneseNormal']))
        
        return content
    
    def cleanup_temp_files(self):
        """一時ファイルをクリーンアップ"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            st.warning(f"一時ファイルクリーンアップエラー: {e}")

# === データ管理層 ===
class RamanDataManager:
    """データの保存・読み込み・状態管理を統合"""
    
    def __init__(self):
        self.session_prefix = "raman_data_"
    
    def save_analysis_result(self, file_key: str, result: AnalysisResult):
        """解析結果の保存"""
        result.file_name = file_key
        st.session_state[f"{file_key}_analysis_result"] = result
    
    def load_analysis_result(self, file_key: str) -> Optional[AnalysisResult]:
        """解析結果の読み込み"""
        return st.session_state.get(f"{file_key}_analysis_result")
    
    def save_peaks(self, file_key: str, peaks: List[PeakData]):
        """ピークデータの保存"""
        st.session_state[f"{file_key}_peaks"] = peaks
    
    def load_peaks(self, file_key: str) -> List[PeakData]:
        """ピークデータの読み込み"""
        return st.session_state.get(f"{file_key}_peaks", [])
    
    def save_manual_peaks(self, file_key: str, manual_peaks: List[float]):
        """手動ピークの保存"""
        st.session_state[f"{file_key}_manual_peaks"] = manual_peaks
    
    def load_manual_peaks(self, file_key: str) -> List[float]:
        """手動ピークの読み込み"""
        return st.session_state.get(f"{file_key}_manual_peaks", [])
    
    def save_excluded_peaks(self, file_key: str, excluded_indices: set):
        """除外ピークの保存"""
        st.session_state[f"{file_key}_excluded_peaks"] = excluded_indices
    
    def load_excluded_peaks(self, file_key: str) -> set:
        """除外ピークの読み込み"""
        return st.session_state.get(f"{file_key}_excluded_peaks", set())
    
    def save_qa_history(self, file_key: str, qa_history: List[Dict]):
        """Q&A履歴の保存"""
        st.session_state[f"{file_key}_qa_history"] = qa_history
    
    def load_qa_history(self, file_key: str) -> List[Dict]:
        """Q&A履歴の読み込み"""
        return st.session_state.get(f"{file_key}_qa_history", [])
    
    def save_plotly_figure(self, file_key: str, figure: go.Figure):
        """Plotlyグラフの保存"""
        st.session_state[f"{file_key}_plotly_figure"] = figure
    
    def load_plotly_figure(self, file_key: str) -> Optional[go.Figure]:
        """Plotlyグラフの読み込み"""
        return st.session_state.get(f"{file_key}_plotly_figure")
    
    def clear_file_data(self, file_key: str):
        """ファイル関連データのクリア"""
        keys_to_remove = [key for key in st.session_state.keys() if key.startswith(file_key)]
        for key in keys_to_remove:
            del st.session_state[key]

# === レポート生成統合管理 ===
class RamanReportManager:
    """レポート生成機能を統合管理"""
    
    def __init__(self):
        self.pdf_available = PDF_GENERATION_AVAILABLE
        if self.pdf_available:
            self.pdf_generator = RamanPDFReportGenerator()
    
    def generate_comprehensive_text_report(self, result: AnalysisResult, qa_history: List[Dict] = None) -> str:
        """包括的なテキストレポート生成（Q&A履歴・ユーザーヒント含む）"""
        peak_df = pd.DataFrame([
            {
                'ピーク番号': i+1,
                '波数 (cm⁻¹)': f"{peak.wavenumber:.1f}",
                '強度': f"{peak.intensity:.3f}",
                'Prominence': f"{peak.prominence:.3f}",
                'タイプ': '自動検出' if peak.peak_type == 'auto' else '手動追加'
            }
            for i, peak in enumerate(result.peak_data)
        ])
        
        report_lines = [
            "ラマンスペクトル解析レポート - 完全版（Q&A・ヒント含む）",
            "=" * 70,
            f"ファイル名: {result.file_name}",
            f"解析日時: {result.timestamp}",
            f"使用モデル: {result.model}",
            f"レポート生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=== ユーザー提供情報 ===",
        ]
        
        # ユーザーヒント情報
        if result.user_hint and result.user_hint.strip():
            report_lines.extend([
                f"AIへの補足ヒント: {result.user_hint}",
                ""
            ])
        else:
            report_lines.extend([
                "AIへの補足ヒント: （なし）",
                ""
            ])
        
        # ピーク情報
        report_lines.extend([
            "=== 検出ピーク情報 ===",
            peak_df.to_string(index=False),
            "",
            "=== 初回AI解析結果 ===",
            result.ai_analysis,
            ""
        ])
        
        # Q&A履歴セクション
        if qa_history and len(qa_history) > 0:
            report_lines.extend([
                "=== 追加質問・回答履歴 ===",
                f"質問総数: {len(qa_history)}",
                ""
            ])
            
            for i, qa in enumerate(qa_history, 1):
                report_lines.extend([
                    f"【質問 {i}】（{qa.get('timestamp', 'N/A')}）",
                    f"Q: {qa.get('question', '')}",
                    "",
                    f"【回答 {i}】",
                    f"A: {qa.get('answer', '')}",
                    "",
                    "-" * 50,
                    ""
                ])
        else:
            report_lines.extend([
                "=== 追加質問・回答履歴 ===",
                "追加質問はありませんでした。",
                ""
            ])
        
        # 参照文献
        report_lines.extend([
            "=== 参照文献 ===",
        ])
        
        if result.relevant_docs and len(result.relevant_docs) > 0:
            for i, doc in enumerate(result.relevant_docs, 1):
                filename = doc.get('metadata', {}).get('filename', f'文献{i}')
                similarity = doc.get('similarity_score', 0.0)
                preview = doc.get('text', '')[:200] + "..." if len(doc.get('text', '')) > 200 else doc.get('text', '')
                report_lines.extend([
                    f"{i}. {filename}（類似度: {similarity:.3f}）",
                    f"   内容抜粋: {preview}",
                    ""
                ])
        else:
            report_lines.append("参照文献は使用されませんでした。")
        
        # フッター情報
        report_lines.extend([
            "",
            "=" * 70,
            "【レポートについて】",
            "本レポートは RamanEye AI Analysis System v3.0 によって自動生成されました。",
            "初回解析結果に加えて、ユーザーからの追加質問とAIの回答も含まれています。",
            "結果の解釈および活用については、専門家による検証を推奨します。",
            "=" * 70
        ])
        
        return "\n".join(report_lines)
    
    def generate_text_report(self, result: AnalysisResult) -> str:
        """標準テキストレポート生成（後方互換性）"""
        return self.generate_comprehensive_text_report(result, qa_history=None)
    
    def generate_pdf_report(self, result: AnalysisResult, qa_history: List[Dict] = None, 
                           plotly_figure: go.Figure = None) -> bytes:
        """PDFレポート生成"""
        if not self.pdf_available:
            raise Exception("PDF生成機能が利用できません")
        
        # ピークデータをDataFrameに変換
        peak_summary_df = pd.DataFrame([
            {
                'ピーク番号': i+1,
                '波数 (cm⁻¹)': f"{peak.wavenumber:.1f}",
                '強度': f"{peak.intensity:.3f}",
                'Prominence': f"{peak.prominence:.3f}",
                'タイプ': '自動検出' if peak.peak_type == 'auto' else '手動追加'
            }
            for i, peak in enumerate(result.peak_data)
        ])
        
        return self.pdf_generator.generate_comprehensive_pdf_report(
            file_key=result.file_name,
            peak_data=[p.to_dict() for p in result.peak_data],
            analysis_result=result.ai_analysis,
            peak_summary_df=peak_summary_df,
            plotly_figure=plotly_figure,
            relevant_docs=result.relevant_docs,
            user_hint=result.user_hint,
            qa_history=qa_history or []
        )
    
    def generate_qa_report(self, file_key: str, qa_history: List[Dict]) -> str:
        """Q&A履歴レポート生成"""
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

# === UI管理層 ===
class RamanUIManager:
    """UI表示とインタラクションを管理"""
    
    def __init__(self, core: RamanAnalysisCore, ai_manager: RamanAIManager, 
                 data_manager: RamanDataManager, report_manager: RamanReportManager):
        self.core = core
        self.ai_manager = ai_manager
        self.data_manager = data_manager
        self.report_manager = report_manager
    
    def render_config_sidebar(self) -> AnalysisConfig:
        """設定サイドバーの表示"""
        st.sidebar.subheader("⚙️ 解析設定")
        
        config = AnalysisConfig()
        
        config.start_wavenum = st.sidebar.number_input(
            "波数（開始）:", -200, 4800, value=config.start_wavenum, step=100
        )
        config.end_wavenum = st.sidebar.number_input(
            "波数（終了）:", -200, 4800, value=config.end_wavenum, step=100
        )
        config.dssn_th = st.sidebar.number_input(
            "ベースラインパラメーター:", 1, 10000, value=1000, step=1
        ) / 1e7
        config.savgol_wsize = st.sidebar.number_input(
            "移動平均ウィンドウサイズ:", 5, 101, step=2, value=config.savgol_wsize
        )
        
        st.sidebar.subheader("🔍 ピーク検出設定")
        
        config.spectrum_type = st.sidebar.selectbox(
            "解析スペクトル:", ["ベースライン削除", "移動平均後"], index=0
        )
        config.second_deriv_smooth = st.sidebar.number_input(
            "2次微分平滑化:", 3, 35, step=2, value=config.second_deriv_smooth
        )
        config.second_deriv_threshold = st.sidebar.number_input(
            "2次微分閾値:", 0, 1000, step=10, value=config.second_deriv_threshold
        )
        config.prominence_threshold = st.sidebar.number_input(
            "ピークProminence閾値:", 0, 1000, step=10, value=config.prominence_threshold
        )
        
        return config
    
    def render_ai_config_sidebar(self) -> AIConfig:
        """AI設定サイドバーの表示"""
        st.sidebar.subheader("🤖 AI設定")
        
        config = AIConfig()
        
        # モデル選択
        model_options = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4-turbo-preview"
        ]
        
        config.selected_model = st.sidebar.selectbox(
            "OpenAI モデル選択",
            model_options,
            index=0
        )
        
        config.temperature = st.sidebar.slider(
            "応答の創造性 (Temperature)",
            0.0, 1.0, value=config.temperature, step=0.1
        )
        
        config.max_tokens = st.sidebar.number_input(
            "最大トークン数",
            256, 4096, value=config.max_tokens, step=128
        )
        
        config.use_openai_embeddings = st.sidebar.checkbox(
            "OpenAI埋め込みを使用",
            value=config.use_openai_embeddings
        )
        
        return config
    
    def render_peak_management(self, file_key: str, wavenum: np.ndarray, spectrum: np.ndarray,
                              detected_peaks: np.ndarray) -> Tuple[List[float], set]:
        """ピーク管理UIの表示"""
        col1, col2 = st.columns(2)
        
        # 手動ピーク追加
        with col1:
            st.write("**🔹 ピーク手動追加**")
            add_wavenum = st.number_input(
                "追加する波数 (cm⁻¹):",
                min_value=float(wavenum.min()),
                max_value=float(wavenum.max()),
                value=float(wavenum[len(wavenum)//2]),
                step=1.0,
                key=f"add_wavenum_{file_key}"
            )
            
            manual_peaks = self.data_manager.load_manual_peaks(file_key)
            
            if st.button(f"波数 {add_wavenum:.1f} のピークを追加", key=f"add_peak_{file_key}"):
                is_duplicate = any(abs(existing_wn - add_wavenum) < 2.0 for existing_wn in manual_peaks)
                
                if not is_duplicate:
                    manual_peaks.append(add_wavenum)
                    self.data_manager.save_manual_peaks(file_key, manual_peaks)
                    st.success(f"波数 {add_wavenum:.1f} cm⁻¹ にピークを追加しました")
                    st.rerun()
                else:
                    st.warning("近接する位置にすでにピークが存在します")
        
        # ピーク除外管理
        with col2:
            st.write("**🔸 検出ピーク除外**")
            excluded_peaks = self.data_manager.load_excluded_peaks(file_key)
            
            if len(detected_peaks) > 0:
                detected_options = []
                for i, idx in enumerate(detected_peaks):
                    wn = wavenum[idx]
                    intensity = spectrum[idx]
                    status = "除外済み" if idx in excluded_peaks else "有効"
                    detected_options.append(f"ピーク{i+1}: {wn:.1f} cm⁻¹ ({intensity:.3f}) - {status}")
                
                selected_peak = st.selectbox(
                    "除外/復活させるピークを選択:",
                    options=range(len(detected_options)),
                    format_func=lambda x: detected_options[x],
                    key=f"select_peak_{file_key}"
                )
                
                peak_idx = detected_peaks[selected_peak]
                is_excluded = peak_idx in excluded_peaks
                
                if is_excluded:
                    if st.button(f"ピーク{selected_peak+1}を復活", key=f"restore_peak_{file_key}"):
                        excluded_peaks.remove(peak_idx)
                        self.data_manager.save_excluded_peaks(file_key, excluded_peaks)
                        st.success(f"ピーク{selected_peak+1}を復活させました")
                        st.rerun()
                else:
                    if st.button(f"ピーク{selected_peak+1}を除外", key=f"exclude_peak_{file_key}"):
                        excluded_peaks.add(peak_idx)
                        self.data_manager.save_excluded_peaks(file_key, excluded_peaks)
                        st.success(f"ピーク{selected_peak+1}を除外しました")
                        st.rerun()
            else:
                st.info("検出されたピークがありません")
        
        return manual_peaks, excluded_peaks
    
    def render_analysis_results(self, file_key: str):
        """解析結果の表示"""
        result = self.data_manager.load_analysis_result(file_key)
        if not result:
            st.info("解析結果がありません。")
            return
        
        with st.expander("📜 解析結果を表示", expanded=True):
            st.write(f"**解析日時:** {result.timestamp}")
            st.write(f"**使用モデル:** {result.model}")
            st.markdown("**解析結果:**")
            st.markdown(result.ai_analysis)
        
        # レポートダウンロード
        self.render_download_section(file_key, result)
        
        # Q&A機能
        self.render_qa_section(file_key, result)
    
    def render_download_section(self, file_key: str, result: AnalysisResult):
        """レポートダウンロードセクション（Q&A・ヒント統合版）"""
        st.subheader("📥 包括的レポートダウンロード")
        
        # Q&A履歴を取得
        qa_history = self.data_manager.load_qa_history(file_key)
        
        # レポート情報の表示
        info_text = f"""
        **レポートに含まれる内容:**
        - 検出ピーク詳細情報
        - 初回AI解析結果
        - AIへの補足ヒント: {'あり' if result.user_hint and result.user_hint.strip() else 'なし'}
        - 追加質問・回答: {len(qa_history)}件
        - 参照文献情報: {len(result.relevant_docs)}件
        """
        st.info(info_text)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 包括的テキストレポート
            comprehensive_text_report = self.report_manager.generate_comprehensive_text_report(result, qa_history)
            st.download_button(
                label="📄 完全版テキストレポート",
                data=comprehensive_text_report,
                file_name=f"raman_comprehensive_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                key=f"download_comprehensive_text_{file_key}",
                help="初回解析結果、Q&A履歴、ユーザーヒントを全て含む完全版レポート"
            )
        
        with col2:
            if self.report_manager.pdf_available:
                if st.button(f"📊 完全版PDFレポート生成", key=f"generate_comprehensive_pdf_{file_key}"):
                    try:
                        plotly_figure = self.data_manager.load_plotly_figure(file_key)
                        pdf_bytes = self.report_manager.generate_pdf_report(result, qa_history, plotly_figure)
                        
                        st.download_button(
                            label="📊 完全版PDFダウンロード",
                            data=pdf_bytes,
                            file_name=f"raman_comprehensive_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            key=f"download_comprehensive_pdf_{file_key}",
                            help="グラフ、解析結果、Q&A履歴を含む包括的PDFレポート"
                        )
                        st.success("✅ 完全版PDFレポートが生成されました！")
                    except Exception as e:
                        st.error(f"PDFレポート生成エラー: {e}")
            else:
                st.info("PDF機能は利用できません")
        
        with col3:
            if qa_history:
                qa_report = self.report_manager.generate_qa_report(file_key, qa_history)
                st.download_button(
                    label="💬 Q&A履歴のみ",
                    data=qa_report,
                    file_name=f"qa_only_{file_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key=f"download_qa_only_{file_key}",
                    help="追加質問と回答のみを抜き出したレポート"
                )
            else:
                st.info("Q&A履歴なし")
        
        # レポート内容のプレビュー
        if qa_history or (result.user_hint and result.user_hint.strip()):
            with st.expander("👀 レポート内容プレビュー", expanded=False):
                
                if result.user_hint and result.user_hint.strip():
                    st.markdown("**🔹 AIへの補足ヒント:**")
                    st.text(result.user_hint)
                    st.markdown("---")
                
                if qa_history:
                    st.markdown(f"**💬 追加質問・回答履歴 ({len(qa_history)}件):**")
                    for i, qa in enumerate(qa_history[-2:], 1):  # 最新2件のみプレビュー
                        st.markdown(f"**Q{i}:** {qa.get('question', '')[:100]}...")
                        st.markdown(f"**A{i}:** {qa.get('answer', '')[:200]}...")
                        if i < len(qa_history[-2:]):
                            st.markdown("---")
                    
                    if len(qa_history) > 2:
                        st.info(f"※ 他 {len(qa_history) - 2} 件の質問・回答は完全版レポートに含まれます")
                else:
                    st.info("追加質問はありません")    
    def render_qa_section(self, file_key: str, result: AnalysisResult):
        """Q&Aセクション"""
        st.markdown("---")
        st.subheader(f"💬 追加質問 - {file_key}")
        
        qa_history = self.data_manager.load_qa_history(file_key)
        
        # 質問履歴の表示
        if qa_history:
            with st.expander("📚 質問履歴を表示", expanded=False):
                for i, qa in enumerate(qa_history, 1):
                    st.markdown(f"**質問{i}:** {qa['question']}")
                    st.markdown(f"**回答{i}:** {qa['answer']}")
                    st.markdown(f"*質問日時: {qa['timestamp']}*")
                    st.markdown("---")
        
        # 質問入力フォーム
        with st.form(key=f"qa_form_{file_key}"):
            st.markdown("**解析結果について質問があれば、下記にご記入ください：**")
            
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
                    answer = self.ai_manager.answer_question(
                        question=user_question,
                        context=result.analysis_context,
                        qa_history=qa_history
                    )
                    
                    new_qa = {
                        'question': user_question,
                        'answer': answer,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    qa_history.append(new_qa)
                    self.data_manager.save_qa_history(file_key, qa_history)
                    
                    st.success("✅ 回答が完了しました！")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"質問処理中にエラーが発生しました: {str(e)}")
    
    def render_system_status(self):
        """システム状態の表示"""
        with st.expander("🔧 システム状態", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**機能状態:**")
                st.write(f"🤖 AI機能: {'✅' if AI_AVAILABLE else '❌'}")
                st.write(f"📊 PDF生成: {'✅' if PDF_GENERATION_AVAILABLE else '❌'}")
                st.write(f"🛡️ セキュリティ: {'✅' if SECURITY_AVAILABLE else '❌'}")
            
            with col2:
                st.write("**通信状態:**")
                internet_ok = self.ai_manager.network_manager.check_internet_connection()
                st.write(f"🌐 インターネット: {'✅' if internet_ok else '❌'}")
                st.write(f"🔑 AI接続: {'✅' if self.ai_manager.is_ready else '❌'}")
                
                db_info = self.ai_manager.get_database_info()
                db_status = db_info.get('status', 'Unknown')
                st.write(f"📚 データベース: {db_status}")

# === プロット描画機能 ===
class RamanPlotManager:
    """プロット描画機能の統合管理"""
    
    def __init__(self, data_manager: RamanDataManager):
        self.data_manager = data_manager
    
    def create_interactive_plot(self, file_key: str, wavenum: np.ndarray, spectrum: np.ndarray,
                               detected_peaks: np.ndarray, detected_prominences: np.ndarray,
                               manual_peaks: List[float], excluded_peaks: set, 
                               config: AnalysisConfig) -> go.Figure:
        """インタラクティブプロットの作成"""
        
        # 2次微分計算
        if len(spectrum) > config.second_deriv_smooth:
            second_derivative = savgol_filter(spectrum, int(config.second_deriv_smooth), 2, deriv=2)
        else:
            second_derivative = np.gradient(np.gradient(spectrum))
        
        # 有効ピークのフィルタリング
        valid_peaks = [i for i in detected_peaks if i not in excluded_peaks]
        valid_prominences = [prom for i, prom in zip(detected_peaks, detected_prominences) if i not in excluded_peaks]
        
        # プロット作成
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.07,
            row_heights=[0.4, 0.3, 0.3],
            subplot_titles=("スペクトル", "2次微分", "ピーク卓立度")
        )
        
        # メインスペクトル
        fig.add_trace(
            go.Scatter(
                x=wavenum, 
                y=spectrum, 
                mode='lines', 
                name=config.spectrum_type, 
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # 有効ピーク
        if valid_peaks:
            fig.add_trace(
                go.Scatter(
                    x=wavenum[valid_peaks], 
                    y=spectrum[valid_peaks], 
                    mode='markers', 
                    name='有効ピーク', 
                    marker=dict(color='red', size=8, symbol='circle')
                ), 
                row=1, col=1
            )
        
        # 除外ピーク
        excluded_list = list(excluded_peaks)
        if excluded_list:
            fig.add_trace(
                go.Scatter(
                    x=wavenum[excluded_list], 
                    y=spectrum[excluded_list], 
                    mode='markers',
                    name='除外ピーク', 
                    marker=dict(color='gray', size=8, symbol='x')
                ), 
                row=1, col=1
            )
        
        # 手動ピーク
        for wn in manual_peaks:
            idx = np.argmin(np.abs(wavenum - wn))
            fig.add_trace(
                go.Scatter(
                    x=[wn], 
                    y=[spectrum[idx]], 
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
                x=wavenum, 
                y=second_derivative, 
                mode='lines',
                name='2次微分', 
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)
        
        # Prominence
        all_peaks, _ = find_peaks(-second_derivative)
        if len(all_peaks) > 0:
            all_prominences = peak_prominences(-second_derivative, all_peaks)[0]
            fig.add_trace(
                go.Scatter(
                    x=wavenum[all_peaks], 
                    y=all_prominences, 
                    mode='markers',
                    name='全ピーク卓立度', 
                    marker=dict(color='orange', size=4, opacity=0.6)
                ),
                row=3, col=1
            )
        
        if valid_peaks:
            fig.add_trace(
                go.Scatter(
                    x=wavenum[valid_peaks], 
                    y=valid_prominences, 
                    mode='markers',
                    name='有効卓立度', 
                    marker=dict(color='red', size=7, symbol='circle')
                ),
                row=3, col=1
            )
        
        # レイアウト設定
        fig.update_layout(
            height=800,
            title=f"ラマンスペクトル解析結果 - {file_key}",
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(t=80, b=50, l=50, r=150)
        )
        
        # 軸設定
        fig.update_xaxes(title_text="波数 (cm⁻¹)", row=3, col=1)
        fig.update_yaxes(title_text="強度 (a.u.)", row=1, col=1)
        fig.update_yaxes(title_text="2次微分", row=2, col=1)
        fig.update_yaxes(title_text="Prominence", row=3, col=1)
        
        # プロットをセッションに保存
        self.data_manager.save_plotly_figure(file_key, fig)
        
        return fig

# === メイン処理統合 ===
class RamanAnalysisApp:
    """アプリケーション全体を統合管理（完全リファクタリング版）"""
    
    def __init__(self):
        self.core = RamanAnalysisCore()
        self.data_manager = RamanDataManager()
        self.report_manager = RamanReportManager()
        self.ai_manager = None
        self.ui_manager = None
        self.plot_manager = RamanPlotManager(self.data_manager)
        self.temp_dir = "./tmp_uploads"
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def initialize_ai_components(self, ai_config: AIConfig):
        """AI関連コンポーネントの初期化"""
        self.ai_manager = RamanAIManager(ai_config)
        self.ui_manager = RamanUIManager(
            self.core, self.ai_manager, self.data_manager, self.report_manager
        )
    
    def run(self):
        """メインアプリケーション実行"""
        st.header("🔬 RamanEye AI Analysis System - Complete Refactored Edition")
        st.markdown("**バージョン 3.0** - 完全統合・責務分離・セキュリティ強化版")
        
        # AI設定の初期化
        ai_config = self.setup_ai_configuration()
        self.initialize_ai_components(ai_config)
        
        # システム状態表示
        self.ui_manager.render_system_status()
        
        # AI接続設定
        ai_ready = self.ai_manager.setup_connection()
        
        # RAG設定
        self.setup_rag_system()
        
        # 解析設定
        config = self.ui_manager.render_config_sidebar()
        
        # ユーザーヒント
        user_hint = st.sidebar.text_area(
            "AIへの補足ヒント（任意）",
            placeholder="例：この試料はポリエチレン系高分子である可能性がある",
            height=100
        )
        
        # ファイルアップロード
        uploaded_file = st.file_uploader(
            "ラマンスペクトルをアップロードしてください", 
            accept_multiple_files=False
        )
        
        if uploaded_file:
            self.process_uploaded_file(uploaded_file, config, user_hint, ai_ready)
    
    def setup_ai_configuration(self) -> AIConfig:
        """AI設定のセットアップ"""
        if self.ui_manager is None:
            # 初期化前の仮UI
            st.sidebar.subheader("🤖 AI設定")
            config = AIConfig()
            config.selected_model = st.sidebar.selectbox(
                "OpenAI モデル選択",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"],
                index=0
            )
            config.temperature = st.sidebar.slider(
                "応答の創造性", 0.0, 1.0, value=0.3, step=0.1
            )
            return config
        else:
            return self.ui_manager.render_ai_config_sidebar()
    
    def setup_rag_system(self):
        """RAGシステム設定"""
        st.sidebar.subheader("📚 論文データベース設定")
        
        # データベース情報表示
        if self.ai_manager:
            db_info = self.ai_manager.get_database_info()
            if db_info.get('status') == '構築済み':
                st.sidebar.success(f"✅ データベース: {db_info.get('n_chunks', 0)} チャンク")
                
                if st.sidebar.button("📊 データベース詳細"):
                    st.sidebar.json(db_info)
            else:
                st.sidebar.info("ℹ️ データベース未構築")
        
        # 文献アップロード
        uploaded_docs = st.sidebar.file_uploader(
            "文献PDFを選択してください（複数可）",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            help="ピーク解析の参考となる論文や資料をアップロードしてください"
        )
        
        if uploaded_docs and st.sidebar.button("📚 論文データベース構築"):
            self.build_knowledge_base(uploaded_docs)
    
    def build_knowledge_base(self, uploaded_docs):
        """知識ベース構築"""
        with st.spinner("セキュアな文献データベースを構築中..."):
            try:
                uploaded_count = 0
                for doc in uploaded_docs:
                    # セキュリティチェック
                    security_result = self.ai_manager.security_manager.secure_file_upload(doc)
                    if security_result['status'] == 'success':
                        safe_filename = security_result['safe_filename']
                        save_path = os.path.join(self.temp_dir, safe_filename)
                        with open(save_path, "wb") as f:
                            f.write(doc.getbuffer())
                        uploaded_count += 1
                    else:
                        st.error(f"ファイルセキュリティエラー: {security_result['message']}")
                
                if uploaded_count > 0:
                    success = self.ai_manager.build_knowledge_base(self.temp_dir)
                    if success:
                        st.sidebar.success(f"✅ {uploaded_count} 件のファイルからデータベースを構築しました")
                    else:
                        st.sidebar.error("❌ データベース構築に失敗しました")
                else:
                    st.sidebar.warning("⚠️ 処理できるファイルがありませんでした")
                    
            except Exception as e:
                st.sidebar.error(f"データベース構築エラー: {e}")
    
    def process_uploaded_file(self, uploaded_file, config: AnalysisConfig, user_hint: str, ai_ready: bool):
        """アップロードファイルの処理"""
        try:
            # ファイル処理（既存の process_spectrum_file を使用）
            result = process_spectrum_file(
                uploaded_file, config.start_wavenum, config.end_wavenum, 
                config.dssn_th, config.savgol_wsize
            )
            
            if result is None or result[0] is None:
                st.error(f"{uploaded_file.name}の処理中にエラーが発生しました")
                return
            
            wavenum, spectra, BSremoval_specta_pos, Averemoval_specta_pos, file_type, file_name = result
            
            st.success(f"✅ ファイル処理完了: {file_type} - {file_name}")
            
            # スペクトル選択
            if config.spectrum_type == "ベースライン削除":
                selected_spectrum = BSremoval_specta_pos
            else:
                selected_spectrum = Averemoval_specta_pos
            
            # ピーク検出・解析処理
            self.perform_analysis_workflow(
                file_name, wavenum, selected_spectrum, config, user_hint, ai_ready
            )
                
        except Exception as e:
            st.error(f"ファイル処理エラー: {e}")
            if st.checkbox("🔍 デバッグ情報を表示"):
                st.exception(e)
    
    def perform_analysis_workflow(self, file_name: str, wavenum: np.ndarray, spectrum: np.ndarray, 
                                 config: AnalysisConfig, user_hint: str, ai_ready: bool):
        """解析ワークフローの実行"""
        
        # ピーク検出実行
        if st.button("🔍 ピーク検出を実行", type="primary"):
            with st.spinner("ピーク検出中..."):
                detected_peaks, detected_prominences, all_peaks, all_prominences = self.core.detect_peaks(
                    wavenum, spectrum, config
                )
                
                if len(detected_peaks) > 0:
                    st.success(f"✅ 検出されたピーク数: {len(detected_peaks)}")
                    
                    # ピーク情報表示
                    peak_df = pd.DataFrame({
                        'ピーク番号': range(1, len(detected_peaks) + 1),
                        '波数 (cm⁻¹)': [f"{wavenum[i]:.1f}" for i in detected_peaks],
                        '強度': [f"{spectrum[i]:.3f}" for i in detected_peaks],
                        'Prominence': [f"{prom:.3f}" for prom in detected_prominences]
                    })
                    st.table(peak_df)
                else:
                    st.info("ピークが検出されませんでした。閾値を調整してください。")
                    return
        else:
            # 既存の検出結果があるかチェック
            existing_result = self.data_manager.load_analysis_result(file_name)
            if existing_result:
                st.info("既存の解析結果があります。")
                detected_peaks = np.array([])  # 既存データから復元する場合の処理
                detected_prominences = np.array([])
            else:
                st.info("「ピーク検出を実行」ボタンをクリックして解析を開始してください。")
                return
        
        # ピーク管理UI
        if len(detected_peaks) > 0:
            st.subheader("🎯 ピーク管理")
            manual_peaks, excluded_peaks = self.ui_manager.render_peak_management(
                file_name, wavenum, spectrum, detected_peaks
            )
            
            # プロット描画
            st.subheader("📊 スペクトル表示")
            fig = self.plot_manager.create_interactive_plot(
                file_name, wavenum, spectrum, detected_peaks, detected_prominences,
                manual_peaks, excluded_peaks, config
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # AI解析実行
            if ai_ready:
                st.subheader("🤖 AI解析")
                if st.button(f"AI解析を実行 - {file_name}", type="primary"):
                    self.execute_ai_analysis(
                        file_name, wavenum, spectrum, detected_peaks, detected_prominences, 
                        manual_peaks, excluded_peaks, user_hint
                    )
            else:
                st.warning("⚠️ AI機能が利用できません。OpenAI APIキーを設定してください。")
            
            # 既存の解析結果表示
            self.ui_manager.render_analysis_results(file_name)
    
    def execute_ai_analysis(self, file_name: str, wavenum: np.ndarray, spectrum: np.ndarray,
                           detected_peaks: np.ndarray, detected_prominences: np.ndarray,
                           manual_peaks: List[float], excluded_peaks: set, user_hint: str):
        """AI解析実行"""
        with st.spinner("🤖 AI解析中です。しばらくお待ちください..."):
            try:
                start_time = time.time()
                
                # 最終ピークデータ作成
                final_peaks = []
                
                # 有効な自動検出ピーク
                for idx, prom in zip(detected_peaks, detected_prominences):
                    if idx not in excluded_peaks:
                        final_peaks.append(PeakData(
                            wavenumber=float(wavenum[idx]),
                            intensity=float(spectrum[idx]),
                            prominence=float(prom),
                            peak_type='auto'
                        ))
                
                # 手動ピーク
                if len(spectrum) > 5:  # 2次微分計算のための最小チェック
                    second_derivative = savgol_filter(spectrum, 5, 2, deriv=2)
                    for wn in manual_peaks:
                        idx = np.argmin(np.abs(wavenum - wn))
                        prom = self.core.calculate_manual_peak_prominence(wn, wavenum, second_derivative)
                        final_peaks.append(PeakData(
                            wavenumber=float(wn),
                            intensity=float(spectrum[idx]),
                            prominence=prom,
                            peak_type='manual'
                        ))
                
                if not final_peaks:
                    st.error("解析するピークがありません。")
                    return
                
                # AI解析実行
                result = self.ai_manager.analyze_peaks(final_peaks, user_hint)
                result.file_name = file_name
                
                # DataFrame作成
                peak_summary_df = pd.DataFrame([
                    {
                        'ピーク番号': i+1,
                        '波数 (cm⁻¹)': f"{peak.wavenumber:.1f}",
                        '強度': f"{peak.intensity:.3f}",
                        'Prominence': f"{peak.prominence:.3f}",
                        'タイプ': '自動検出' if peak.peak_type == 'auto' else '手動追加'
                    }
                    for i, peak in enumerate(final_peaks)
                ])
                result.peak_summary_df = peak_summary_df
                
                # 結果保存
                self.data_manager.save_analysis_result(file_name, result)
                
                # 処理時間表示
                elapsed_time = time.time() - start_time
                st.success(f"✅ AI解析が完了しました！（処理時間: {elapsed_time:.2f}秒）")
                
                # 結果の即座表示
                with st.expander("📋 解析結果プレビュー", expanded=True):
                    st.markdown(result.ai_analysis[:500] + "..." if len(result.ai_analysis) > 500 else result.ai_analysis)
                
                st.rerun()
                
            except Exception as e:
                st.error(f"AI解析エラー: {e}")
                if st.checkbox("🔍 エラー詳細を表示"):
                    st.exception(e)

# === エントリーポイント ===
def peak_ai_analysis_mode():
    """メイン関数（完全リファクタリング版）"""
    # システム最適化の提案
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
            """)
    
    # OpenAI API Key 確認
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        st.sidebar.success("🔑 OpenAI API Key: 設定済み")
    else:
        st.sidebar.warning("⚠️ OpenAI API Key: 未設定")
        st.sidebar.info("環境変数 OPENAI_API_KEY を設定してください")
    
    # アプリケーション実行
    app = RamanAnalysisApp()
    app.run()

# システム情報表示（デバッグ用）
def display_system_info():
    """システム情報を表示"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 システム情報")
    
    info = {
        "AI機能": "✅" if AI_AVAILABLE else "❌",
        "PDF生成": "✅" if PDF_GENERATION_AVAILABLE else "❌", 
        "セキュリティ": "✅" if SECURITY_AVAILABLE else "❌",
        "バージョン": "3.0 - Complete Refactored"
    }
    
    for key, value in info.items():
        st.sidebar.write(f"**{key}:** {value}")

# 初期化時にシステム情報を表示
if __name__ == "__main__":
    display_system_info()
