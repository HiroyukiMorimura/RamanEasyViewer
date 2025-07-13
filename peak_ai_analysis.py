# -*- coding: utf-8 -*-
"""
ピークAI解析モジュール
RAG機能とOpenAI APIを使用したラマンスペクトルの高度な解析
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import json
import pickle
import requests
from datetime import datetime
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import savgol_filter, find_peaks, peak_prominences
from pathlib import Path
from common_utils import *
from peak_analysis import optimize_thresholds_via_gridsearch

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
# openai_api_key = "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"
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY", openai_api_key)
print(openai_api_key)
print(openai.api_key)

def check_internet_connection():
    """インターネット接続をチェックする"""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

class LLMConnector:
    """OpenAI LLM接続設定クラス"""
    def __init__(self):
        self.is_online = check_internet_connection()
        self.selected_model = "gpt-3.5-turbo"
        self.openai_client = None
        
    def setup_llm_connection(self):
        """OpenAI API接続を設定する"""
        # if not self.is_online:
        #     st.sidebar.error("❌ インターネット接続が必要です")
        #     return False
        
        # st.sidebar.success("🌐 インターネット接続: 正常")
        
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
        
        # if api_key_input:
        try:
            # openai.api_key = api_key_input
            openai.api_key = os.getenv("OPENAI_API_KEY", openai_api_key)
            self.selected_model = selected_model
            self.openai_client = "openai"
            st.sidebar.success(f"✅ OpenAI API接続設定完了 ({selected_model})")
            return True
        except Exception as e:
            st.sidebar.error(f"API設定エラー: {e}")
            return False
        # else:
        #     st.sidebar.warning("⚠️ OpenAI API キーを入力してください")
        #     st.sidebar.info("""
        #     **API キーの取得方法:**
        #     1. https://platform.openai.com にアクセス
        #     2. アカウントを作成してログイン
        #     3. API Keys セクションでキーを生成
        #     4. 環境変数 OPENAI_API_KEY に設定することを推奨
        #     """)
        #     return False
    
    def generate_analysis(self, prompt, temperature=0.3, max_tokens=1024, stream_display=True):
        """OpenAI APIで解析を実行する"""
        if not self.selected_model:
            raise Exception("OpenAI モデルが設定されていません")
        
        system_message = "あなたはラマンスペクトロスコピーの専門家です。ピーク位置と論文、またはインターネット上の情報を比較して、このサンプルが何の試料なのか当ててください。すべて日本語で答えてください。"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.selected_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt + "\n\nすべて日本語で詳しく説明してください。"}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            full_response = ""
            if stream_display:
                stream_area = st.empty()
            
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        full_response += delta["content"]
                        if stream_display:
                            stream_area.markdown(full_response)
            
            return full_response
                
        except Exception as e:
            raise Exception(f"OpenAI API解析エラー: {str(e)}")
    
    def generate_qa_response(self, question, context, previous_qa_history=None):
        """質問応答専用のOpenAI API呼び出し"""
        if not self.selected_model:
            raise Exception("OpenAI モデルが設定されていません")
        
        system_message = """あなたはラマンスペクトロスコピーの専門家です。
解析結果や過去の質問履歴を踏まえて、ユーザーの質問に日本語で詳しく答えてください。
科学的根拠に基づいた正確な回答を心がけてください。"""
        
        # コンテキストの構築
        context_text = f"【解析結果】\n{context}\n\n"
        
        if previous_qa_history:
            context_text += "【過去の質問履歴】\n"
            for i, qa in enumerate(previous_qa_history, 1):
                context_text += f"質問{i}: {qa['question']}\n回答{i}: {qa['answer']}\n\n"
        
        context_text += f"【新しい質問】\n{question}"
        
        try:
            response = openai.ChatCompletion.create(
                model=self.selected_model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": context_text}
                ],
                temperature=0.3,
                max_tokens=1024,
                stream=True
            )
            
            full_response = ""
            stream_area = st.empty()
            
            for chunk in response:
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0]["delta"]
                    if "content" in delta:
                        full_response += delta["content"]
                        stream_area.markdown(full_response)
            
            return full_response
                
        except Exception as e:
            raise Exception(f"質問応答エラー: {str(e)}")

class RamanRAGSystem:
    """RAG機能のクラス"""
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2', use_openai_embeddings=True, openai_embedding_model="text-embedding-ada-002"):
        self.use_openai = use_openai_embeddings and check_internet_connection()
        self.openai_embedding_model = openai_embedding_model
        
        if PDF_AVAILABLE:
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
        
        self.vector_db = None
        self.documents: List[str] = []
        self.document_metadata: List[Dict] = []
        self.embedding_dim: int = 0
        self.db_info: Dict = {}
    
    def build_vector_database(self, folder_path: str):
        """フォルダ内のPDF/DOCX/TXTを読み込んでベクトルデータベースを構築"""
        if not PDF_AVAILABLE:
            st.error("PDF処理ライブラリが利用できません")
            return
            
        if not os.path.exists(folder_path):
            st.error(f"指定されたフォルダが存在しません: {folder_path}")
            return

        # ファイル一覧取得
        file_patterns = ['*.pdf', '*.docx', '*.txt']
        files = []
        for pat in file_patterns:
            files.extend(glob.glob(os.path.join(folder_path, pat)))
        if not files:
            st.warning("指定フォルダに対応ファイルが見つかりません。")
            return

        # テキスト抽出とチャンク化
        all_chunks, all_metadata = [], []
        st.info(f"{len(files)} 件のファイルを処理中…")
        pbar = st.progress(0)
        for idx, fp in enumerate(files):
            text = self._extract_text(fp)
            chunks = self.chunk_text(text)
            for c in chunks:
                all_chunks.append(c)
                all_metadata.append({
                    'filename': os.path.basename(fp),
                    'filepath': fp,
                    'preview': c[:100] + "…" if len(c) > 100 else c
                })
            pbar.progress((idx + 1) / len(files))

        if not all_chunks:
            st.error("抽出できるテキストチャンクがありませんでした。")
            return

        # 埋め込みベクトルの生成
        st.info("埋め込みベクトルを生成中…")
        if self.use_openai:
            embeddings = self._create_openai_embeddings(all_chunks)
            embeddings = np.array(embeddings, dtype=np.float32)
        else:
            embeddings = self.embedding_model.encode(all_chunks, show_progress_bar=True)
            embeddings = np.array(embeddings, dtype=np.float32)

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
                self.openai_embedding_model if self.use_openai 
                else self.embedding_model.__class__.__name__
            )
        }
        st.success(f"ベクトルDB構築完了: {len(all_chunks)} チャンク")
    
    def _create_openai_embeddings(self, texts: List[str], batch_size: int = 200) -> np.ndarray:
        """OpenAI埋め込みAPIを使用"""
        all_embs = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            resp = openai.Embedding.create(
                model=self.openai_embedding_model,
                input=chunk
            )
            embs = [d['embedding'] for d in resp['data']]
            all_embs.extend(embs)
        return np.array(all_embs, dtype=np.float32)
    
    def _extract_text(self, file_path: str) -> str:
        """ファイルからテキストを抽出"""
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.pdf':
                reader = PyPDF2.PdfReader(file_path)
                return "\n".join(p.extract_text() or "" for p in reader.pages)
            if ext == '.docx':
                doc = docx.Document(file_path)
                return "\n".join(p.text for p in doc.paragraphs)
            if ext == '.txt':
                return open(file_path, encoding='utf-8', errors='ignore').read()
        except Exception as e:
            st.error(f"{file_path} の読み込みエラー: {e}")
        return ""
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """テキストをチャンクに分割"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
                
        return chunks
    
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
    
    def search_relevant_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """クエリに関連する文書を検索"""
        if self.vector_db is None:
            return []
    
        # DB作成時のモデル情報を確認
        model_used = self.db_info.get("embedding_model", "")
        if model_used == "text-embedding-ada-002":
            query_emb = self._create_openai_embeddings([query])
        else:
            query_emb = self.embedding_model.encode([query], show_progress_bar=False)
    
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
    """Peak AI analysis mode (RAG + OpenAI analysis)"""
    if not PDF_AVAILABLE:
        st.error("AI解析機能を使用するには、以下のライブラリが必要です：")
        st.code("pip install PyPDF2 python-docx openai faiss-cpu sentence-transformers")
        return
    
    st.header("ラマンピークAI解析")
    
    # LLM接続設定
    llm_connector = LLMConnector()
    
    # インターネット接続状態の表示
    if llm_connector.is_online:
        st.sidebar.success("🌐 インターネット接続: 正常")
    else:
        st.sidebar.error("❌ インターネット接続: 必要")
        st.error("この機能にはインターネット接続が必要です。")
        return
    
    # OpenAI API設定
    llm_ready = llm_connector.setup_llm_connection()
    
    # RAG設定セクション
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
    
    # RAGシステムの初期化
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
        "🧪 AIへの補足ヒント（任意）",
        placeholder="例：この試料はポリエチレン系高分子である可能性がある、など"
    )
    
    # ピーク解析部分の実行
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
            with st.spinner("論文をアップロードし、データベースを構築中..."):
                for uploaded_file in uploaded_files:
                    save_path = os.path.join(TEMP_DIR, uploaded_file.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                st.session_state.rag_system.build_vector_database(TEMP_DIR)
                st.session_state.rag_db_built = True
                st.sidebar.success(f"✅ {len(uploaded_files)} 件のPDFからデータベースを構築しました。")
    
    # データベース保存セクション
    if st.session_state.rag_db_built:
        setup_database_save()

def setup_database_save():
    """データベース保存セクション"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 データベース保存")

    save_method = st.sidebar.radio("保存方法", ["パス入力", "プリセット選択"], index=0)

    if save_method == "パス入力":
        save_folder = st.sidebar.text_input(
            "保存先フォルダパス",
            value="./saved_databases",
            help="データベースを保存するフォルダパスを入力してください"
        )
    else:
        preset_options = {
            "デスクトップ": str(Path.home() / "Desktop" / "RamanDB"),
            "ドキュメント": str(Path.home() / "Documents" / "RamanDB"),
            "カレントフォルダ": "./saved_databases",
            "プロジェクトフォルダ": "./project_databases"
        }
        selected_preset = st.sidebar.selectbox("プリセットフォルダ", list(preset_options.keys()))
        save_folder = preset_options[selected_preset]
        st.sidebar.info(f"保存先: {save_folder}")

    db_name = st.sidebar.text_input(
        "データベース名",
        value=f"raman_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="保存するデータベースの名前を入力してください（拡張子不要）"
    )

    with st.sidebar.expander("📊 保存予定のデータベース情報"):
        db_info = st.session_state.rag_system.get_database_info()
        st.write(f"文書数: {db_info.get('num_documents', 0)}")
        st.write(f"チャンク数: {db_info.get('num_chunks', 0)}")
        st.write(f"作成日時: {db_info.get('created_at', 'N/A')}")

    if st.sidebar.button("💾 データベースを保存"):
        if save_folder and db_name:
            with st.spinner("データベースを保存中..."):
                success = st.session_state.rag_system.save_database(save_folder, db_name)
                if success:
                    st.sidebar.success("データベースが正常に保存されました！")
                    st.sidebar.info(f"保存場所: {save_folder}")
        else:
            st.sidebar.error("保存先フォルダパスとデータベース名を入力してください。")

def load_existing_database():
    """既存データベースの読み込み"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 既存データベース読み込み")

    uploaded_pkl = st.sidebar.file_uploader(
        "1) ドキュメント & メタデータ (.pkl) を選択",
        type="pkl",
        key="load_pkl"
    )
    if not uploaded_pkl:
        st.sidebar.info("まず①で .pkl ファイルを選択してください")
        return

    # 一時フォルダに保存
    TEMP_LOAD_DIR = "./tmp_load"
    os.makedirs(TEMP_LOAD_DIR, exist_ok=True)
    pkl_path = os.path.join(TEMP_LOAD_DIR, uploaded_pkl.name)
    with open(pkl_path, "wb") as f:
        f.write(uploaded_pkl.getbuffer())

    # ベース名を抽出
    base_name = Path(uploaded_pkl.name).stem
    if base_name.endswith("_documents"):
        base_name = base_name[:-len("_documents")]

    save_folder = st.sidebar.text_input(
        "保存先フォルダパス",
        value="./saved_databases",
        help="例: ./saved_databases"
    )
    st.sidebar.info(f"→ {save_folder}/{base_name}_faiss.index, _info.json を読み込みます")

    # 自動組み立てして存在チェック
    index_path = os.path.join(save_folder, f"{base_name}_faiss.index")
    info_path = os.path.join(save_folder, f"{base_name}_info.json")

    if not os.path.exists(index_path) or not os.path.exists(info_path):
        st.sidebar.error("対応する .index または _info.json が見つかりません")
        return

    # データベース読み込み
    rag = st.session_state.rag_system
    try:
        rag.vector_db = faiss.read_index(index_path)
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        rag.documents = data["documents"]
        rag.document_metadata = data["document_metadata"]
        rag.embedding_dim = data["embedding_dim"]
        
        with open(info_path, "r", encoding="utf-8") as f:
            rag.db_info = json.load(f)
        
        st.sidebar.success("✅ データベースを正常に読み込みました！")
        st.session_state.rag_db_built = True
        
    except Exception as e:
        st.sidebar.error(f"データベース読み込み失敗: {e}")

def perform_peak_analysis_with_ai(llm_connector, user_hint, llm_ready):
    """AI機能を含むピーク解析の実行"""
    # 事前パラメータ
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
    uploaded_files = st.file_uploader("ファイルを選択してください", accept_multiple_files=True, key="file_uploader")
    
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

    # プロット描画（詳細は省略 - 元のコードと同じ）
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
    
    st.plotly_chart(fig, use_container_width=True)

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
