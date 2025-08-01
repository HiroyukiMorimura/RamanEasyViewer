# RamanEye Easy Viewer v2.0.0 - Secure Enterprise Edition

<div align="center">

🔬 **Advanced Raman Spectroscopy Analysis Platform with Enterprise Security**  
統合ラマンスペクトル解析ツール - セキュア企業版

</div>

## 📋 概要

RamanEye Easy Viewerは、ラマン分光法による高度なスペクトル解析を提供する統合プラットフォームです。研究機関・企業向けのセキュリティ機能を備え、データの完全性と監査証跡を保証します。

## 🌟 主要特徴

- 🔬 **包括的スペクトル解析**: 基本解析からAI支援まで
- 🔐 **エンタープライズセキュリティ**: 電子署名・暗号化・監査機能
- 🤖 **AI統合**: OpenAI API + RAG機能による高度解析
- 👥 **ロールベースアクセス制御**: Admin/Analyst/Viewer権限管理
- 📊 **多変量解析**: PCA・NMF・クラスター分析
- 🎯 **ピーク分離**: 高精度ローレンツフィッティング
- 📈 **検量線作成**: 定量分析用検量線の自動生成
- 🗄️ **データベース比較**: スペクトルライブラリとの照合

## 🎯 主要機能

### 📊 解析機能

| 機能 | 説明 | 権限レベル |
|------|------|-----------|
| スペクトル解析 | 基本的なラマンスペクトル表示・前処理 | All Users |
| ピーク分析 | 自動ピーク検出・手動調整・グリッドサーチ最適化 | Analyst+ |
| ピーク分離 | ローレンツフィッティング・制約付き最適化 | Analyst+ |
| 多変量解析 | NMF・PCA・クラスター分析 | Analyst+ |
| 検量線作成 | ピーク面積・PLS回帰による定量分析 | Analyst+ |
| AI解析 | OpenAI API + RAG機能による高度解釈 | Analyst+ |
| データベース比較 | コサイン類似度・相関分析による照合 | All Users |

### 🔐 セキュリティ機能

- **電子署名システム**: RSA-2048暗号化・デジタル証明書
- **データ暗号化**: AES-256-GCM・ファイル完全性保証
- **アクセス制御**: ロールベース権限管理・監査ログ
- **改ざん防止**: HMAC-SHA256・ブロックチェーンハッシュ
- **コンプライアンス対応**: GDPR・HIPAA・FDA 21 CFR Part 11

### 👥 ユーザー管理

- **Admin**: 全機能アクセス・ユーザー管理・システム設定
- **Analyst**: 分析機能フルアクセス・データエクスポート
- **Viewer**: 基本表示・データベース比較のみ

## 🚀 クイックスタート

### システム要件

- **Python**: 3.8+
- **OS**: Windows/macOS/Linux
- **RAM**: 8GB以上推奨
- **ストレージ**: 5GB以上の空き容量

### インストール

```bash
# 1. リポジトリのクローン
git clone https://github.com/your-repo/ramaneye-easy-viewer.git
cd ramaneye-easy-viewer

# 2. 仮想環境の作成
python -m venv ramaneye_env

# 3. 仮想環境の有効化
# Windows:
ramaneye_env\Scripts\activate
# macOS/Linux:
source ramaneye_env/bin/activate

# 4. 依存関係のインストール
pip install -r requirements.txt

# 5. アプリケーションの起動
streamlit run main.py
```

### デモアカウント

| ロール | ユーザー名 | パスワード | 権限 |
|--------|-----------|-----------|------|
| 👑 Admin | admin | Admin123! | 全機能アクセス |
| 🔬 Analyst | analyst | Analyst123! | 分析機能フルアクセス |
| 👁️ Viewer | viewer | Viewer123! | 基本機能のみ |

## 📁 プロジェクト構成

```
ramaneye-easy-viewer/
├── main.py                          # メインアプリケーション
├── requirements.txt                 # 依存関係
├── .env.example                     # 環境設定テンプレート
├── README.md                        # このファイル
│
├── 🔐 認証・セキュリティ
│   ├── auth_system.py              # 認証・認可システム
│   ├── security_manager.py         # セキュリティ管理
│   ├── electronic_signature.py     # 電子署名システム
│   └── user_management_ui.py       # ユーザー管理UI
│
├── 📊 解析モジュール
│   ├── spectrum_analysis.py        # スペクトル解析
│   ├── peak_analysis_web.py        # ピーク解析
│   ├── peak_deconvolution.py       # ピーク分離
│   ├── multivariate_analysis.py    # 多変量解析
│   ├── calibration_mode.py         # 検量線作成
│   ├── peak_ai_analysis.py         # AI解析
│   └── raman_database.py           # データベース比較
│
├── 🔧 ユーティリティ
│   ├── common_utils.py              # 共通関数
│   └── config.py                    # 設定管理
│
├── 📋 統合例・UI
│   ├── signature_integration_example.py  # 署名統合例
│   └── signature_management_ui.py        # 署名管理UI
│
└── 📁 データ・ログ（実行時作成）
    ├── secure/                      # セキュリティデータ
    ├── tmp_uploads/                 # 一時ファイル
    └── raman_spectra/               # スペクトルDB
```

## 🔧 使用方法

### 1. 基本的な解析フロー

1. **ログイン**: デモアカウントまたは作成したアカウントでログイン
2. **ファイルアップロード**: CSV/TXTファイルをアップロード
3. **解析実行**: 目的に応じた解析モードを選択
4. **結果確認**: インタラクティブなグラフで結果を確認
5. **データエクスポート**: CSVファイルでダウンロード

### 2. サポートファイル形式

| ファイル形式 | 対応装置 | 説明 |
|-------------|----------|------|
| CSV | 汎用 | カンマ区切り形式 |
| TXT | 汎用 | タブ区切り形式 |
| RamanEye | RamanEye装置 | 専用フォーマット |
| Wasatch | Wasatch Photonics | 自動skipping対応 |
| Eagle | Applied Spectra | RT100-F対応 |

### 3. AI解析機能の設定

```bash
# OpenAI APIキーの設定
export OPENAI_API_KEY="your-api-key-here"

# または .env ファイルに追加
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

## 🔐 セキュリティ機能

### 電子署名システム

```python
# 電子署名が必要な操作の例
@require_signature(
    operation_type="重要レポート確定",
    signature_level=SignatureLevel.DUAL  # 二段階署名
)
def finalize_critical_report():
    # 重要な処理
    pass
```

### データ暗号化

- **ファイル暗号化**: AES-256-GCM
- **完全性チェック**: HMAC-SHA256
- **デジタル署名**: RSA-2048
- **ブロックチェーンハッシュ**: 改ざん防止

### 監査機能

- **ログイン試行**: 成功/失敗の記録
- **ファイルアクセス**: 読み込み/書き込み履歴
- **操作履歴**: 全ての重要操作を記録
- **完全性検証**: 定期的なデータ検証

## 🤖 AI解析機能

### RAG（Retrieval-Augmented Generation）

- **論文アップロード**: PDF/DOCXファイルをアップロード
- **ベクトルDB構築**: 自動的に検索用データベースを作成
- **ピーク解析**: スペクトルから自動ピーク検出
- **AI考察生成**: 学術論文を参照した高度な解釈

### サポートAIモデル

- **OpenAI GPT-4**: 最高品質の解析
- **OpenAI GPT-3.5-turbo**: コスト効率重視
- **ローカルモデル**: オフライン対応（オプション）

## 📊 解析機能詳細

### ピーク分離（Peak Deconvolution）

- **ローレンツフィッティング**: 高精度ピーク分離
- **制約付き最適化**: 波数固定・自動最適化
- **複数試行**: ロバストな結果取得
- **統計的基準**: AIC/BICによる最適ピーク数決定

### 多変量解析

- **NMF**: 非負値行列因子分解
- **PCA**: 主成分分析
- **クラスター分析**: 階層・K-means
- **相関解析**: ピアソン・スピアマン

### データベース比較

- **類似度指標**: コサイン類似度・ピアソン相関・相互相関
- **高速化機能**: プーリング・上位N選択
- **自動照合**: 最高一致スペクトルの自動検出

## 🔄 開発・カスタマイズ

### 新機能の追加

1. **解析モジュール**: `your_analysis.py`を作成
2. **main.py**: メニューに追加
3. **権限設定**: 必要に応じて権限チェック
4. **電子署名**: 重要操作には署名機能を統合

### セキュリティ設定のカスタマイズ

```python
# config.py での設定例
class SecurityConfig:
    PASSWORD_MIN_LENGTH = 12  # パスワード最小長
    MAX_LOGIN_ATTEMPTS = 3    # 最大ログイン試行回数
    SESSION_TIMEOUT = 7200    # セッションタイムアウト（秒）
```

## 🐛 トラブルシューティング

### よくある問題

**Problem: `ModuleNotFoundError: No module named 'streamlit'`**
```bash
pip install -r requirements.txt
```

**Problem: `OpenAI API key not found`**
```bash
# APIキーを環境変数に設定
export OPENAI_API_KEY="your-api-key-here"
```

### ログの確認

```bash
# セキュリティ監査ログ
tail -f ./secure/security_audit.log

# デバッグモード起動
streamlit run main.py --logger.level debug
```

## 🤝 貢献・開発参加

### 開発環境のセットアップ

```bash
# 開発用依存関係のインストール
pip install -r requirements-dev.txt

# pre-commit フックの設定
pre-commit install

# テストの実行
pytest tests/
```

### コード貢献の手順

1. **Fork** このリポジトリ
2. **Feature branch** を作成: `git checkout -b feature/amazing-feature`
3. **Commit** 変更: `git commit -m 'Add amazing feature'`
4. **Push** ブランチ: `git push origin feature/amazing-feature`
5. **Pull Request** を作成

### コーディング規約

- **PEP 8**: Python標準コーディングスタイル
- **Type Hints**: 関数にはタイプヒントを追加
- **Docstrings**: 全ての関数・クラスにドキュメント
- **Security First**: セキュリティを最優先に考慮

## 📜 ライセンス

このプロジェクトは MIT License の下で公開されています。詳細は LICENSE ファイルをご覧ください。

## 🔄 更新履歴

### v2.0.0 (2025-08-01) - Secure Enterprise Edition

- 🔐 **エンタープライズセキュリティ**: 電子署名・暗号化・監査機能
- 🤖 **AI統合**: OpenAI API + RAG機能
- 👥 **ロールベース認証**: Admin/Analyst/Viewer権限管理
- 📊 **高度多変量解析**: NMF・相関解析の拡充
- 🎯 **制約付きピーク分離**: 波数固定フィッティング
- 🗄️ **スペクトルDB比較**: 類似度マトリックス計算
- 🔧 **UI/UX改善**: インタラクティブ機能の強化

### v1.5.0 (2025-06-15)

- 📈 **検量線作成機能**: PLS回帰・ピーク面積
- 🔍 **グリッドサーチ最適化**: パラメータ自動調整
- 📁 **複数ファイル対応**: バッチ処理機能

### v1.0.0 (2025-01-10)

- 🎉 **初回リリース**: 基本的なスペクトル解析機能

## 📞 サポート・お問い合わせ

### 技術サポート

- **GitHub Issues**: [Issues Page](https://github.com/your-repo/ramaneye-easy-viewer/issues)
- **メール**: support@your-company.com
- **ドキュメント**: [Wiki](https://github.com/your-repo/ramaneye-easy-viewer/wiki)

### 商用利用・ライセンス

- **企業向けサポート**: enterprise@your-company.com
- **カスタマイズ開発**: dev@your-company.com

## 🙏 謝辞

このプロジェクトは以下のオープンソースプロジェクトに基づいています：

- **Streamlit**: WebUIフレームワーク
- **NumPy/SciPy**: 科学計算ライブラリ
- **scikit-learn**: 機械学習ライブラリ
- **OpenAI**: AI機能統合
- **Plotly**: インタラクティブ可視化

---

<div align="center">

🔬 **Advanced Raman Spectroscopy Analysis Platform** 🔬  
Made with ❤️ by [Your Name/Organization]

</div>
