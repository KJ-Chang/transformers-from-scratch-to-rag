# Transformers from Scratch to RAG

本專案展示了從基礎 Transformer 模型到進階的 RAG (Retrieval-Augmented Generation) 系統的完整實現過程，包含三個主要部分：

## 1. Text Classification (1-text-classification)

使用 DistilBERT 模型進行文本分類任務：
- 使用 AG News 數據集進行微調
- 實現文本多分類（World、Sports、Business、Sci/Tech）
- 包含完整的訓練和推理流程
- 支援 GPU 加速和混合精度訓練
- 提供模型評估指標（準確率、F1分數等）

## 2. Semantic Search (2-semantic-transformers)

實現高效能的語意搜尋系統：
- 使用 Sentence Transformers 進行文本編碼
- 整合 FAISS 進行向量索引和搜尋
- 支援大規模文本數據處理
- 實現批次處理以優化記憶體使用
- 提供搜尋效能分析和評估

## 3. RAG System (3-rag)

建立完整的 RAG（檢索增強生成）系統：
- 支援多種生成模型（T5 和 GPT/Falcon）
- 提供 Web 介面進行互動式問答
- 實現實時文檔檢索和答案生成
- 支援對話歷史記錄和管理
- 模型動態載入以優化資源使用
- 使用 FastAPI 提供 RESTful API 服務

### 主要特點

- 完整的程式碼實現和文檔說明
- 模組化設計，易於擴展和修改
- 支援 GPU 加速
- 提供配置文件進行參數管理
- 實現記憶體優化和效能監控

### 技術堆疊

- PyTorch
- Transformers (Hugging Face)
- Sentence Transformers
- FAISS
- FastAPI
- WebSocket
