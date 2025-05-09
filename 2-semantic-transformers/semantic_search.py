from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import torch
import time
import util
import random
    
# 使用FAISS進行向量相似度搜尋
def search_with_faiss(model: SentenceTransformer, documents, queries):
    # 將文件轉換為向量嵌入
    fs_documents_embedding = model.encode(documents, batch_size=256, convert_to_tensor=True)
    fs_documents_embedding_np = fs_documents_embedding.cpu().numpy()

    dimension = fs_documents_embedding.shape[1]
    # 創建FAISS索引，使用L2距離計算相似度
    index = faiss.IndexFlatL2(dimension)

    # 如果有GPU可用，將索引移至GPU以加速搜尋
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        index = gpu_index

    # 將文件向量加入索引
    index.add(fs_documents_embedding_np)
    
    print(f'Dimension: {dimension}\n')
    print(fs_documents_embedding)

    # 將查詢轉換為向量嵌入
    fs_queries_embedding = model.encode(queries, batch_size=256, convert_to_tensor=True)
    fs_queries_embedding_np = fs_queries_embedding.cpu().numpy()

    search_start_time = time.time()

    # 設定返回前k個最相似的結果
    top_k = 2
    # 執行批次搜尋
    D, I = util.batch_search(fs_queries_embedding_np, index, top_k, batch_size=8)
    search_end_time = time.time()

    # 顯示搜尋結果
    for i, (dists, idxs) in enumerate(zip(D, I), 1):
        print(f'\nQuery{i}: {queries[i-1]}')
        for j, (dist, idx) in enumerate(zip(dists, idxs), 1):
            print(f'   top_{j}')
            print(f'\tDistance: {dist:.4f}')  # 顯示與查詢的距離（越小表示越相似）
            print(f'\tAnswer: {documents[idx]}')

        # 只顯示前5個查詢結果
        if i == 5:
            break

    search_time = search_end_time - search_start_time
    return search_time


def main():
    # 檢查是否有GPU可用
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    print(f'\nLoading model...')
    # 載入預訓練的sentence-transformer模型
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model = SentenceTransformer(model_name, device=device)

    print(f'\nLoad model done')

    print(f'\nLoading dataset')
    # 載入MS MARCO數據集
    dataset_name = 'ms_marco'
    dataset = load_dataset(dataset_name, 'v2.1', split='train').select(range(100000))
    print(f'\nLoad dataset done')

    # 準備文檔和查詢數據
    documents = [
        passage_text
        for passages in dataset['passages']
        for passage_text in passages['passage_text']
    ]

    queries = dataset['query']
    random.shuffle(queries)  # 隨機打亂查詢順序

    # 設定要處理的查詢數量
    NUM_QUERIES=10000
    queries = queries[:NUM_QUERIES]

    # 執行搜尋並計時
    search_time = search_with_faiss(model, documents, queries)
    print(f'\nSearch time: {search_time}')
    

if __name__ == '__main__':
    main()