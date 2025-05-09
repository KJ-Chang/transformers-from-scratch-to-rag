import numpy as np

def batch_search(queries, index, top_k=3, batch_size=8):
    """
    批次處理向量搜尋，將大量查詢分批進行以節省記憶體
    
    參數:
        queries: 查詢向量，形狀為 (n_queries, dimension)
        index: FAISS 索引對象
        top_k: 每個查詢要返回的最相似結果數量
        batch_size: 每批處理的查詢數量
        
    返回:
        all_distances: 所有查詢結果的距離矩陣，形狀為 (n_queries, top_k)
                      使用 np.vstack 將多個批次的結果垂直堆疊成一個矩陣
                      例如: 若有 100 個查詢，top_k=3，則結果為 (100, 3) 的矩陣
        all_idxs: 所有查詢結果的索引矩陣，形狀為 (n_queries, top_k)
                 同樣使用 np.vstack 將多個批次的結果垂直堆疊
    """
    # 初始化存儲所有批次結果的列表
    all_distances = []
    all_idxs = []
    
    # 依據 batch_size 分批處理查詢
    for i in range(0, queries.shape[0], batch_size):
        # 取出當前批次的查詢向量
        q_batch = queries[i:i+batch_size]
        # 執行向量搜尋
        # D: 距離矩陣 (batch_size, top_k)
        # I: 索引矩陣 (batch_size, top_k)
        D, I = index.search(q_batch, top_k)
        # 將當前批次的結果添加到列表中
        all_distances.append(D)  # D 的形狀是 (當前批次大小, top_k)
        all_idxs.append(I)      # I 的形狀是 (當前批次大小, top_k)
    
    # 使用 np.vstack 垂直堆疊所有批次結果
    # 這會將所有批次結果合併成一個大矩陣，保持每個查詢的結果順序
    # 例如：如果有 3 個批次，每個批次形狀為 (8, 3)，最終結果形狀為 (24, 3)
    return np.vstack(all_distances), np.vstack(all_idxs)