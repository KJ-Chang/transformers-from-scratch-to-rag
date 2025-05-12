"""
A customized RAG pipeline using FAISS for retrieval and a pretrained language model for generation.
"""

import faiss
import torch
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertForQuestionAnswering

class CustomRAGPipeline:
    def __init__(self, encoder_model_name, qa_model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.encoder = SentenceTransformer(encoder_model_name, device=device)
        self.qa_model = BertForQuestionAnswering.from_pretrained(qa_model_name).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(qa_model_name)
        self.index = None
        self.device = device
        self.documents = []

    def add_documents(self, docs: list[str]):
        self.documents.extend(docs)
        doc_embeddings = self.encoder.encode(self.documents, convert_to_numpy=True, show_progress_bar=True)
        
        if self.index is None:
            dimension = doc_embeddings.shape[1]
            self.index = faiss.IndexFlat(dimension)
            if self.device == 'cuda':
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.index.add(doc_embeddings)
        print(f'Added {len(docs)} documents to the index.')

    def search(self, quers: list[str]):
        queries = []
        queries.extend(quers)
        queries_embeddings = self.encoder.encode(queries, convert_to_numpy=True)
        _, ids = self.index.search(queries_embeddings, k=1)
        retrieved_passages = [self.documents[i] for id in ids for i in id]
        
        return retrieved_passages

    def answer_questions(self, queries: list[str]):
        questions = queries
        contexts = self.search(queries)
        inputs = self.tokenizer(questions, contexts, return_tensors='pt', padding=True, truncation=True).to(self.device)
    
        with torch.no_grad():
            outputs = self.qa_model(**inputs)

        # 提取 start 和 end logits
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        # 轉換為答案的起始和結束位置
        start_idx = torch.argmax(start_logits, dim=1)
        end_idx = torch.argmax(end_logits, dim=1)

        # 使用 tokenizer 解碼出答案
        for i in range(len(inputs.input_ids)):
            answer = self.tokenizer.convert_tokens_to_string(
                self.tokenizer.convert_ids_to_tokens(inputs.input_ids[i][start_idx[i]:end_idx[i]+1])
            )
            print('==================================================')
            print(f'\nQuestion{i+1}: {queries[i]}')
            print(f'\nContext{i+1}: {contexts[i]}')
            print(f'\nAnswer{i+1}: {answer}')

def main():
    # 使用 SoundCloud Addresses Terms of Use Allowing AI Training on Uploaded Music 文章
    url = 'https://pitchfork.com/news/soundcloud-addresses-terms-of-use-allowing-ai-training-on-uploaded-music/'
    headers = {
        'User-Agent': 'Mosilla/5.0'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    passages =[p.text for p in soup.find_all('p')]

    # 問題
    queries = [
        'What clause did SoundCloud add to its terms of use in February 2024 regarding AI?',
        'Has SoundCloud used artist content to train AI models according to their official statement?',
        'What are the intended AI use cases described by SoundCloud?',
    ]

    # SQuAD
    # dataset_name = 'squad'
    # dataset = load_dataset(dataset_name, split='train')
    # passages = [item['context'] for item in dataset]
    # 排除相同的context
    # passages = list(set(passages))

    # queries = [
    #     'Who developed the theory of relativity?',
    #     'Who invented the telephone?',
    #     'What is the capital of United States?',
    # ]
    
    rag = CustomRAGPipeline(
        encoder_model_name='all-MiniLM-L6-v2',
        qa_model_name='Raghan/bert-finetuned-squad',
        )
    
    rag.add_documents(passages)
    # retrieved_passages = rag.search(queries)
    rag.answer_questions(queries)
    

if __name__ == '__main__':
    main()