"""
A customized RAG pipeline using FAISS for retrieval and a pretrained language model for generation.
"""

import faiss
import torch
import yaml
from pathlib import Path
from util import scrape_and_chunk_website

from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration, FalconForCausalLM, AutoTokenizer

def load_config():
    """Load configuration from YAML file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class CustomRAGPipeline:
    def __init__(self, encoder, generator, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.encoder = encoder.to(device)
        self.tokenizer = tokenizer
        self.generator = generator.to(device)
        self.index = None
        self.device = device
        self.documents = []
        self.config = load_config()

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
        _, I = self.index.search(queries_embeddings, k=5)
        
        retrieved_passages = []
        for idxs in I:
            tmp = []
            for idx in idxs:
                tmp.append(self.documents[idx])
            retrieved_passages.append(tmp)
        return retrieved_passages
    
    def build_prompt(self, question, contexts):
        context = '\n'.join(contexts)
        prompt = f"""You are an AI assistant for question-answering tasks.
You must follow the following six strict rules.

STRICT RULES:
1. ONLY use information directly stated in the context
2. DO NOT generate any HTML tags or markup in your response
3. If information is not in the context, say "I cannot find this information in the context"
4. Provide clear and direct answers in plain text format
5. DO NOT include any formatting tags like <strong>, <em>, or other
6. Review the answer twice before outputing the final result  and keep the answer concise

Context:
{context}

Question: 
{question}

Answer:"""
        return prompt

    def answer_questions(self, queries: list[str]):
        contexts = self.search(queries)
        
        for q, c in zip(queries, contexts):
            prompt = self.build_prompt(q, c)
            if isinstance(self.generator, T5ForConditionalGeneration):
                input_ids = self.tokenizer(
                    prompt, 
                    truncation=True,
                    return_tensors='pt'
                ).input_ids.to(self.device)
                
                output = self.generator.generate(
                    input_ids,
                    **self.config['generation']
                )
                answer = self.tokenizer.decode(
                    output[0],
                    skip_special_tokens=True
                )
                
            elif isinstance(self.generator, FalconForCausalLM):
                inputs = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                ).to(self.device)

                output = self.generator.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **self.config['generation']
                )

                # GPT架構生成會包含prompt
                prompt_length = inputs.input_ids.shape[1]
                answer = self.tokenizer.decode(
                    output[0][prompt_length:],
                    skip_special_tokens=True
                )
            
            print(f'\nQuestion: {q}')
            print(f'Answer: {answer}')

def run_rag_pipeline_gpt(passages: list[str], queries: list[str]):
    """Run RAG pipeline with GPT (Falcon) as the generator."""
    config = load_config()
    gpt_config = config['models']['gpt']
    
    encoder = SentenceTransformer(gpt_config['encoder'])
    tokenizer = AutoTokenizer.from_pretrained(gpt_config['generator'])
    tokenizer.pad_token = tokenizer.eos_token

    generator = FalconForCausalLM.from_pretrained(
        gpt_config['generator'],
        trust_remote_code=True,
        torch_dtype=torch.float16  #  VRAM不足使用半精度
    )
    
    print('\n' + '='*50)
    print('Starting GPT (Falcon-7B) Pipeline')
    print(f'Encoder: {gpt_config["encoder"]}')
    print(f'Generator: {gpt_config["generator"]}')
    print('='*50 + '\n')
    
    rag = CustomRAGPipeline(encoder, generator, tokenizer)
    rag.add_documents(passages)
    rag.answer_questions(queries)

def run_rag_pipeline_t5(passages: list[str], queries: list[str]):
    """Run RAG pipeline with T5 as the generator."""
    config = load_config()
    t5_config = config['models']['t5']
    
    encoder = SentenceTransformer(t5_config['encoder'])
    tokenizer = T5Tokenizer.from_pretrained(t5_config['generator'])
    generator = T5ForConditionalGeneration.from_pretrained(t5_config['generator'])

    print('\n' + '='*50)
    print('Starting T5 Pipeline')
    print(f'Encoder: {t5_config["encoder"]}')
    print(f'Generator: {t5_config["generator"]}')
    print('='*50 + '\n')
    
    rag = CustomRAGPipeline(encoder, generator, tokenizer)
    rag.add_documents(passages)
    rag.answer_questions(queries)

def main():
    urls = [
        'https://www.royaltek.com/about/whoweare/',
        'https://www.royaltek.com/about/services/',
        'https://www.royaltek.com/about/thecompany/',
    ]

    passages = scrape_and_chunk_website(urls, 300, 35)

    queries = [
        "What products does Royaltek offer?",
        "In what year was RoyalTek founded?",
        "What types of products does RoyalTek offer for the automotive industry?",
        "How has RoyalTek benefited from being part of Quanta Inc. since 2006?",
        "What is RoyalTek's approach to customer service and product delivery?",
        "What technologies does RoyalTek integrate into their solutions for location awareness and vehicle safety?",
    ]
    
    run_rag_pipeline_t5(passages, queries)
    run_rag_pipeline_gpt(passages, queries)

if __name__ == '__main__':
    main()