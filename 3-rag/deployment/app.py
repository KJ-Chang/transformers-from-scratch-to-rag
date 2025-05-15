import sys
from pathlib import Path

# 將父目錄添加到 Python 路徑中
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import json
from customized import CustomRAGPipeline, load_config
from chat_history import ChatHistoryManager
from util import scrape_and_chunk_website
from sentence_transformers import SentenceTransformer
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration, 
    FalconForCausalLM, 
    AutoTokenizer
)
import torch
import gc

app = FastAPI()
# 更新靜態文件和模板的路徑
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")

# 配置
config = load_config()
t5_config = config['models']['t5']
gpt_config = config['models']['gpt']

# 全局變量用於追踪當前加載的模型
current_pipeline = None
current_model_type = None

# 聊天歷史管理器
history_manager = ChatHistoryManager(Path(__file__).parent / "chat_history")

# 要爬取的所有網址
urls = [
    'https://www.royaltek.com/about/whoweare/',
    'https://www.royaltek.com/about/services/',
    'https://www.royaltek.com/about/thecompany/',
]

# 全局變量用於儲存文件內容
passages = None

def load_model(model_type: str):
    global current_pipeline, current_model_type
    is_change_model = False
    
    # 如果要加載的模型類型與當前加載的相同，直接返回
    if current_model_type == model_type and current_pipeline is not None:
        return current_pipeline, is_change_model
    
    # 清理當前模型的內存
    if current_pipeline is not None:
        del current_pipeline
        torch.cuda.empty_cache()
        gc.collect()
    
    # 加載新模型
    if model_type == 't5':
        encoder = SentenceTransformer(t5_config['encoder'])
        tokenizer = T5Tokenizer.from_pretrained(t5_config['generator'])
        generator = T5ForConditionalGeneration.from_pretrained(t5_config['generator'])
        current_pipeline = CustomRAGPipeline(encoder, generator, tokenizer)
    else:
        encoder = SentenceTransformer(gpt_config['encoder'])
        tokenizer = AutoTokenizer.from_pretrained(gpt_config['generator'])
        tokenizer.pad_token = tokenizer.eos_token
        generator = FalconForCausalLM.from_pretrained(
            gpt_config['generator'],
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        current_pipeline = CustomRAGPipeline(encoder, generator, tokenizer)
    
    is_change_model = True
    current_model_type = model_type
    return current_pipeline, is_change_model

# 儲存websocket連線
connections = []

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/conversation-history")
async def get_conversation_history():
    return JSONResponse(content=history_manager.get_recent_history())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global passages
    await websocket.accept()
    connections.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            data = json.loads(data)
            
            query = data['message']
            model_type = data['model']
            
            # 動態加載所選模型
            pipeline, is_change_model = load_model(model_type)
            
            # 如果還沒有爬取過網頁，則進行爬取
            if passages is None:
                passages = scrape_and_chunk_website(urls, 300, 35)
            
            if is_change_model:
                pipeline.add_documents(passages)
            
            # 取得答案
            answer = pipeline.answer_questions([query])[0]
            # contexts = pipeline.search([query])
            # prompt = pipeline.build_prompt(query, contexts[0])
            
            # if model_type == 't5':
            #     input_ids = pipeline.tokenizer(
            #         prompt, 
            #         truncation=True,
            #         return_tensors='pt'
            #     ).input_ids.to(pipeline.device)
                
            #     output = pipeline.generator.generate(
            #         input_ids,
            #         **config['generation']
            #     )
            #     answer = pipeline.tokenizer.decode(
            #         output[0],
            #         skip_special_tokens=True
            #     )
            # else:
            #     inputs = pipeline.tokenizer(
            #         prompt,
            #         truncation=True,
            #         padding=True,
            #         return_attention_mask=True,
            #         return_tensors='pt'
            #     ).to(pipeline.device)

            #     output = pipeline.generator.generate(
            #         input_ids=inputs.input_ids,
            #         attention_mask=inputs.attention_mask,
            #         pad_token_id=pipeline.tokenizer.pad_token_id,
            #         **config['generation']
            #     )
            #     prompt_length = inputs.input_ids.shape[1]
            #     answer = pipeline.tokenizer.decode(
            #         output[0][prompt_length:],
            #         skip_special_tokens=True
            #     )
            
            # 儲存對話歷史
            history_manager.add_conversation(query, answer, model_type)
            
            response = {
                'answer': answer,
                'model': model_type,
                'history': history_manager.get_recent_history()
            }
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        connections.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)