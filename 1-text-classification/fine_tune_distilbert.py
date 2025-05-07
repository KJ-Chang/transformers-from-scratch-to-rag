import torch
import json
import yaml
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average='macro'
    )
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 載入資料集 (用 ag_news 做示範)
    dataset_name = config['dataset']
    dataset = load_dataset(dataset_name)
    num_labels = len(set(dataset['train']['label']))

    # 載入預訓練模型和tokenizer  
    model_name = config['model_name']
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)

    def tokenize_func(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(device)
    
    # 處理資料集
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=['text'],
        )
    
    print(f'Dataset features: {tokenized_dataset["train"].features}')

    # 訓練參數設定
    training_args = TrainingArguments(
        output_dir=config['output_dir'],
        num_train_epochs=10,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=128,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=config['logging_dir'], # 請先確保環境有安裝tensorboard，否則不會存
        logging_steps=100,
        logging_strategy='steps',
        logging_first_step=True,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True, # 訓練完成後，加載最佳模型
        # fp16=True,
        # fp16_full_eval=True
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
    )

    # 開始訓練
    trainer.train()

    # 保存最佳模型及tokenizer
    model_path = config['model_path']
    trainer.save_model(model_path)
    tokenizer.save_pretrained(model_path)

    id2label = {
        0: 'World',
        1: 'Sports',
        2: 'Business',
        3: 'Sci/Tech'
    }

    with open(f'{model_path}/id2label.json', 'w') as f:
        json.dump(id2label, f, indent=4)

    print(f'Model and tokenizer saved to {model_path}')


if __name__ == "__main__":
    main()