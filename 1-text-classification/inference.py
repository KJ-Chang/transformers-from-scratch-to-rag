from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import yaml
import json
import torch

def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        output = model(**inputs)
        prediction = torch.nn.functional.softmax(output.logits, dim=-1)
        predicted_id = torch.argmax(prediction, dim=-1)
    
    return prediction, predicted_id


def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = config['model_path']
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    best_model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)

    texts = [
        "European leaders gather for climate change summit in Paris",
        "NBA team clinches playoff spot with decisive victory",
        "Tech company announces major acquisition worth $5 billion",
        "Scientists discover new species in Amazon rainforest"
    ]

    with open(f'{config["model_path"]}/id2label.json', 'r') as f:
        id2label = json.load(f)

    for text in texts:
        prediction, predicted_id = predict(text, tokenizer, best_model, device)
        print(f'Text: {text}  Predicted labels: {id2label[str(predicted_id[0].item())]} \n  Prediction: {prediction}')
        

        
if __name__ == '__main__':
    main()