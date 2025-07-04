import os
import requests
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:32b"

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, embeddings=None):
        self.texts = texts
        self.labels = labels
        self.embeddings = embeddings
        
    def __len__(self):
        return len(self.embeddings) if self.embeddings is not None else len(self.texts)
    
    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.embeddings[idx], self.labels[idx]
        else:
            return self.texts[idx], self.labels[idx]

def get_embedding(text):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"提取以下文本的特征向量: {text}"
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        return np.array(response.json()["embedding"])
    except Exception as e:
        print(f"获取嵌入时出错: {e}")
        return np.random.rand(768)

def extract_sentiment(text):
    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": f"判断以下文本的情感倾向（积极、消极、中性），只返回一个词: {text}"
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, stream=True)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if line:
                full_response += line.decode('utf-8')
        
        sentiment_text = ""
        for line in full_response.split('\n'):
            if line.strip():
                try:
                    data = eval(line)
                    if "response" in data:
                        sentiment_text += data["response"]
                except:
                    continue
        
        sentiment_text = sentiment_text.strip().lower()
        
        if "积极" in sentiment_text or "positive" in sentiment_text:
            return np.array([1, 0, 0])
        elif "消极" in sentiment_text or "negative" in sentiment_text:
            return np.array([0, 1, 0])
        else:
            return np.array([0, 0, 1])
    except Exception as e:
        print(f"获取情感时出错: {e}")
        return np.array([0, 0, 1])

def get_text_features(text):
    embedding = get_embedding(text)
    sentiment = extract_sentiment(text)
    return np.concatenate([embedding, sentiment])

class FakeNewsDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(FakeNewsDetector, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for embeddings, labels in train_loader:
            embeddings = embeddings.float()
            labels = labels.long()
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings = embeddings.float()
                labels = labels.long()
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  验证准确率: {val_accuracy:.2f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("模型已保存")
    
    return model

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.float()
            
            outputs = model(embeddings)
            _, predicted = torch.max(outputs, 1)
            
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.numpy())
    
    unique_labels = np.unique(all_labels)
    target_names = ['real', 'fake']
    
    print("分类报告:")
    try:
        print(classification_report(all_labels, all_predictions, target_names=target_names))
    except ValueError:
        actual_target_names = [target_names[i] for i in unique_labels]
        print(classification_report(all_labels, all_predictions, target_names=actual_target_names))
    
    print("混淆矩阵:")
    print(confusion_matrix(all_labels, all_predictions))

def predict_single_text(model, text):
    model.eval()
    feature = get_text_features(text)
    feature_tensor = torch.tensor(feature).float().unsqueeze(0)
    
    with torch.no_grad():
        prediction = model(feature_tensor)
        _, predicted_class = torch.max(prediction, 1)
        confidence = torch.max(prediction).item() * 100
        
    return predicted_class.item(), confidence

def main():
    data_path = r"C:\Users\Kim\Desktop\学习\云计算与大数据\期末\数据\data.txt"
    
    if not os.path.exists(data_path):
        print(f"错误：文件 '{data_path}' 不存在")
        return
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:101]
    
    print(f"已读取 {len(lines)} 条数据")
    
    texts = []
    labels = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split('\t')
        
        if len(parts) < 2:
            continue
            
        text = parts[1]
        label_str = parts[-1].strip().lower()
        
        if label_str == 'fake':
            label = 1
        elif label_str == 'real':
            label = 0
        else:
            continue
            
        texts.append(text)
        labels.append(label)
    
    if len(texts) == 0:
        print("错误：没有有效的数据用于训练")
        return
    
    print(f"有效数据数量: {len(texts)}")
    
    print("正在提取特征...")
    features = [get_text_features(text) for text in tqdm(texts)]
    
    features_array = np.array(features)
    labels_array = np.array(labels)
    
    feature_dim = features_array.shape[1]
    
    X_train, X_test, y_train, y_test = train_test_split(
        features_array, labels_array, test_size=0.2, random_state=42, stratify=labels_array
    )
    
    train_dataset = FakeNewsDataset(texts=None, labels=y_train, embeddings=X_train)
    test_dataset = FakeNewsDataset(texts=None, labels=y_test, embeddings=X_test)
    
    train_loader = DataLoader(train_dataset, batch_size=min(8, len(train_dataset)), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=min(8, len(test_dataset)))
    
    model = FakeNewsDetector(input_size=feature_dim)
    
    print("开始训练模型...")
    model = train_model(model, train_loader, test_loader)
    
    print("评估模型性能...")
    evaluate_model(model, test_loader)
    
    print("\n对所有数据进行预测:")
    for i, text in enumerate(texts):
        label, confidence = predict_single_text(model, text)
        predicted_label = 'fake' if label == 1 else 'real'
        actual_label = 'fake' if labels[i] == 1 else 'real'
        
        print(f"\n文本: {text[:100]}...")
        print(f"预测结果: {predicted_label} (置信度: {confidence:.2f}%)")
        print(f"实际标签: {actual_label}")

if __name__ == "__main__":
    main()