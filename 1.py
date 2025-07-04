import requests
import time
from tqdm import tqdm

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:32b"

def extract_prediction(response_text):
    text = response_text.lower()
    real_pos = text.rfind("real")
    fake_pos = text.rfind("fake")
    
    if real_pos > fake_pos and real_pos != -1:
        return "real"
    elif fake_pos > real_pos and fake_pos != -1:
        return "fake"
    else:
        if "true" in text or "真实" in text or "可信" in text:
            return "real"
        elif "false" in text or "不真实" in text or "虚假" in text:
            return "fake"
        else:
            print(f"警告: 无法提取结果: {response_text[:50]}...")
            return "fake"

def get_llm_prediction(text):
    prompt = f"""
    你是新闻验证专家，请判断以下新闻是真实还是虚假，若不确定则为假。

    只回答"real"或"fake"，不要包含其他内容。
    
    新闻内容: {text}
    """
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {"temperature": 0.0},
        "stream": False
    }
    
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
        model_response = response.json().get("response", "").strip()
        predicted = extract_prediction(model_response)
        
        if predicted not in ["real", "fake"] or len(model_response) > 10:
            print(f"非标准输出 (提取为'{predicted}'): {model_response[:80]}...")
            
        return predicted
    except Exception as e:
        print(f"API错误: {e}")
        return "fake"

def clean_label(label):
    return label.strip().replace('"', '').replace("'", "").lower()

def main():
    file_path = r"C:\Users\Kim\Desktop\学习\云计算与大数据\期末\数据\posts.txt"
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()[1:51]
    
    correct = 0
    total = len(lines)
    print(f"处理 {total} 条数据...")
    
    for line in tqdm(lines, desc="进度"):
        parts = line.strip().rsplit('\t', maxsplit=1)
        if len(parts) < 2:
            print(f"行格式错误: {line[:50]}...")
            continue
            
        post_text, raw_label = parts
        true_label = clean_label(raw_label)
        
        if true_label not in ["real", "fake"]:
            print(f"未知标签 '{true_label}' (原始: '{raw_label}')")
            true_label = "fake"
        
        predicted = get_llm_prediction(post_text)
        
        if predicted == true_label:
            correct += 1
            print(f"正确: {predicted}")
        else:
            print(f"错误: 预测='{predicted}', 真实='{true_label}'")
        
        time.sleep(0.5)
    
    accuracy = correct / total * 100
    print(f"\n处理完成")
    print(f"总条数: {total}")
    print(f"正确数: {correct}")
    print(f"准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()