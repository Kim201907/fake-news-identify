import re
import requests
import nltk
import gensim
import pyLDAvis.gensim_models
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from tqdm import tqdm

# 下载NLTK资源
nltk.download('stopwords')
nltk.download('wordnet')

# 配置
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "deepseek-r1:32b"
NUM_TOPICS = 3  # 主题数量
NUM_WORDS = 10  # 每个主题显示的关键词数量
DATA_PATH = r"C:\Users\Kim\Desktop\学习\云计算与大数据\期末\数据\data.txt"

# 数据准备：提取前10条新闻文本
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:11] 
    texts = [line.strip().split('\t')[1] for line in lines]  # 提取post_text
    return texts

# 数据预处理
def preprocess_text(text):
    # 去除非字母字符，转为小写
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # 分词
    words = text.split()
    # 去停用词和短词
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words and len(w) > 2]
    # 词形还原
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

# 调用大模型分析主题
def analyze_topic_with_llm(keywords):
    prompt = f"""
    请分析以下关键词集合，总结它们代表的Twitter主题内容：
    关键词：{', '.join(keywords)}
    请用简洁的语言描述该主题的核心内容和讨论焦点。
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3}
    }
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        return response.json()["response"].strip()
    except Exception as e:
        return f"分析失败: {str(e)}"

# 生成词云
def generate_wordcloud(words, topic_id):
    wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(words)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title(f"Topic {topic_id + 1} Word Cloud")
    plt.show()

# 主函数
def main():
    # 1. 数据准备
    texts = load_data(DATA_PATH)
    print(f"加载了 {len(texts)} 条文本数据")

    # 2. 数据预处理
    processed_texts = [preprocess_text(text) for text in tqdm(texts, desc="预处理")]
    processed_texts = [t for t in processed_texts if t]  # 过滤空列表

    # 3. 模型构建与训练
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    lda_model = models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS,
        random_state=100,
        update_every=1,
        chunksize=100,
        passes=10,
        alpha='auto'
    )

    # 4. 可视化分析
    # LDA交互可视化
    lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(lda_vis, 'lda_visualization.html')
    print("LDA可视化已保存为 lda_visualization.html")

    # 主题关键词与词云
    topics = lda_model.show_topics(formatted=False, num_words=NUM_WORDS)
    for idx, (topic_id, word_probs) in enumerate(topics):
        words = [word for word, prob in word_probs]
        word_freq = {word: prob for word, prob in word_probs}
        
        # 词云
        generate_wordcloud(word_freq, idx)
        
        # 大模型主题分析
        print(f"\n主题 {idx + 1} 关键词: {', '.join(words)}")
        analysis = analyze_topic_with_llm(words)
        print(f"主题 {idx + 1} 分析: {analysis}")

if __name__ == "__main__":
    main()