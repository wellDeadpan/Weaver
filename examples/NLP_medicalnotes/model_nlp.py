import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 下载必要的 NLTK 数据
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# 加载数据集
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

# 文本预处理
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# 主函数
def main():
    # 假设数据集有 'text' 和 'label' 列
    df = load_data(flpth)
    df['processed_text'] = df['text'].apply(preprocess_text)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )

    # 特征提取
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 训练分类器
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # 预测
    y_pred = classifier.predict(X_test_vec)

    # 评估
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()


flpth = "examples/NLP_medicalnotes/medical-nlp/data/X.csv"
df = load_data(flpth)  # or notes.csv, icd10.csv, etc.
print(df.head())

print(df.columns)