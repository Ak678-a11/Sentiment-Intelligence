# 🧠 Sentiment Intelligence  
### Interpretable Sentiment Analysis System

---

## 🚀 Overview

Sentiment Intelligence is a machine learning web application that classifies text into **positive, negative, or neutral sentiment**.

It also provides **word-level insights**, highlighting which words influenced the prediction.

---

## 📊 Dataset

- Source: Twitter Sentiment Dataset  
- Contains labeled tweets  

### Features:
- `text` → input sentence  
- `label` → sentiment (positive, negative, neutral)

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/sentiment-intelligence.git
cd sentiment-intelligence

2. Create virtual environment (optional)
Bash
python -m venv venv
venv\Scripts\activate

📦 Requirements
Install dependencies using:
Bash
pip install -r requirements.txt

🔹 requirements.txt
Plain text
streamlit
pandas
scikit-learn
plotly

▶️ Run the Application
Bash
streamlit run app.py

🔄 Workflow
Preprocessing
Clean text (remove links, symbols)
Vectorization
TF-IDF converts text into numbers
Model
Logistic Regression predicts sentiment
Output
Prediction + confidence score
Word highlighting

📈 Results
Accuracy: ~67%
Metrics:
Precision
Recall
F1-score
Confusion Matrix

🧪 How to Use
Open the app
Enter text
Click Analyze
View prediction + insights

📊 Insights
Sentiment distribution (pie chart)
Word frequency chart
Word highlighting
Model performance metrics

⚠️ Limitations
Limited context understanding
Cannot detect sarcasm
Depends on dataset quality

🔮 Future Improvements
Use BERT / Deep Learning
Improve accuracy
Handle complex language

📜 License
MIT License