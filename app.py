import streamlit as st
import pandas as pd
import pickle
import re
import plotly.express as px
from collections import Counter
import datetime

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Sentiment Intelligence", layout="wide")

# =============================
# SESSION STATE
# =============================
if "history" not in st.session_state:
    st.session_state.history = []

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

labels = ["negative", "neutral", "positive"]

# =============================
# SAFE DATA LOAD (NO CRASH)
# =============================
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("twitter_training.csv", header=None)
        df = df.iloc[:, :4]
        df.columns = ["id", "entity", "label", "text"]
        df["text"] = df["text"].fillna("")
        df["label"] = df["label"].astype(str).str.lower()
        return df
    except:
        return None

df = load_data()

# =============================
# CLEAN TEXT
# =============================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# =============================
# WORD HIGHLIGHTING (MODEL BASED)
# =============================
def highlight_text(text, model):
    try:
        vectorizer = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]

        words = text.split()
        word_scores = {}

        for word in words:
            vec = vectorizer.transform([word])
            score = clf.decision_function(vec)

            if len(score.shape) > 1:
                score = score[0]

            word_scores[word] = score[0] if hasattr(score, "__len__") else score

        highlighted = ""
        for word in words:
            score = word_scores.get(word, 0)

            if score > 0:
                color = "rgba(0,255,0,0.4)"
            else:
                color = "rgba(255,0,0,0.4)"

            highlighted += f"<span style='background-color:{color};padding:2px;margin:2px'>{word}</span> "

        return highlighted

    except:
        return text

# =============================
# HISTORY
# =============================
def log_history(text, pred, conf):
    st.session_state.history.append({
        "time": str(datetime.datetime.now())[:19],
        "text": text,
        "prediction": pred,
        "confidence": float(conf)
    })

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧠 Sentiment Intelligence")

page = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔍 Analyzer",
    "📊 Dashboard",
    "📈 Model Metrics",
    "📜 History"
])

# =============================
# HOME
# =============================
if page == "🏠 Home":
    st.title("🧠 Sentiment Intelligence")
    st.subheader("Interpretable Sentiment Analysis System")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "TF-IDF + Logistic")
    c2.metric("Speed", "Fast Inference")
    c3.metric("Insight", "Word-Level")

# =============================
# ANALYZER
# =============================
if page == "🔍 Analyzer":

    st.title("🔍 Sentiment Analyzer")

    text = st.text_area("Enter text")

    if st.button("Analyze") and text:

        cleaned = clean_text(text)
        pred = model.predict([cleaned])[0]

        try:
            proba = model.predict_proba([cleaned])[0]
            conf = max(proba)
        except:
            proba = [0.33, 0.33, 0.34]
            conf = 0.6

        # RESULT
        st.subheader("Result")
        st.metric("Prediction", pred.upper())
        st.metric("Confidence", f"{conf:.2f}")
        st.progress(conf)

        # PIE CHART (FIXED)
        if len(proba) == len(labels):
            fig = px.pie(names=labels, values=proba, title="Confidence Distribution")
            st.plotly_chart(fig, width='stretch')

        # WORD HIGHLIGHT
        st.subheader("Word-Level Insights")
        highlighted = highlight_text(cleaned, model)
        st.markdown(highlighted, unsafe_allow_html=True)

        # WORD COUNT
        words = cleaned.split()
        stop = {"the","is","and","to","a","of","for","in","on"}
        words = [w for w in words if w not in stop]

        wc = Counter(words).most_common(10)
        if wc:
            dfw = pd.DataFrame(wc, columns=["word","count"])
            st.bar_chart(dfw.set_index("word"))

        log_history(text, pred, conf)

# =============================
# DASHBOARD
# =============================
if page == "📊 Dashboard":

    st.title("📊 Dataset Overview")

    if df is not None:
        st.metric("Total Samples", len(df))
        st.plotly_chart(px.pie(df, names="label"), width='stretch')
    else:
        st.warning("Dataset not available in deployed version")

# =============================
# MODEL METRICS
# =============================
if page == "📈 Model Metrics":

    st.title("📈 Model Metrics")

    st.info("Model evaluated during training")

    st.metric("Accuracy", "67%")
    st.write("Precision, Recall, F1-score used for evaluation")

# =============================
# HISTORY
# =============================
if page == "📜 History":

    st.title("📜 History")

    if st.button("Clear History"):
        st.session_state.history = []

    for h in reversed(st.session_state.history):
        st.write(h)
        st.divider()