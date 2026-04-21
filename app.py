import streamlit as st
import pandas as pd
import pickle
import re
import plotly.express as px
import datetime
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Sentiment Intelligence",
    page_icon="🧠",
    layout="wide"
)

# =============================
# CUSTOM UI STYLING
# =============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #0f172a, #020617);
    color: white;
}
.card {
    background: #111827;
    padding: 20px;
    border-radius: 12px;
    margin-bottom: 15px;
}
.highlight-box {
    padding: 10px;
    border-radius: 8px;
    background: #1f2937;
}
</style>
""", unsafe_allow_html=True)

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
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
labels = list(model.classes_)

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df = pd.read_csv("twitter_training.csv", header=None)
    df = df.iloc[:, :4]
    df.columns = ["id", "entity", "label", "text"]
    df["text"] = df["text"].fillna("")
    df["label"] = df["label"].astype(str).str.lower()
    return df

df = load_data()

# =============================
# HISTORY FUNCTION
# =============================
def log_history(text, pred, conf):
    st.session_state.history.append({
        "time": str(datetime.datetime.now())[:19],
        "text": text,
        "prediction": pred,
        "confidence": float(conf)
    })

# =============================
# WORD HIGHLIGHT FUNCTION
# =============================
def highlight_text(text, word_impacts):
    words = re.findall(r"\b\w+\b", text)
    html = ""

    for w in words:
        impact = word_impacts.get(w.lower(), 0)

        if impact > 0:
            color = "rgba(34,197,94,0.35)"   # green
        elif impact < 0:
            color = "rgba(239,68,68,0.35)"  # red
        else:
            color = "transparent"

        html += f"<span style='background-color:{color}; padding:4px; margin:2px; border-radius:6px'>{w}</span> "

    return html

# =============================
# SIDEBAR
# =============================
st.sidebar.title("🧠 Sentiment Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Analyzer", "📊 Dashboard", "📊 Model Metrics", "📜 History"]
)

show_live = st.sidebar.toggle("Live History", value=True)

# =============================
# HOME
# =============================
if page == "🏠 Home":
    st.title("🧠 Sentiment Intelligence")
    st.subheader("Explainable AI Sentiment System")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "TF-IDF + Logistic Regression")
    col2.metric("Explainability", "Word-Level AI")
    col3.metric("Status", "Production Ready")

    st.info("Use the Analyzer to test predictions with explanations.")

# =============================
# ANALYZER
# =============================
if page == "🔍 Analyzer":

    st.title("🔍 Sentiment Analyzer")

    text = st.text_area("Enter your text")

    if st.button("Analyze") and text:

        pred = model.predict([text])[0]

        try:
            proba = model.predict_proba([text])[0]
            conf = max(proba)
        except:
            proba = [1/len(labels)] * len(labels)
            conf = 0.5

        st.subheader("Prediction Result")

        if pred == labels[0]:
            st.success(f"{pred.upper()} ({conf:.2f})")
        elif len(labels) > 1 and pred == labels[1]:
            st.error(f"{pred.upper()} ({conf:.2f})")
        else:
            st.info(f"{pred.upper()} ({conf:.2f})")

        st.progress(conf)

        # Pie chart
        st.subheader("Confidence Distribution")
        fig = px.pie(names=labels, values=proba)
        st.plotly_chart(fig, width="stretch")

        # Word frequency
        words = re.findall(r"\b\w+\b", text.lower())
        stop = {"the","is","and","to","a","of","for","in","on"}
        words = [w for w in words if w not in stop]

        wc = Counter(words).most_common(10)
        if wc:
            dfw = pd.DataFrame(wc, columns=["word","count"])
            st.subheader("Key Words")
            st.bar_chart(dfw.set_index("word"))

        # Highlight explanation
        st.subheader("🧠 AI Explanation")

        try:
            tfidf = model.named_steps["tfidf"]
            clf = model.named_steps["clf"]

            X_vec = tfidf.transform([text])
            feature_names = tfidf.get_feature_names_out()

            coefs = clf.coef_
            class_idx = list(model.classes_).index(pred)

            contributions = X_vec.toarray()[0] * coefs[class_idx]
            word_impacts = dict(zip(feature_names, contributions))

            html = highlight_text(text, word_impacts)
            st.markdown(html, unsafe_allow_html=True)

            st.caption("🟢 Positive | 🔴 Negative influence")

        except:
            st.warning("Explanation unavailable")

        log_history(text, pred, conf)

# =============================
# DASHBOARD
# =============================
if page == "📊 Dashboard":

    st.title("📊 Dataset Overview")

    st.metric("Total Samples", len(df))

    fig = px.pie(df, names="label", title="Sentiment Distribution")
    st.plotly_chart(fig, width="stretch")

# =============================
# MODEL METRICS
# =============================
if page == "📊 Model Metrics":

    st.title("📊 Model Performance")

    sample = df.sample(min(2000, len(df)), random_state=42)

    X = sample["text"]
    y_true = sample["label"]

    y_pred = model.predict(X)

    acc = accuracy_score(y_true, y_pred)
    st.metric("Accuracy", f"{acc*100:.2f}%")

    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        labels=dict(x="Predicted", y="Actual")
    )

    st.plotly_chart(fig, width="stretch")

# =============================
# HISTORY
# =============================
if page == "📜 History":

    st.title("📜 History")

    if st.button("Clear History"):
        st.session_state.history = []

    for item in reversed(st.session_state.history):
        st.write(f"🕒 {item['time']}")
        st.write(item["text"])
        st.write(f"{item['prediction']} ({item['confidence']:.2f})")
        st.divider()

# =============================
# LIVE SIDEBAR
# =============================
if show_live:
    st.sidebar.subheader("Recent")

    for item in st.session_state.history[-5:][::-1]:
        st.sidebar.write(f"{item['prediction']} ({item['confidence']:.2f})")