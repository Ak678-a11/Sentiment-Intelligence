import pandas as pd
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =============================
# LOAD DATA
# =============================
df = pd.read_csv("twitter_training.csv", header=None)
df = df.iloc[:, :4]
df.columns = ["id", "entity", "label", "text"]

df["text"] = df["text"].fillna("").astype(str)
df["label"] = df["label"].astype(str).str.lower()

# =============================
# CLEAN TEXT
# =============================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+|#\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean)

X = df["text"]
y = df["label"]

# =============================
# TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =============================
# OPTIMIZED PIPELINE (FAST + STABLE)
# =============================
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,   # speed boost
        ngram_range=(1,2),   # better context
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        solver="saga",       # fast for large data
        n_jobs=-1
    ))
])

# =============================
# TRAIN
# =============================
model.fit(X_train, y_train)

# =============================
# EVALUATION
# =============================
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"Accuracy: {acc:.4f}")

# =============================
# SAVE MODEL
# =============================
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")