import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("IMDB Dataset.csv")

# Rename columns
df = df.rename(columns={
    "review": "text",
    "sentiment": "label"
})

# =========================================
# CLEAN TEXT
# =========================================

def clean_text(text):

    text = str(text).lower()

    # remove html
    text = re.sub(r"<.*?>", "", text)

    # remove urls
    text = re.sub(r"http\S+", "", text)

    # keep alphabets
    text = re.sub(r"[^a-z\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

df["text"] = df["text"].apply(clean_text)

# =========================================
# CUSTOM REAL-WORLD TRAINING DATA
# =========================================

custom_data = pd.DataFrame({

    "text": [

        # =====================================
        # POSITIVE
        # =====================================

        "this glass looks premium",
        "this phone looks premium",
        "premium quality product",
        "excellent quality",
        "beautiful product",
        "amazing quality",
        "awesome experience",
        "luxury feel",
        "great design",
        "worth buying",
        "highly recommended",
        "very elegant design",
        "nice premium finish",
        "good product",
        "excellent premium build quality",
        "this looks amazing",
        "very impressive",
        "super quality",
        "beautiful design",
        "great purchase",

        # NEGATION POSITIVE
        "not bad",
        "not bad actually",
        "not terrible",
        "not poor quality",

        # =====================================
        # NEGATIVE
        # =====================================

        "worst product ever",
        "terrible service",
        "waste of money",
        "poor quality",
        "bad experience",
        "very disappointing",
        "awful experience",
        "cheap quality",
        "not recommended",
        "completely useless",
        "horrible product",
        "very low quality",
        "extremely disappointing",

        # =====================================
        # NEUTRAL
        # =====================================

        "the water is clear",
        "the sky is blue",
        "this is a phone",
        "the glass is on the table",
        "this is a laptop",
        "the product exists",
        "this is a bottle",
        "the device is here",
        "the room is empty",
        "the screen is black"

    ],

    "label": [

        # POSITIVE
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",
        "positive","positive","positive","positive","positive",

        # NEGATION POSITIVE
        "positive","positive","positive","positive",

        # NEGATIVE
        "negative","negative","negative","negative","negative",
        "negative","negative","negative","negative","negative",
        "negative","negative","negative",

        # NEUTRAL
        "neutral","neutral","neutral","neutral","neutral",
        "neutral","neutral","neutral","neutral","neutral"
    ]
})

# Merge datasets
df = pd.concat([df, custom_data], ignore_index=True)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================================
# PIPELINE
# =========================================

pipeline = Pipeline([

    (
        "tfidf",

        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=15000,
            min_df=2
        )
    ),

    (
        "clf",

        LogisticRegression(
            max_iter=3000
        )
    )
])

# =========================================
# TRAIN
# =========================================

pipeline.fit(X_train, y_train)

# =========================================
# EVALUATE
# =========================================

preds = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, preds)

print(f"\nAccuracy: {accuracy:.4f}")

# =========================================
# SAVE MODEL
# =========================================

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully.")

# =========================================
# QUICK TESTS
# =========================================

tests = [

    "this glass looks premium",
    "amazing quality",
    "worst product ever",
    "not bad actually",
    "the water is clear",
    "the sky is blue",
    "beautiful premium design",
    "terrible service"

]

print("\nQuick Predictions:\n")

for t in tests:

    cleaned = clean_text(t)

    pred = pipeline.predict([cleaned])[0]

    probs = pipeline.predict_proba([cleaned])[0]

    confidence = max(probs)

    print(f"{t} --> {pred} ({confidence:.2f})")