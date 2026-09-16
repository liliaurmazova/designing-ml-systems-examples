"""Topic 5, Step 3: Classical text features vs dense embeddings — combined.

Task: 4-class news topic classification on ag_news (World / Sports /
Business / Sci-Tech).

Two feature families, then concatenated:

1) Classical manual features — cheap, interpretable, model-independent:
     * char_len, word_count, uppercase_ratio (3 statistical features)
     * TF-IDF over 1-2 grams, max_features=500 (sparse lexical features)

2) Dense embeddings — sentence-transformers/all-MiniLM-L6-v2 (384 dims):
     one vector per document capturing meaning, not surface tokens.

Leakage discipline: TF-IDF vocabulary/IDF and StandardScaler statistics
are fitted on TRAIN only. The embedding model is pre-trained on external
corpora, which is NOT leakage — it has never seen our labels.

Note on scaling: TF-IDF/embedding matrices are combined into one dense
matrix and standardized column-wise so LogisticRegression coefficients
live on comparable scales — which is exactly what makes the
"top informative features" ranking at the end meaningful.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_TRAIN, N_TEST = 8_000, 2_000
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

# --- Load a stratified-enough random subsample (ag_news is balanced) ---------
ds = load_dataset("fancyzhx/ag_news")
train = ds["train"].shuffle(seed=RANDOM_STATE).select(range(N_TRAIN)).to_pandas()
test = ds["test"].shuffle(seed=RANDOM_STATE).select(range(N_TEST)).to_pandas()
print(f"{len(train)} train / {len(test)} test docs, "
      f"labels: {np.bincount(train['label'])}")


# --- 1) Classical manual features --------------------------------------------
def stat_features(texts):
    """char length, word count, uppercase ratio — per document."""
    out = np.empty((len(texts), 3))
    for i, s in enumerate(texts):
        n = len(s)
        out[i] = (n, len(s.split()), sum(c.isupper() for c in s) / max(n, 1))
    return out


STAT_NAMES = ["char_len", "word_count", "uppercase_ratio"]
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=500)
X_tr_tfidf = tfidf.fit_transform(train["text"]).toarray()   # fit on TRAIN only
X_te_tfidf = tfidf.transform(test["text"]).toarray()
manual_names = STAT_NAMES + [f"tfidf:{t}" for t in tfidf.get_feature_names_out()]

# --- 2) Dense embeddings -----------------------------------------------------
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X_tr_emb = encoder.encode(train["text"].tolist(), batch_size=64,
                          show_progress_bar=False)
X_te_emb = encoder.encode(test["text"].tolist(), batch_size=64,
                          show_progress_bar=False)
print(f"Features: {len(STAT_NAMES)} stats + {X_tr_tfidf.shape[1]} tf-idf "
      f"+ {X_tr_emb.shape[1]} embedding dims")

# --- Combine and scale (fit on TRAIN only) -----------------------------------
X_tr = np.hstack([stat_features(train["text"]), X_tr_tfidf, X_tr_emb])
X_te = np.hstack([stat_features(test["text"]), X_te_tfidf, X_te_emb])
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

# --- Train and evaluate ------------------------------------------------------
clf = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)
clf.fit(X_tr, train["label"])
pred = clf.predict(X_te)

p, r, f1, _ = precision_recall_fscore_support(test["label"], pred,
                                              average="macro")
print(f"\nMacro Precision={p:.4f}  Recall={r:.4f}  F1={f1:.4f}\n")
print(classification_report(test["label"], pred, target_names=CLASS_NAMES,
                            digits=3))

# --- Top 10 most informative MANUAL features ---------------------------------
# coef_ is (4 classes x 887 features), columns 0..502 are the manual block
# (3 stats + 500 tf-idf). Because features were standardized, |coef| is a
# fair importance measure. For each manual feature take the class where its
# |coef| peaks.
n_manual = len(manual_names)
manual_coef = clf.coef_[:, :n_manual]
best_class = np.abs(manual_coef).argmax(axis=0)
strength = np.abs(manual_coef).max(axis=0)
top = np.argsort(strength)[::-1][:10]

print("Top 10 manual features by |coefficient| (class where it peaks):")
rows = [(manual_names[i], CLASS_NAMES[best_class[i]],
         manual_coef[best_class[i], i]) for i in top]
print(pd.DataFrame(rows, columns=["feature", "class", "coef"])
      .to_string(index=False))
