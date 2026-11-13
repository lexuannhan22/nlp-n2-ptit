"""
BƯỚC 2c — TRÍCH CHỌN ĐẶC TRƯNG: WORD EMBEDDING (Word2Vec)  (bản viết gọn, dễ đọc)

1. Huấn luyện Word2Vec (gensim) trên các câu của tập TRAIN: mỗi token -> 1 vector
   dày (dense) 100 chiều; token dùng trong ngữ cảnh giống nhau có vector gần nhau
   (vd "tốt" ~ "ổn" ~ "ok").
2. Vector của cả bình luận = TRUNG BÌNH các vector token trong câu (mean pooling).
   Tùy chọn POOLING = "tfidf": trung bình có trọng số IDF (token hiếm, nhiều thông tin
   được ưu tiên hơn).
   Câu không có token nào trong từ điển -> vector 0.

Khác với BoW/TF-IDF (vector thưa, hàng chục nghìn chiều, mỗi chiều = 1 từ),
WE cho vector dày, ít chiều, các giá trị có thể ÂM -> MultinomialNB không dùng được,
phải dùng GaussianNB (xem 3_model/nb-we.py).

Word2Vec chỉ được huấn luyện trên TRAIN để tập test hoàn toàn "chưa thấy".
(Dữ liệu train khá nhỏ ~3.9 nghìn câu nên vector chưa thật tốt; dùng vector
huấn luyện sẵn như PhoW2V sẽ cải thiện đáng kể.)

Đầu vào : data/processed/new_processed.csv
Đầu ra  : features/we.pkl            (X_train, X_test, y_train, y_test)
          features/we_word2vec.model (mô hình Word2Vec để biến đổi câu mới)

Cài đặt:  pip install gensim
Chạy (đứng trong thư mục BTL1):  python 2_trich_chon_dac_trung/WE.py
"""
import math
import os
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "new_processed.csv")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "features")

DIM = 100          # số chiều của mỗi vector từ
WINDOW = 5         # cửa sổ ngữ cảnh: xét 5 từ bên trái và 5 từ bên phải
MIN_COUNT = 2      # bỏ token xuất hiện ít hơn 2 lần
SG = 1             # 1 = Skip-gram, 0 = CBOW
EPOCHS = 30
POOLING = "mean"   # "mean": trung bình thường | "tfidf": trung bình có trọng số IDF
SEED = 42


def compute_idf(docs):
    """Tính IDF của từng token trên tập train (cùng công thức smooth của sklearn)."""
    n = len(docs)
    df = Counter(tok for doc in docs for tok in set(doc))
    return {tok: math.log((1 + n) / (1 + c)) + 1 for tok, c in df.items()}


def doc_vector(tokens, wv, idf=None):
    """Gộp vector các token trong câu thành 1 vector cho cả câu."""
    vecs, weights = [], []
    for tok in tokens:
        if tok in wv.key_to_index:                     # chỉ lấy token có trong từ điển Word2Vec
            vecs.append(wv[tok])
            weights.append(idf.get(tok, 1.0) if idf else 1.0)
    if not vecs:                                       # câu không có token nào -> vector 0
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.average(np.vstack(vecs), axis=0, weights=weights).astype(np.float32)


# --- 1. Đọc dữ liệu, tách token ---
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
train_tokens = [t.split() for t in train["noi_dung"]]
test_tokens = [t.split() for t in test["noi_dung"]]

# --- 2. Huấn luyện Word2Vec trên train ---
w2v = Word2Vec(
    sentences=train_tokens,
    vector_size=DIM,
    window=WINDOW,
    min_count=MIN_COUNT,
    sg=SG,
    epochs=EPOCHS,
    workers=1,       # 1 luồng -> kết quả lặp lại được
    seed=SEED,
)
wv = w2v.wv
print(f"Word2Vec ({'Skip-gram' if SG else 'CBOW'}): từ điển {len(wv):,} token, {DIM} chiều, {EPOCHS} epoch")

# Kiểm tra trực quan: các từ gần nghĩa nhất
for word in ["tốt", "ồn", "zalo", "giao", "phonetoken"]:
    if word in wv.key_to_index:
        gan = ", ".join(f"{w}({s:.2f})" for w, s in wv.most_similar(word, topn=6))
        print(f"  gần '{word}': {gan}")

# --- 3. Biến mỗi câu thành 1 vector ---
idf = compute_idf(train_tokens) if POOLING == "tfidf" else None
X_train = np.vstack([doc_vector(t, wv, idf) for t in train_tokens])
X_test = np.vstack([doc_vector(t, wv, idf) for t in test_tokens])
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()

n_oov = sum(1 for t in test_tokens if not any(tok in wv.key_to_index for tok in t))
print(f"\nPooling: {POOLING} | X_train: {X_train.shape}   X_test: {X_test.shape}")
print(f"Số câu test không có token nào trong từ điển (vector 0): {n_oov}")

# --- 4. Lưu kết quả ---
os.makedirs(FEAT_DIR, exist_ok=True)
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
             "pooling": POOLING, "idf": idf, "test_text": test["noi_dung"].tolist()},
            f"{FEAT_DIR}/we.pkl")
w2v.save(f"{FEAT_DIR}/we_word2vec.model")
print(f"\nĐã lưu: {FEAT_DIR}/we.pkl và {FEAT_DIR}/we_word2vec.model")
