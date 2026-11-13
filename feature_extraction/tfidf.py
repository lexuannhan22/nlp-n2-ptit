"""
BƯỚC 2b — TRÍCH CHỌN ĐẶC TRƯNG: TF-IDF  (bản viết gọn, dễ đọc)

Giống BoW nhưng mỗi số đếm được nhân thêm trọng số IDF:

    tfidf(t, d) = tf(t, d) * idf(t),   idf(t) = ln((1 + N) / (1 + df(t))) + 1

  - tf : số lần token t xuất hiện trong câu d (dùng sublinear: 1 + log(tf))
  - df : số câu chứa token t;  N: tổng số câu
  -> Token xuất hiện ở hầu hết các câu (",", "máy", "là"...) bị giảm trọng số,
     token hiếm mà đặc trưng ("zalo", "phonetoken", "trả góp") được tăng trọng số.
  Sau đó mỗi vector được chuẩn hóa L2 (độ dài = 1) để câu dài/ngắn so sánh được.

Vectorizer chỉ fit trên TRAIN (IDF tính từ train), rồi transform TEST.

"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "new_processed.csv")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "features")

NGRAM_MAX = 2
MIN_DF = 2
MAX_FEATURES = 20000

# --- 1. Đọc dữ liệu và tách train / test ---
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# --- 2. Tạo bộ vector hóa TF-IDF ---
vectorizer = TfidfVectorizer(
    tokenizer=str.split,
    token_pattern=None,
    lowercase=False,
    ngram_range=(1, NGRAM_MAX),
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    sublinear_tf=True,   # tf -> 1 + log(tf): giảm ảnh hưởng của từ lặp nhiều lần
    norm="l2",
)

# --- 3. fit CHỈ trên train, transform cho cả train và test ---
X_train = vectorizer.fit_transform(train["noi_dung"])
X_test = vectorizer.transform(test["noi_dung"])
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
feature_names = vectorizer.get_feature_names_out()

print(f"Kích thước từ điển: {len(feature_names):,} đặc trưng (n-gram 1..{NGRAM_MAX}, min_df={MIN_DF})")
print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")

# --- 4. Xem IDF: token phổ biến có IDF thấp, token hiếm có IDF cao ---
idf = vectorizer.idf_
thu_tu = np.argsort(idf)
print("\nIDF thấp nhất (token phổ biến, ít thông tin):",
      ", ".join(f"{feature_names[i]}({idf[i]:.2f})" for i in thu_tu[:10]))
print("IDF cao nhất (token hiếm):",
      ", ".join(f"{feature_names[i]}({idf[i]:.2f})" for i in thu_tu[-10:]))

# --- 5. Xem token nghiêng về spam / ham ---
mean_spam = np.asarray(X_train[y_train == 1].mean(axis=0)).ravel()
mean_ham = np.asarray(X_train[y_train == 0].mean(axis=0)).ravel()
diff = mean_spam - mean_ham
print("\nTop token nghiêng về SPAM:", ", ".join(feature_names[np.argsort(-diff)[:15]]))
print("Top token nghiêng về HAM :", ", ".join(feature_names[np.argsort(diff)[:15]]))

# --- 6. Lưu kết quả ---
os.makedirs(FEAT_DIR, exist_ok=True)
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
             "feature_names": feature_names, "test_text": test["noi_dung"].tolist()},
            f"{FEAT_DIR}/tfidf.pkl")
joblib.dump(vectorizer, f"{FEAT_DIR}/tfidf_vectorizer.pkl")
print(f"\nĐã lưu: {FEAT_DIR}/tfidf.pkl và {FEAT_DIR}/tfidf_vectorizer.pkl")
