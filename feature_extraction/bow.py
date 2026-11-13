"""
BƯỚC 2a — TRÍCH CHỌN ĐẶC TRƯNG: BAG OF WORDS (BoW)  (bản viết gọn, dễ đọc)

Ý tưởng: mỗi bình luận -> một vector đếm số lần xuất hiện của từng từ / cụm 2 từ.

    "máy chạy êm , máy đẹp"  ->  {máy: 2, chạy: 1, êm: 1, ",": 1, đẹp: 1,
                                  "máy chạy": 1, "chạy êm": 1, ...}

Lưu ý quan trọng: CountVectorizer chỉ được fit trên tập TRAIN, rồi dùng từ điển đó
để transform tập TEST. Nếu fit trên toàn bộ dữ liệu thì từ điển "biết trước" từ của
tập test -> rò rỉ thông tin.

Đầu vào : data/processed/new_processed.csv
Đầu ra  : feature_extraction/features/bow.pkl
          feature_extraction/features/bow_vectorizer.pkl

Chạy từ thư mục dự án:  python feature_extraction/bow.py
"""
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "new_processed.csv")
FEAT_DIR = os.path.join(os.path.dirname(__file__), "features")

NGRAM_MAX = 2         # 1 = chỉ từ đơn; 2 = thêm cụm 2 từ ("tủ lạnh", "không tốt" thành 1 đặc trưng)
MIN_DF = 2            # bỏ token xuất hiện ở ít hơn 2 câu (lọc lỗi chính tả, từ hiếm)
MAX_FEATURES = 20000  # giữ tối đa 20.000 đặc trưng phổ biến nhất

# --- 1. Đọc dữ liệu và tách train / test ---
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
train = df[df["split"] == "train"]
test = df[df["split"] == "test"]

# --- 2. Tạo bộ vector hóa BoW ---
vectorizer = CountVectorizer(
    tokenizer=str.split,       # văn bản đã tách token (cách nhau dấu cách) ở bước 1
    token_pattern=None,        # tắt regex mặc định (nó sẽ bỏ dấu câu và từ 1 ký tự)
    lowercase=False,           # đã chữ thường ở bước 1
    ngram_range=(1, NGRAM_MAX),
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
)

# --- 3. fit CHỈ trên train, transform cho cả train và test ---
X_train = vectorizer.fit_transform(train["noi_dung"])
X_test = vectorizer.transform(test["noi_dung"])
y_train = train["label"].to_numpy()
y_test = test["label"].to_numpy()
feature_names = vectorizer.get_feature_names_out()

print(f"Kích thước từ điển: {len(feature_names):,} đặc trưng (n-gram 1..{NGRAM_MAX}, min_df={MIN_DF})")
print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
density = X_train.nnz / (X_train.shape[0] * X_train.shape[1])
print(f"Ma trận thưa: chỉ {density:.3%} phần tử khác 0 -> lưu dạng sparse")
n_empty = int((X_test.sum(axis=1) == 0).sum())
print(f"Số câu test không có token nào trong từ điển: {n_empty}")

# --- 4. Xem token nghiêng về spam / ham (so sánh tần suất trung bình giữa 2 lớp) ---
mean_spam = np.asarray(X_train[y_train == 1].mean(axis=0)).ravel()
mean_ham = np.asarray(X_train[y_train == 0].mean(axis=0)).ravel()
diff = mean_spam - mean_ham
print("\nTop token nghiêng về SPAM:", ", ".join(feature_names[np.argsort(-diff)[:15]]))
print("Top token nghiêng về HAM :", ", ".join(feature_names[np.argsort(diff)[:15]]))

# --- 5. Lưu kết quả ---
os.makedirs(FEAT_DIR, exist_ok=True)
joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
             "feature_names": feature_names, "test_text": test["noi_dung"].tolist()},
            f"{FEAT_DIR}/bow.pkl")
joblib.dump(vectorizer, f"{FEAT_DIR}/bow_vectorizer.pkl")
print(f"\nĐã lưu: {FEAT_DIR}/bow.pkl và {FEAT_DIR}/bow_vectorizer.pkl")
