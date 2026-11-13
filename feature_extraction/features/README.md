# `feature_extraction/features/`

Thư mục này chứa **đầu ra của bước trích chọn đặc trưng** (Bước 2 của pipeline): các ma trận đặc trưng đã vector hóa, nhãn, và các bộ vectorizer / mô hình Word2Vec đã huấn luyện. Các script trong `model/` đọc trực tiếp từ đây để huấn luyện và đánh giá bộ phân loại Spam / Ham.

> Các file trong thư mục này được **sinh tự động** bởi `bow.py`, `tfidf.py`, `WE.py` (nằm ở thư mục cha `feature_extraction/`). Không chỉnh sửa thủ công; nếu cần cập nhật, hãy chạy lại script tương ứng.

---

## 1. Nội dung thư mục

```text
feature_extraction/features/
├── bow.pkl                # Đặc trưng Bag-of-Words (X_train, X_test, y_train, y_test, ...)
├── bow_vectorizer.pkl     # CountVectorizer đã fit trên tập train
├── tfidf.pkl              # Đặc trưng TF-IDF
├── tfidf_vectorizer.pkl   # TfidfVectorizer đã fit trên tập train
├── we.pkl                 # Đặc trưng Word Embedding (vector câu từ Word2Vec)
├── we_word2vec.model      # Mô hình Word2Vec (gensim) đã huấn luyện trên tập train
└── README.md
```

| Phương pháp | Script sinh ra | File đặc trưng | File vectorizer / mô hình | Dạng ma trận | Mô hình dùng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Bag-of-Words | `bow.py` | `bow.pkl` | `bow_vectorizer.pkl` | Sparse, ≤ 20.000 chiều | `nb-bow.py`, `svm-bow.py` |
| TF-IDF | `tfidf.py` | `tfidf.pkl` | `tfidf_vectorizer.pkl` | Sparse, ≤ 20.000 chiều, chuẩn hóa L2 | `nb-tfidf.py`, `svm-tfidf.py` |
| Word Embedding | `WE.py` | `we.pkl` | `we_word2vec.model` | Dense, 100 chiều | `nb-we.py`, `svm-we.py` |

---

## 2. Cách tạo các file

Chạy từ **thư mục gốc** của dự án (`nlp-n2-ptit/`), sau khi đã chạy bước tiền xử lý (`preprocess/preprocess.py`):

```bash
python feature_extraction/bow.py
python feature_extraction/tfidf.py
python feature_extraction/WE.py        # cần: pip install gensim
```

- **Đầu vào:** file CSV đã tiền xử lý trong `data/processed/`, cần có các cột:
  - `noi_dung`: văn bản đã làm sạch và tách token (các token cách nhau bằng dấu cách),
  - `label`: nhãn (`0` = Ham, `1` = Spam),
  - `split`: `train` hoặc `test` (chia phân tầng 80/20, dùng chung cho cả 3 phương pháp).
- Nếu chỉ cần demo một mô hình, chỉ cần chạy script đặc trưng tương ứng.

> **Lưu ý:** hiện tại các script đọc `data/processed/new_processed.csv` (hằng số `DATA_PATH` ở đầu mỗi file), trong khi `README.md` gốc nhắc tới `processed.csv`. Hãy đảm bảo tên file đầu ra của bước tiền xử lý khớp với `DATA_PATH`, hoặc sửa `DATA_PATH` cho phù hợp.

---

## 3. Cấu hình trích chọn đặc trưng

| Tham số | BoW | TF-IDF | Word2Vec |
| :--- | :---: | :---: | :---: |
| Đơn vị | n-gram 1–2 | n-gram 1–2 | token đơn |
| `min_df` / `min_count` | 2 | 2 | 2 |
| `max_features` | 20.000 | 20.000 | – |
| Trọng số | tần suất thô | `sublinear_tf=True` (1 + log tf), IDF, chuẩn hóa L2 | – |
| Số chiều | ≤ 20.000 | ≤ 20.000 | 100 (`DIM`) |
| Thuật toán | `CountVectorizer` | `TfidfVectorizer` | Word2Vec Skip-gram (`SG=1`), `WINDOW=5`, `EPOCHS=30`, `SEED=42` |
| Gộp vector câu | – | – | mean pooling (`POOLING="mean"`, hoặc `"tfidf"` để trung bình có trọng số IDF) |

Các tham số nằm ở đầu mỗi script (`NGRAM_MAX`, `MIN_DF`, `MAX_FEATURES`, `DIM`, `WINDOW`, `POOLING`, ...), có thể chỉnh rồi chạy lại.

---

## 4. Cấu trúc dữ liệu trong từng file `.pkl`

Tất cả file `.pkl` được lưu bằng `joblib` và là một `dict`.

**`bow.pkl`, `tfidf.pkl`**

| Khóa | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `X_train`, `X_test` | `scipy.sparse` matrix | Ma trận đặc trưng tập train / test |
| `y_train`, `y_test` | `numpy.ndarray` | Nhãn tương ứng (0 = Ham, 1 = Spam) |
| `feature_names` | `numpy.ndarray` | Tên từng đặc trưng (token / cụm 2 token) |
| `test_text` | `list[str]` | Văn bản gốc của tập test (dùng để phân tích lỗi) |

**`we.pkl`**

| Khóa | Kiểu | Mô tả |
| :--- | :--- | :--- |
| `X_train`, `X_test` | `numpy.ndarray` (dense, 100 chiều) | Vector câu (trung bình các vector token) |
| `y_train`, `y_test` | `numpy.ndarray` | Nhãn tương ứng |
| `pooling` | `str` | Kiểu gộp vector đã dùng (`"mean"` hoặc `"tfidf"`) |
| `idf` | `dict` hoặc `None` | Bảng IDF nếu dùng `pooling="tfidf"` |
| `test_text` | `list[str]` | Văn bản tập test |

---

## 5. Cách nạp và sử dụng

```python
import joblib

# Nạp đặc trưng đã tính sẵn
data = joblib.load("feature_extraction/features/tfidf.pkl")
X_train, y_train = data["X_train"], data["y_train"]
X_test, y_test = data["X_test"], data["y_test"]

# Vector hóa một bình luận MỚI (văn bản phải được tiền xử lý và tách token giống bước 1)
vectorizer = joblib.load("feature_extraction/features/tfidf_vectorizer.pkl")
x_new = vectorizer.transform(["giao hàng nhanh , máy chạy êm"])
```

Với Word2Vec:

```python
from gensim.models import Word2Vec

w2v = Word2Vec.load("feature_extraction/features/we_word2vec.model")
print(w2v.wv.most_similar("tốt", topn=5))
```

> Để chuyển một câu mới thành vector 100 chiều, dùng hàm `doc_vector()` trong `WE.py` (trung bình vector các token có trong từ điển; câu không có token nào → vector 0).

---

## 6. Lưu ý quan trọng

1. **Chống rò rỉ dữ liệu (data leakage):** `CountVectorizer`, `TfidfVectorizer` và Word2Vec đều chỉ được **fit trên tập train**; tập test chỉ `transform`. Vì vậy không fit lại vectorizer trên toàn bộ dữ liệu.
2. **Chia train/test dùng chung:** cả ba phương pháp dùng cùng một phép chia (cột `split`), đảm bảo so sánh công bằng giữa các mô hình.
3. **Word Embedding có giá trị âm:** vector từ Word2Vec có thể âm nên không dùng được với `MultinomialNB`; dự án dùng `GaussianNB` (`nb-we.py`) và SVM RBF kèm `StandardScaler` trong `Pipeline` (`svm-we.py`).
4. **Từ ngoài từ điển (OOV):** token chưa từng thấy ở tập train bị bỏ qua khi vector hóa; câu không có token nào trong từ điển sẽ ra vector 0 (script có in số lượng câu như vậy ở tập test).
5. **Dữ liệu train nhỏ (~3,9 nghìn câu):** Word2Vec tự huấn luyện chưa thật tốt; có thể cải thiện bằng vector huấn luyện sẵn (ví dụ PhoW2V).
6. **Kích thước file:** các file `.pkl` / `.model` có thể nặng và có thể tái tạo bằng script, nên cân nhắc thêm vào `.gitignore` thay vì commit.