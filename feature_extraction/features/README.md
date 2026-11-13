# 📦 Đặc Trưng Mô Hình (Model Features) — Kho Lưu Trữ Vector Biểu Diễn

Tài liệu này cung cấp **đặc tả kỹ thuật chi tiết, cấu trúc dữ liệu, cơ sở toán học và hướng dẫn sử dụng** cho toàn bộ các tệp ma trận đặc trưng và bộ vectorizer được lưu trữ trong thư mục `feature_extraction/features/`.

Đây là kết quả đầu ra trực tiếp của giai đoạn trích chọn đặc trưng (Feature Extraction) từ dữ liệu tiền xử lý `data/processed/new_processed.csv`, đồng thời là đầu vào chuẩn hóa phục vụ trực tiếp cho các mô hình học máy tại thư mục `model/`.

> ⚠️ **Quy ước kiểm soát phiên bản (Git):** Để tối ưu hóa dung lượng kho mã nguồn, thư mục này được cấu hình trong `.gitignore` nhằm bỏ qua các tệp nhị phân lớn (`*.pkl`, `*.model`) và **chỉ theo dõi duy nhất tệp tài liệu này (`README.md`)**. Toàn bộ các tệp đặc trưng có thể được tái tạo tự động bất kỳ lúc nào bằng các script tương ứng trong thư mục `feature_extraction/`.

---

## 🗂️ Cấu Trúc Thư Mục & Danh Mục Tệp Tin

```text
feature_extraction/features/
├── README.md                  # Tài liệu hướng dẫn kỹ thuật & đặc tả đặc trưng (Tệp này)
│
├── bow.pkl                    # Bộ ma trận đặc trưng BoW (Train/Test) + Nhãn + Tên token (1,71 MB)
├── bow_vectorizer.pkl         # Đối tượng CountVectorizer đã fit trên tập Train (320 KB)
│
├── tfidf.pkl                  # Bộ ma trận đặc trưng TF-IDF (Train/Test) + Nhãn + Tên token (1,71 MB)
├── tfidf_vectorizer.pkl       # Đối tượng TfidfVectorizer đã fit trên tập Train (397 KB)
│
├── we.pkl                     # Bộ ma trận đặc trưng Word Embedding 100D (Train/Test) + Nhãn (2,22 MB)
└── we_word2vec.model          # Mô hình Word2Vec (Skip-gram, gensim) đã train trên tập Train (1,79 MB)
```

### 📋 Bảng tổng hợp thông số kỹ thuật các tệp tin

| Tên tệp tin | Dung lượng | Định dạng | Kiểu đối tượng Python | Script sinh tệp | Mô tả và mục đích sử dụng |
|---|:---:|:---:|---|---|---|
| `bow.pkl` | **1,71 MB** | Pickle (`joblib`) | `dict` chứa `scipy.sparse.csr_matrix` & `numpy.ndarray` | `feature_extraction/bow.py` | Huấn luyện & đánh giá các mô hình dựa trên tần suất đếm từ (`nb-bow`, `svm-bow`) |
| `bow_vectorizer.pkl` | **320 KB** | Pickle (`joblib`) | `sklearn.feature_extraction.text.CountVectorizer` | `feature_extraction/bow.py` | Biến đổi (Transform) các câu đánh giá mới trong giai đoạn suy luận (Inference) |
| `tfidf.pkl` | **1,71 MB** | Pickle (`joblib`) | `dict` chứa `scipy.sparse.csr_matrix` & `numpy.ndarray` | `feature_extraction/tfidf.py` | Huấn luyện & đánh giá các mô hình dựa trên trọng số TF-IDF (`nb-tfidf`, `svm-tfidf`) |
| `tfidf_vectorizer.pkl` | **397 KB** | Pickle (`joblib`) | `sklearn.feature_extraction.text.TfidfVectorizer` | `feature_extraction/tfidf.py` | Biến đổi (Transform) câu văn mới sang vector trọng số TF-IDF cho inference |
| `we.pkl` | **2,22 MB** | Pickle (`joblib`) | `dict` chứa `numpy.ndarray` (Dense) & metadata | `feature_extraction/WE.py` | Huấn luyện & đánh giá các mô hình dựa trên ngữ nghĩa embedding (`nb-we`, `svm-we`) |
| `we_word2vec.model` | **1,79 MB** | Gensim Model | Đối tượng `gensim.models.Word2Vec` | `feature_extraction/WE.py` | Trích xuất vector từ vựng, tính toán tương đồng ngữ nghĩa và vector hóa câu mới |

---

## 📐 Đặc Tả Chi Tiết Từng Phương Pháp Trích Chọn Đặc Trưng

### 1. Bag-of-Words (BoW) — Mô hình túi đếm từ & cụm từ
* **Nguyên lý hoạt động:** Biểu diễn mỗi đánh giá thành một vector đếm số lần xuất hiện của các từ đơn (Unigram) và cụm 2 từ liền kề (Bigram). Cấu trúc ngữ pháp và thứ tự câu bị bỏ qua, chỉ tập trung vào sự hiện diện và tần suất lặp lại của token.
* **Cơ sở toán học:**
  $$\vec{x} = \left[ c(w_1, d), c(w_2, d), \dots, c(w_{|V|}, d) \right]$$
  Trong đó $c(w_i, d)$ là số lần token hoặc cụm token $w_i$ xuất hiện trong văn bản $d$, $|V|$ là kích thước từ điển ($|V| = 9.821$).
* **Cấu hình siêu tham số (Hyperparameters):**
  - `ngram_range=(1, 2)`: Trích xuất cả từ đơn (vd: `sản_phẩm`, `chất_lượng`) và cặp từ ghép (vd: `không tốt`, `quá tệ`).
  - `min_df=2`: Loại bỏ các token chỉ xuất hiện 1 lần trong toàn bộ tập Train (nhằm lọc lỗi gõ máy, từ cực hiếm, giảm chiều không gian và chống overfitting).
  - `max_features=20000`: Giới hạn tối đa 20.000 đặc trưng phổ biến nhất. Thực tế từ vựng thỏa mãn điều kiện `min_df=2` đạt **9.821 đặc trưng**.
  - `tokenizer=str.split`: Sử dụng trực tiếp phân tách khoảng trắng do văn bản đã được tiền xử lý và tách từ ghép bằng dấu gạch dưới (`_`) qua `underthesea`.
  - `lowercase=False`, `token_pattern=None`: Bảo toàn nguyên vẹn định dạng token đã qua chuẩn hóa ở bước tiền xử lý.
* **Đặc tả Schema tệp `bow.pkl`:**
  - `X_train`: `scipy.sparse.csr_matrix` kích thước `(4277, 9821)`, kiểu `int64`. Độ thưa (Sparsity) đạt **99,77%** (chỉ có 96.227 phần tử khác 0, mật độ ~0,229%).
  - `X_test`: `scipy.sparse.csr_matrix` kích thước `(1070, 9821)`, kiểu `int64`. Có 24.391 phần tử khác 0 (mật độ ~0,232%).
  - `y_train`: `numpy.ndarray` kích thước `(4277,)`, kiểu `int64` (4.165 Ham / 112 Spam).
  - `y_test`: `numpy.ndarray` kích thước `(1070,)`, kiểu `int64` (1.042 Ham / 28 Spam).
  - `feature_names`: `numpy.ndarray` gồm 9.821 chuỗi tên token/n-gram.
  - `test_text`: `list` gồm 1.070 chuỗi văn bản gốc của tập test (phục vụ phân tích lỗi định tính).
* **Đặc trưng phân biệt nổi bật (Hiệu số tần suất trung bình giữa 2 lớp):**
  - *Nghiêng về SPAM:* `em`, `mình`, `cần`, `ai`, `nhắn`, `giá`, `nha`, `nhé`, `inbox`, `shop`
  - *Nghiêng về HAM:* `numtoken`, `không`, `tốt`, `rất`, `dùng`, `sản_phẩm`, `máy`, `ok`, `nấu`, `là`

---

### 2. TF-IDF (Term Frequency – Inverse Document Frequency)
* **Nguyên lý hoạt động:** Kế thừa từ điển đặc trưng của BoW nhưng khắc phục nhược điểm "các từ xuất hiện phổ biến trong mọi câu làm lấn át các từ mang tính đặc thù". Những từ xuất hiện ở hầu hết câu (`máy`, `là`, `và`...) bị giảm trọng số, trong khi các từ hiếm và mang tính phân loại cao (`inbox`, `zalo`, `phonetoken`, `trả góp`...) được tăng mạnh trọng số.
* **Cơ sở toán học:**
  $$\text{tfidf}(t, d) = \text{tf}_{\text{sublinear}}(t, d) \times \text{idf}(t)$$
  $$\text{tf}_{\text{sublinear}}(t, d) = 1 + \ln(\text{tf}(t, d)) \quad (\text{khi } \text{tf}(t, d) > 0)$$
  $$\text{idf}(t) = \ln\left( \frac{1 + N}{1 + \text{df}(t)} \right) + 1$$
  Trong đó $N$ là tổng số văn bản trong tập Train ($N = 4.277$), $\text{df}(t)$ là số văn bản chứa token $t$.
  Sau đó, mỗi vector được chuẩn hóa L2 (Euclidean Norm) về độ dài đơn vị ($\|\vec{x}\|_2 = 1$):
  $$\vec{x}_{\text{norm}} = \frac{\vec{x}}{\|\vec{x}\|_2} = \frac{\vec{x}}{\sqrt{\sum_{i=1}^{|V|} x_i^2}}$$
* **Cấu hình siêu tham số (Hyperparameters):**
  - Cấu hình từ điển: `ngram_range=(1, 2)`, `min_df=2`, `max_features=20000`, `tokenizer=str.split`.
  - `sublinear_tf=True`: Sử dụng thang đo logarit $1 + \ln(\text{tf})$ thay vì đếm tuyến tính thô, giúp giảm tác động chi phối của các từ lặp lại nhiều lần trong câu dài.
  - `norm="l2"`: Chuẩn hóa L2 giúp loại bỏ ảnh hưởng của độ dài văn bản, đưa các đánh giá ngắn và dài về cùng một thang đo tương quan.
* **Đặc tả Schema tệp `tfidf.pkl`:**
  - `X_train`: `scipy.sparse.csr_matrix` kích thước `(4277, 9821)`, kiểu `float64`.
  - `X_test`: `scipy.sparse.csr_matrix` kích thước `(1070, 9821)`, kiểu `float64`.
  - `y_train`, `y_test`, `feature_names`, `test_text`: Định dạng và cấu trúc hoàn toàn đồng bộ với `bow.pkl`.
* **Đặc trưng phân biệt nổi bật (Hiệu số trọng số TF-IDF trung bình giữa 2 lớp):**
  - *Nghiêng về SPAM:* `em`, `nhắn`, `cần`, `mình`, `inbox`, `ai`, `shop`, `nha`, `nhé`, `nhận`
  - *Nghiêng về HAM:* `tốt`, `rất`, `dùng`, `ok`, `sản_phẩm`, `máy`, `không`, `sử_dụng`, `nấu`, `xài`

---

### 3. Word Embedding (Word2Vec Skip-gram + Mean Pooling)
* **Nguyên lý hoạt động:**
  1. Thay vì sử dụng không gian vector thưa hàng chục nghìn chiều độc lập (xem các từ đồng nghĩa như những chiều trực giao không liên quan), phương pháp này ánh xạ mỗi từ vựng sang một vector thực liên tục dày đặc (dense) gồm **100 chiều**.
  2. Các từ có ngữ cảnh sử dụng tương đồng (ví dụ: `tốt` ~ `ổn` ~ `ok`, `chất_lượng` ~ `tuyệt_vời`) sẽ nằm gần nhau trong không gian vector đa chiều (đo bằng độ tương đồng Cosine).
  3. Để tạo vector đại diện cho cả câu đánh giá $d$, áp dụng kỹ thuật **Mean Pooling** (trung bình cộng không trọng số của tất cả các vector từ xuất hiện trong câu):
     $$\vec{x}_d = \frac{1}{|V_d|} \sum_{w \in V_d} \vec{v}_w$$
     Trong đó $V_d$ là tập hợp các token trong câu $d$ có mặt trong từ điển Word2Vec. Nếu câu không chứa bất kỳ từ nào thuộc từ điển (Out-Of-Vocabulary hoàn toàn), câu sẽ nhận vector toàn số 0 ($\vec{0} \in \mathbb{R}^{100}$).
* **Cấu hình huấn luyện Word2Vec:**
  - Kiến trúc: **Skip-gram** (`sg = 1`) — tối ưu cho tập dữ liệu có quy mô vừa và nhỏ, nắm bắt tốt các từ hiếm và sắc thái từ vựng đặc trưng.
  - Số chiều vector: `vector_size = 100` (`DIM = 100`).
  - Cửa sổ ngữ cảnh: `window = 5` (xét 5 từ phía trước và 5 từ phía sau).
  - Tần suất tối thiểu: `min_count = 2`.
  - Số vòng lặp huấn luyện: `epochs = 30`.
  - Tái lập kết quả (Reproducibility): `workers = 1`, `seed = 42`.
  - Quy mô từ điển Word2Vec: **2.238 từ vựng**.
* **Đặc tả Schema tệp `we.pkl`:**
  - `X_train`: `numpy.ndarray` kích thước `(4277, 100)`, kiểu `float32`. Ma trận dày (dense), các giá trị có thể mang dấu **âm** hoặc **dương** trong miền số thực.
  - `X_test`: `numpy.ndarray` kích thước `(1070, 100)`, kiểu `float32`.
  - `pooling`: `"mean"`.
  - `idf`: `None` (chỉ kích hoạt nếu chuyển đổi cấu hình sang TF-IDF weighted pooling).
  - `test_text`: `list` gồm 1.070 chuỗi văn bản gốc của tập test.
  - Tỷ lệ vector rỗng (Zero vector do OOV hoàn toàn): Tập Train có **14 mẫu**, tập Test có **3 mẫu**.
* **Tệp `we_word2vec.model`:**
  - Lưu trữ toàn bộ trọng số mạng neural Word2Vec của thư viện `gensim`.
  - Cho phép truy vấn các từ gần nghĩa nhất (`wv.most_similar`), tính tương đồng cosin giữa các cặp từ (`wv.similarity`), hoặc trích xuất embedding cho từ vựng mới.

---

## ⚖️ Bảng So Sánh Đối Chiếu 3 Phương Pháp Biểu Diễn

| Tiêu chí kỹ thuật | Bag-of-Words (BoW) | TF-IDF | Word Embedding (Word2Vec) |
|---|:---:|:---:|:---:|
| **Không gian biểu diễn** | Thưa (Sparse CSR Matrix) | Thưa (Sparse CSR Matrix) | Dày đặc (Dense NumPy Array) |
| **Số chiều đặc trưng (Dimensions)** | **9.821** chiều | **9.821** chiều | **100** chiều |
| **Miền giá trị** | Số nguyên $\ge 0$ (Đếm tần suất) | Số thực $[0, 1]$ (Chuẩn hóa L2) | Số thực $[-\infty, +\infty]$ (Có giá trị âm) |
| **Mật độ phần tử khác 0 (Density)** | ~0,23% (Thưa 99,77%) | ~0,23% (Thưa 99,77%) | 100% (Dày đặc hoàn toàn) |
| **Khả năng hiểu từ đồng nghĩa** | ❌ Không (Mỗi từ là 1 chiều độc lập) | ❌ Không | ✅ Có (Khoảng cách cosin phản ánh ngữ nghĩa) |
| **Giảm nhiễu từ phổ biến** | ❌ Phụ thuộc vào `min_df` | ✅ Tự động qua trọng số nghịch đảo IDF | ⚠️ Một phần (qua subsampling trong Word2Vec) |
| **Xử lý từ mới (OOV)** | Bỏ qua token không có trong từ điển | Bỏ qua token không có trong từ điển | Bỏ qua token; gán vector $\vec{0}$ nếu toàn bộ câu OOV |
| **Thuật toán học máy tương thích tối ưu** | Multinomial Naive Bayes, Linear SVM | Complement NB, Linear SVM, Logistic Regression | RBF SVM, Gaussian NB, Neural Networks (MLP) |
| **Hiệu năng thực tế trên lớp Spam (F1-Score)** | **Linear SVM: 92,59%** · NB: 80,65% | Linear SVM: 89,29% · NB: 70,42% | **RBF SVM: 90,57%** · NB: 50,49% |

---

## 🔒 Đồng Bộ Phân Chia Dữ Liệu & Phòng Chống Rò Rỉ Thông Tin (Zero Data Leakage)

Một lỗi kỹ thuật nghiêm trọng trong xử lý ngôn ngữ tự nhiên là tính toán từ điển (vocabulary) hoặc tần suất tài liệu (IDF) trên toàn bộ dữ liệu trước khi chia train/test. Quy trình tại thư mục này tuân thủ nguyên tắc **cô lập dữ liệu tuyệt đối**:

1. **Đồng bộ phân chia (Dataset Alignment):**
   - Dữ liệu đầu vào bắt buộc là `data/processed/new_processed.csv` gồm đúng **5.347 mẫu**.
   - Phân chia cứng theo cột `split` với tỷ lệ chuẩn hóa 80/20 có phân tầng nhãn (Stratified Split):
     * **Tập Train (80%):** 4.277 mẫu (4.165 Ham - 97,38%, 112 Spam - 2,62%).
     * **Tập Test (20%):** 1.070 mẫu (1.042 Ham - 97,38%, 28 Spam - 2,62%).
2. **Quy tắc Fit & Transform:**
   - `CountVectorizer` và `TfidfVectorizer`: **Chỉ được gọi `.fit_transform()` trên tập Train**. Toàn bộ từ điển 9.821 đặc trưng và bảng trọng số IDF chỉ được học từ 4.277 câu huấn luyện. Tập Test chỉ được gọi `.transform()` dựa trên từ điển cố định sẵn có.
   - `Word2Vec`: Mô hình embedding chỉ học ngữ cảnh từ các câu của tập Train. Tập Test hoàn toàn "vô hình" trong giai đoạn học biểu diễn vector.

---

## 💻 Hướng Dẫn Sử Dụng & Tích Hợp Mã Nguồn (Code Snippets)

### 1. Nạp đặc trưng để huấn luyện mô hình (Chuẩn cấu trúc `model/evaluate.py`)
```python
from pathlib import Path
import joblib

FEAT_DIR = Path("feature_extraction/features")

# Nạp đặc trưng BoW (hoặc thay bằng "tfidf.pkl", "we.pkl")
data = joblib.load(FEAT_DIR / "bow.pkl")

X_train = data["X_train"]      # scipy.sparse.csr_matrix (4277, 9821)
X_test  = data["X_test"]       # scipy.sparse.csr_matrix (1070, 9821)
y_train = data["y_train"]      # numpy.ndarray (4277,)
y_test  = data["y_test"]       # numpy.ndarray (1070,)
feature_names = data["feature_names"]

print(f"X_train shape: {X_train.shape}, Số nhãn Spam trong train: {y_train.sum()}")
```

### 2. Suy luận (Inference) trên câu đánh giá mới

#### Trường hợp A: Sử dụng BoW hoặc TF-IDF Vectorizer
```python
import joblib

# Nạp bộ vectorizer đã lưu
vectorizer = joblib.load("feature_extraction/features/tfidf_vectorizer.pkl")

# Câu văn mới đã qua tiền xử lý (tách từ underthesea, chuẩn hóa teencode & token)
new_reviews = [
    "nồi_cơm điện dùng rất tốt nấu cơm dẻo ngon numtoken sao",
    "kết_bạn zalo em số phonetoken nhận giá sỉ cực ưu_đãi nhé"
]

# Chuyển đổi sang ma trận đặc trưng TF-IDF
X_new = vectorizer.transform(new_reviews)

# Nạp mô hình phân loại để dự đoán
model = joblib.load("results/svm-tfidf.pkl")
y_pred = model.predict(X_new)

print("Kết quả dự đoán (0 = HAM, 1 = SPAM):", y_pred)
```

#### Trường hợp B: Sử dụng mô hình Word2Vec
```python
import joblib
import numpy as np
from gensim.models import Word2Vec

# Nạp mô hình Word2Vec
w2v = Word2Vec.load("feature_extraction/features/we_word2vec.model")
wv = w2v.wv

def review_to_vector(text: str, wv) -> np.ndarray:
    tokens = text.split()
    vectors = [wv[token] for token in tokens if token in wv.key_to_index]
    if not vectors:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0).astype(np.float32)

review = "máy chạy êm tiết_kiệm điện rất hài_lòng"
x_vec = review_to_vector(review, wv).reshape(1, -1)

# Nạp mô hình SVM-WE (Pipeline gồm StandardScaler + RBF SVM)
model = joblib.load("results/svm-we.pkl")
y_pred = model.predict(x_vec)

print("Kết quả dự đoán:", "SPAM" if y_pred[0] == 1 else "HAM")
```

---

## 🛠️ Quy Trình Tái Tạo Tệp Đặc Trưng (Reproducibility)

Để tạo mới hoặc cập nhật toàn bộ kho tệp đặc trưng trong thư mục này, thực thi lần lượt các lệnh sau từ **thư mục gốc của dự án**:

```bash
# 1. Trích chọn đặc trưng Bag-of-Words
python feature_extraction/bow.py

# 2. Trích chọn đặc trưng TF-IDF
python feature_extraction/tfidf.py

# 3. Huấn luyện Word2Vec và trích chọn Word Embedding
python feature_extraction/WE.py
```

*Điều kiện tiên quyết:* Tệp dữ liệu sạch `data/processed/new_processed.csv` đã được tạo thành công từ pipeline tiền xử lý và chứa đầy đủ 3 trường `noi_dung`, `label`, `split`.

---

## ⚠️ Các Lưu Ý Kỹ Thuật Quan Trọng

1. **Xử lý giá trị âm của Word2Vec với Naive Bayes:**
   - Thuật toán `MultinomialNB` chỉ chấp nhận các giá trị không âm ($x \ge 0$). Do vector Word2Vec có giá trị thực âm, mô hình `model/nb-we.py` bắt buộc phải đặt trong một `Pipeline` với `MinMaxScaler(feature_range=(0, 1))` được `fit` trên tập Train trước khi đưa vào Naive Bayes.
2. **Bảo toàn định dạng ma trận thưa (Sparse Matrix):**
   - Tuyệt đối không chuyển đổi `X_train` và `X_test` từ `scipy.sparse.csr_matrix` sang ma trận dày bằng `.toarray()` khi huấn luyện Linear SVM hoặc Naive Bayes. Ma trận thưa giúp tiết kiệm hơn 99% bộ nhớ RAM và tăng tốc độ hội tụ của thuật toán tối ưu.
3. **Hiện tượng OOV (Out-Of-Vocabulary):**
   - Trong tập Test, có **3 câu** chỉ chứa các từ không xuất hiện trong từ điển train của BoW/TF-IDF và Word2Vec (sinh ra vector toàn số 0). Các mô hình phân loại vẫn xử lý an toàn nhờ vào xác suất tiên nghiệm (Prior probability) hoặc hệ số chặn (Bias/Intercept).
