# `results/`

Thư mục này chứa **toàn bộ kết quả thực nghiệm** của 6 mô hình phân loại bình luận Spam / Ham (Naive Bayes và SVM, kết hợp với BoW, TF-IDF và Word2Vec). Các file được **sinh tự động** bởi hàm `run_experiment()` trong `model/evaluate.py` mỗi khi chạy một script mô hình; không chỉnh sửa thủ công.

> Quy ước nhãn: `0` = Ham (bình luận hợp lệ), `1` = Spam. Các chỉ số "spam" được tính trên lớp `1`.

---

## 1. Nội dung thư mục

```text
results/
├── summary.csv                    # Bảng tổng hợp tất cả mô hình, sắp theo F1 (spam) giảm dần
├── <model>.json                   # Chỉ số chi tiết + ma trận nhầm lẫn + siêu tham số tốt nhất
├── <model>.pkl                    # Mô hình tốt nhất đã huấn luyện (joblib)
├── <model>_confusion_matrix.png   # Ảnh ma trận nhầm lẫn trên tập test
├── <model>_errors.csv             # Các câu test bị dự đoán sai (để phân tích lỗi)
└── README.md
```

`<model>` nhận một trong 6 giá trị:

| `<model>` | Mô hình | Đặc trưng | Script |
| :--- | :--- | :--- | :--- |
| `nb-bow` | Multinomial Naive Bayes | Bag-of-Words | `model/nb-bow.py` |
| `nb-tfidf` | Multinomial Naive Bayes | TF-IDF | `model/nb-tfidf.py` |
| `nb-we` | Multinomial Naive Bayes (+ `MinMaxScaler`) | Word2Vec | `model/nb-we.py` |
| `svm-bow` | Linear SVM (`LinearSVC`) | Bag-of-Words | `model/svm-bow.py` |
| `svm-tfidf` | Linear SVM (`LinearSVC`) | TF-IDF | `model/svm-tfidf.py` |
| `svm-we` | SVM nhân RBF (+ `StandardScaler`) | Word2Vec | `model/svm-we.py` |

---

## 2. Cách tạo lại kết quả

Chạy từ thư mục gốc của dự án, sau khi đã chạy tiền xử lý và trích chọn đặc trưng tương ứng (xem `feature_extraction/features/README.md`):

```bash
python model/nb-bow.py
python model/nb-tfidf.py
python model/nb-we.py
python model/svm-bow.py
python model/svm-tfidf.py
python model/svm-we.py
```

Mỗi lần chạy:
1. Dò siêu tham số bằng `GridSearchCV` (5-fold Stratified CV, tiêu chí **F1 lớp spam**) **chỉ trên tập train**.
2. Huấn luyện lại mô hình tốt nhất trên toàn bộ train, dự đoán trên tập test.
3. In chỉ số, ma trận nhầm lẫn, một số câu dự đoán sai ra terminal.
4. Ghi/ghi đè các file `<model>.*` và cập nhật dòng tương ứng trong `summary.csv` (mỗi mô hình một dòng; chạy lại sẽ thay dòng cũ).

---

## 3. Mô tả từng loại file

### `summary.csv`

Mỗi dòng là một mô hình, sắp theo `f1_spam` giảm dần. Các cột:

| Cột | Ý nghĩa |
| :--- | :--- |
| `model`, `feature` | Tên mô hình và loại đặc trưng |
| `accuracy` | Độ chính xác tổng thể trên tập test |
| `precision_spam` | Trong số câu bị dự đoán là spam, bao nhiêu phần trăm đúng là spam |
| `recall_spam` | Trong số câu spam thật, bao nhiêu phần trăm được phát hiện |
| `f1_spam` | F1-score của lớp spam (tiêu chí chính) |
| `f1_macro` | F1 trung bình của hai lớp (ham, spam) |
| `roc_auc`, `pr_auc` | Diện tích dưới đường ROC / Precision-Recall (tính từ điểm liên tục của mô hình) |
| `cv_f1_spam` | F1 spam trung bình của cross-validation trên tập train (với siêu tham số tốt nhất) |
| `train_time_s`, `predict_time_s` | Thời gian huấn luyện lại / dự đoán trên test (giây) |
| `best_params` | Siêu tham số tốt nhất tìm được (chuỗi JSON) |

### `<model>.json`

Chứa đầy đủ các cột trên của mô hình đó, cộng thêm khóa `confusion_matrix` dạng `[[TN, FP], [FN, TP]]` (hàng = thực tế, cột = dự đoán, thứ tự nhãn `[0, 1]`).

### `<model>_errors.csv`

Danh sách các mẫu test bị đoán sai, gồm các cột `review` (văn bản), `label` (nhãn thật), `pred` (nhãn dự đoán), `score` (điểm liên tục của mô hình cho lớp spam: xác suất với Naive Bayes, khoảng cách tới siêu phẳng với SVM).

- `label = 0, pred = 1`: **False Positive** — ham bị báo nhầm là spam.
- `label = 1, pred = 0`: **False Negative** — spam bị bỏ sót.

Đây là nguồn chính để phân tích lỗi (ví dụ câu quá ngắn, teencode, spam ngụy trang như đánh giá thật).

### `<model>_confusion_matrix.png`

Ảnh ma trận nhầm lẫn 2x2 của mô hình trên tập test. Chỉ được tạo khi đã cài `matplotlib`.

### `<model>.pkl`

Mô hình tốt nhất đã huấn luyện, lưu bằng `joblib`. Với các mô hình dùng Word2Vec, đây là một `Pipeline` gồm bộ scaler và bộ phân loại.

```python
import joblib

model = joblib.load("results/svm-we.pkl")
# X_new phải được tạo bằng đúng bộ đặc trưng/vectorizer đã dùng khi huấn luyện
y_pred = model.predict(X_new)
```

---

## 4. Kết quả tham khảo

Theo `README.md` gốc của dự án, trên tập test (964 mẫu: 758 Ham, 206 Spam, tỉ lệ spam ~21,4%):

| Mô hình | Đặc trưng | Accuracy | Precision (Spam) | Recall (Spam) | F1 (Spam) | F1 Macro | ROC-AUC | PR-AUC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| SVM (RBF) | Word2Vec | 95,64% | 91,41% | 87,86% | 89,60% | 93,42% | 0,9779 | 0,9340 |
| SVM (Linear) | BoW | 95,02% | 84,35% | 94,17% | 88,99% | 92,89% | 0,9772 | 0,9078 |
| SVM (Linear) | TF-IDF | 94,61% | 85,98% | 89,32% | 87,62% | 92,09% | 0,9737 | 0,8950 |
| Naive Bayes | TF-IDF | 93,98% | 84,58% | 87,86% | 86,19% | 91,17% | 0,9747 | 0,9212 |
| Multinomial NB | BoW | 93,98% | 87,76% | 83,50% | 85,57% | 90,89% | 0,9739 | 0,9125 |
| Naive Bayes | Word2Vec | 80,91% | 54,17% | 69,42% | 60,85% | 74,12% | 0,8754 | 0,5862 |

> Số liệu trên lấy từ `README.md` gốc. Nguồn chính xác nhất luôn là `summary.csv` sau khi chạy lại; hãy đối chiếu nếu dữ liệu hoặc siêu tham số đã thay đổi.

**Nhận xét nhanh**
- **SVM RBF + Word2Vec** có F1 spam, precision, accuracy và PR-AUC cao nhất.
- **SVM tuyến tính + BoW** có recall spam cao nhất (94,17%): ít bỏ sót spam nhất, đổi lại precision thấp hơn.
- **Naive Bayes + Word2Vec** kém rõ rệt, vì Naive Bayes giả định đặc trưng độc lập và phù hợp hơn với vector đếm/trọng số thưa.

---

## 5. Lưu ý khi đọc kết quả

1. **Dữ liệu lệch lớp** (~21,4% spam): ưu tiên **F1 spam** và **PR-AUC** hơn Accuracy.
2. **Siêu tham số chọn bằng CV trên train**, tập test chỉ dùng để báo cáo cuối, nên không bị "nhìn trộm".
3. **Thời gian train/predict** chỉ mang tính tham khảo, phụ thuộc máy chạy.
4. **FastText** là pipeline độc lập (`FastText_Performance/`) và **không** ghi kết quả vào thư mục này.
5. Các file `.pkl`, `.png`, `.csv`, `.json` đều tái tạo được bằng script; cân nhắc `.gitignore` các file nặng như `.pkl` nếu không cần commit.