"""
MODEL: Naive Bayes + Word Embedding

Vector WE là vector DÀY, giá trị thực có thể ÂM -> MultinomialNB yêu cầu
đặc trưng không âm, nên cần dịch chuyển (shift) toàn bộ vector về miền [0, +∞)
trước khi đưa vào mô hình.

Dùng MinMaxScaler(feature_range=(0, 1)) để:
  - Fit trên tập TRAIN (học min/max của từng chiều từ train)
  - Transform cả train lẫn test theo cùng thang đo -> không rò rỉ thông tin test
  - Đảm bảo mọi giá trị >= 0, tương thích với MultinomialNB

    P(spam | x') ∝ P(spam) · Π_j P(x'_j | spam)
    với x'_j = (x_j - min_j) / (max_j - min_j)  (fit từ tập train)

Siêu tham số dò: alpha.

Chạy (đứng trong thư mục BTL1, sau khi đã chạy preprocess.py và WE.py):
    python 3_model/nb-we.py
"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from evaluate import run_experiment

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

# Pipeline: scale về [0,1] trước (fit trên train) -> MultinomialNB sau
estimator = Pipeline([
    ("scaler", MinMaxScaler()),
    ("nb",     MultinomialNB()),
])

if __name__ == "__main__":
    run_experiment(
        model_name="nb-we",
        feature="we",
        estimator=estimator,
        param_grid={"nb__alpha": ALPHAS},
        note="MultinomialNB trên vector Word2Vec trung bình (MinMaxScaler về [0,1] trước)",
        balance_classes=True,
    )
