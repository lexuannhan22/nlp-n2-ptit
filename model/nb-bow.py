"""
MODEL: Naive Bayes + Bag of Words

Multinomial Naive Bayes: lựa chọn kinh điển cho đặc trưng dạng ĐẾM (BoW).

    P(spam | câu) ∝ P(spam) · Π_t P(t | spam)^count(t)

  P(t | spam) = (số lần t xuất hiện trong các câu spam + alpha) / (tổng token spam + alpha·|V|)
  alpha = hệ số làm trơn Laplace: tránh xác suất 0 khi gặp từ chưa thấy trong lớp.

Siêu tham số dò: alpha.

Chạy (đứng trong thư mục BTL1, sau khi đã chạy preprocess.py và bow.py):
    python 3_model/nb-bow.py
"""
from sklearn.naive_bayes import MultinomialNB

from evaluate import run_experiment

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]   # các giá trị alpha sẽ thử

if __name__ == "__main__":
    run_experiment(
        model_name="nb-bow",
        feature="bow",
        estimator=MultinomialNB(),
        param_grid={"alpha": ALPHAS},
        note="MultinomialNB trên vector đếm từ (BoW)",
        balance_classes=True,
    )
