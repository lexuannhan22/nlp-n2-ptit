"""
MODEL: Naive Bayes + TF-IDF

Multinomial Naive Bayes: lựa chọn phù hợp cho đặc trưng TF-IDF (số thực
không âm), coi TF-IDF như "số đếm có trọng số".

    P(spam | câu) ∝ P(spam) · Π_t P(t | spam)^tfidf(t)

  P(t | spam) = (tổng tfidf(t) trong câu spam + alpha) / (tổng tfidf spam + alpha·|V|)
  alpha = hệ số làm trơn Laplace: tránh xác suất 0 khi gặp từ chưa thấy trong lớp.

Siêu tham số dò: alpha.

Chạy (đứng trong thư mục BTL1, sau khi đã chạy preprocess.py và tfidf.py):
    python 3_model/nb-tfidf.py
"""
from sklearn.naive_bayes import MultinomialNB

from evaluate import run_experiment

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]

if __name__ == "__main__":
    run_experiment(
        model_name="nb-tfidf",
        feature="tfidf",
        estimator=MultinomialNB(),
        param_grid={"alpha": ALPHAS},
        note="MultinomialNB trên vector TF-IDF",
        balance_classes=True,
    )
