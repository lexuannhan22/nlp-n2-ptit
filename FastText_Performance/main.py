"""Chạy: python main.py [đường_dẫn_CSV]. Chỉ in performance ra terminal."""
import argparse
import sys
from pathlib import Path
from time import perf_counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from data_utils import load_data
from fasttext_model import predict_test, train_model


def main():
    # Tránh lỗi in tiếng Việt trên terminal Windows dùng bảng mã cũ.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Đánh giá FastText spam/ham")
    parser.add_argument(
        "csv", nargs="?", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed" / "new_processed.csv",
    )
    args = parser.parse_args()

    train, test = load_data(args.csv)
    start = perf_counter()
    model = train_model(train)
    train_seconds = perf_counter() - start

    y_true = test["label"]
    y_pred = predict_test(model, test["text"])

    print(f"Train: {len(train)} câu | Test: {len(test)} câu")
    print(f"Spam trong test: {int((y_true == 1).sum())} câu")
    print(f"Thời gian huấn luyện: {train_seconds:.2f} giây")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro-F1 : {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\nPrecision, Recall, F1-score:")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1], target_names=["ham", "spam"],
        digits=4, zero_division=0,
    ))
    print("Confusion matrix (hàng: nhãn thật; cột: dự đoán; thứ tự ham, spam):")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))


if __name__ == "__main__":
    main()
