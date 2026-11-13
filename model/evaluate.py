"""
Hàm dùng chung cho 6 file model: nạp đặc trưng, tìm siêu tham số, huấn luyện,
đánh giá trên tập test và IN RA PERFORMANCE.

Mỗi file model (nb-bow.py, svm-tfidf.py, ...) chỉ cần khai báo mô hình + lưới
siêu tham số rồi gọi run_experiment(...).

Quy trình trong run_experiment:
  1. Nạp features/<feature>.pkl (X_train, X_test, y_train, y_test).
  2. GridSearchCV với 5-fold Stratified CV CHỈ trên tập train, tiêu chí F1 của lớp spam
     -> chọn siêu tham số mà không "nhìn trộm" tập test.
  3. Huấn luyện lại mô hình tốt nhất trên toàn bộ train, dự đoán tập test.
  4. In: accuracy, precision / recall / F1 (spam), F1 macro, ROC-AUC, PR-AUC,
     classification report, ma trận nhầm lẫn, thời gian train/predict, ví dụ đoán sai.
  5. Lưu vào results/: <model>.json, <model>_confusion_matrix.png, <model>_errors.csv,
     và cập nhật bảng so sánh results/summary.csv.
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
FEAT_DIR = ROOT / "feature_extraction" / "features"
RESULT_DIR = ROOT / "results"
SEED = 42


def load_features(feature: str):
    path = FEAT_DIR / f"{feature}.pkl"
    if not path.exists():
        raise SystemExit(f"Không thấy {path}. Hãy chạy 2_trich_chon_dac_trung/"
                         f"{'WE' if feature == 'we' else feature}.py trước.")
    return joblib.load(path)


def _scores(model, X):
    """Điểm liên tục cho lớp spam (để tính ROC-AUC / PR-AUC)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)   # SVM không có xác suất -> dùng khoảng cách tới siêu phẳng


def _print_confusion(cm):
    (tn, fp), (fn, tp) = cm
    print("  Ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán):")
    print("                 ham    spam")
    print(f"    thực tế ham  {tn:5d}   {fp:5d}    <- {fp} ham bị báo nhầm là spam (FP)")
    print(f"    thực tế spam {fn:5d}   {tp:5d}    <- {fn} spam bị bỏ sót (FN)")


def _save_confusion_png(cm, title, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["ham", "spam"])
    ax.set_yticks([0, 1], ["ham", "spam"])
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _update_summary(row: dict):
    path = RESULT_DIR / "summary.csv"
    df = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(df):
        df = df[df["model"] != row["model"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values("f1_spam", ascending=False)
    df.to_csv(path, index=False)
    return df


def run_experiment(
    model_name: str,
    feature: str,
    estimator,
    param_grid: dict,
    note: str = "",
    balance_classes: bool = False,
):
    print("=" * 70)
    print(f" MODEL: {model_name}   |   ĐẶC TRƯNG: {feature.upper()}")
    if note:
        print(f" {note}")
    print("=" * 70)

    data = load_features(feature)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]
    print(f"Train: {X_train.shape[0]:,} mẫu | Test: {X_test.shape[0]:,} mẫu | "
          f"{X_train.shape[1]:,} đặc trưng | tỉ lệ spam (test): {y_test.mean():.1%}")

    # ---- 1. Tìm siêu tham số bằng cross-validation trên TRAIN ------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(estimator, param_grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
    fit_params = {}
    if balance_classes:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        if hasattr(estimator, "steps"):
            ten_buoc_cuoi = estimator.steps[-1][0]
            fit_params[f"{ten_buoc_cuoi}__sample_weight"] = sample_weight
        else:
            fit_params["sample_weight"] = sample_weight

    t0 = time.perf_counter()
    search.fit(X_train, y_train, **fit_params)
    search_time = time.perf_counter() - t0
    print(f"\nGridSearchCV 5-fold ({len(search.cv_results_['params'])} tổ hợp) "
          f"trong {search_time:.1f}s")
    print(f"  Siêu tham số tốt nhất: {search.best_params_}")
    print(f"  F1 (spam) trung bình CV trên train: {search.best_score_:.4f} "
          f"± {search.cv_results_['std_test_score'][search.best_index_]:.4f}")

    # ---- 2. Huấn luyện lại mô hình tốt nhất (đo thời gian) ---------------
    model = search.best_estimator_
    t0 = time.perf_counter()
    model.fit(X_train, y_train, **fit_params)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0
    scores = _scores(model, X_test)

    # ---- 3. Performance trên TEST ------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics = {
        "model": model_name,
        "feature": feature,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_spam": precision_score(y_test, y_pred, zero_division=0),
        "recall_spam": recall_score(y_test, y_pred, zero_division=0),
        "f1_spam": f1_score(y_test, y_pred, zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "pr_auc": average_precision_score(y_test, scores),
        "cv_f1_spam": search.best_score_,
        "train_time_s": train_time,
        "predict_time_s": predict_time,
        "best_params": json.dumps(search.best_params_, ensure_ascii=False, default=str),
    }

    print("\n" + "-" * 70)
    print(f" PERFORMANCE TRÊN TẬP TEST — {model_name}")
    print("-" * 70)
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  Precision (spam)  : {metrics['precision_spam']:.4f}")
    print(f"  Recall    (spam)  : {metrics['recall_spam']:.4f}")
    print(f"  F1-score  (spam)  : {metrics['f1_spam']:.4f}")
    print(f"  F1-score  (macro) : {metrics['f1_macro']:.4f}")
    print(f"  ROC-AUC           : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC            : {metrics['pr_auc']:.4f}")
    print(f"  Thời gian train   : {train_time * 1000:.1f} ms")
    print(f"  Thời gian predict : {predict_time * 1000:.1f} ms "
          f"({predict_time / len(y_test) * 1e6:.1f} µs/mẫu)")
    print()
    _print_confusion(cm)
    print("\nClassification report:")
    print(classification_report(
        y_test, y_pred,
        labels=[0, 1], target_names=["ham", "spam"],
        digits=4, zero_division=0,
    ))

    # ---- 4. Ví dụ đoán sai ------------------------------------------------
    texts = np.array(data.get("test_text", [""] * len(y_test)), dtype=object)
    err = pd.DataFrame({"review": texts, "label": y_test, "pred": y_pred, "score": scores})
    err = err[err.label != err.pred]
    fp = err[err.label == 0].sort_values("score", ascending=False)
    fn = err[err.label == 1].sort_values("score")
    print("Ví dụ HAM bị báo nhầm là SPAM:")
    for t in fp.review.head(5):
        print(f"   - {str(t)[:90]!r}")
    print("Ví dụ SPAM bị bỏ sót:")
    for t in fn.review.head(5):
        print(f"   - {str(t)[:90]!r}")

    # ---- 5. Lưu kết quả ---------------------------------------------------
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_DIR / f"{model_name}.json", "w", encoding="utf-8") as f:
        json.dump({**metrics, "confusion_matrix": cm.tolist()}, f, ensure_ascii=False, indent=2)
    err.to_csv(RESULT_DIR / f"{model_name}_errors.csv", index=False, encoding="utf-8-sig")
    _save_confusion_png(cm, model_name, RESULT_DIR / f"{model_name}_confusion_matrix.png")
    joblib.dump(model, RESULT_DIR / f"{model_name}.pkl")

    summary = _update_summary(metrics)
    print(f"\n✔ Kết quả đã lưu trong {RESULT_DIR}")
    print("\nBẢNG SO SÁNH CÁC MODEL ĐÃ CHẠY (results/summary.csv, sắp theo F1 spam):")
    cols = ["model", "accuracy", "precision_spam", "recall_spam", "f1_spam", "f1_macro", "roc_auc"]
    print(summary[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return model, metrics
