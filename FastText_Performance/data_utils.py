"""Đọc dữ liệu FastText; dùng split có sẵn nếu CSV đã được chia."""
import re
import unicodedata

import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = unicodedata.normalize("NFC", str(text)).lower()
    text = text.replace("__label__", " ")
    text = re.sub(r"([.,!?;:()\"/])", r" \1 ", text)
    # Giữ dấu tiếng Việt, số, emoji và nhận xét ngắn.
    return re.sub(r"\s+", " ", text).strip() or "empty_review_token"


def load_data(csv_path):
    df = pd.read_csv(csv_path, keep_default_na=False)
    text_column = "noi_dung" if "noi_dung" in df.columns else "review"
    if text_column not in df.columns or "label" not in df.columns:
        raise ValueError("CSV cần có cột noi_dung (hoặc review) và label.")

    df["label"] = pd.to_numeric(df["label"], errors="raise")
    if not df["label"].isin([0, 1]).all():
        raise ValueError("Nhãn phải là 0 (ham) hoặc 1 (spam).")
    df["label"] = df["label"].astype(int)
    df["text"] = df[text_column].apply(clean_text)

    # Tách câu có nhãn mâu thuẫn, không tự quyết định sửa nhãn.
    conflict = df.groupby("text")["label"].transform("nunique") > 1
    df = df.loc[~conflict].drop_duplicates("text")

    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 5:
        raise ValueError("Cần ít nhất 5 câu riêng biệt mỗi lớp sau xử lý.")

    # Nếu CSV đã có split, giữ nguyên cách chia dùng chung với BoW/TF-IDF/WE.
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.strip().str.lower()
        if not df["split"].isin(["train", "test"]).all():
            raise ValueError("Cột split chỉ được chứa 'train' hoặc 'test'.")
        train = df[df["split"] == "train"].copy()
        test = df[df["split"] == "test"].copy()
        if train.empty or test.empty:
            raise ValueError("Cột split phải có cả dữ liệu train và test.")
        return train, test

    # CSV chưa có split thì tự chia ngẫu nhiên, có phân tầng theo nhãn.
    return train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )
