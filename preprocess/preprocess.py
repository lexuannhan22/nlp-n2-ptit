"""
BƯỚC 1 — TIỀN XỬ LÍ DỮ LIỆU
===========================

Đầu vào : data/labeled/reviews_spam_ham.csv
           (các cột gốc: phan_loai, san_pham, nguoi_danh_gia, so_sao, noi_dung, link, label)
Đầu ra  : data/processed/processed.csv
           (các cột: category, product, review, rating, label, clean_text, split)

Pipeline gồm 5 bước:
  1. load_data()            — Đọc CSV, chuẩn hóa tên cột, loại dòng thiếu dữ liệu.
  2. clean_text()           — Làm sạch từng bình luận (Unicode, chữ thường, token đặc biệt,
                              rút gọn ký tự lặp, tách dấu câu, chuẩn hóa teencode).
  3. resolve_conflicts()    — Giải quyết câu có nhãn mâu thuẫn bằng cách lấy nhãn đa số.
  4. deduplicate()          — Loại bỏ bản ghi trùng lặp (dựa trên clean_text).
  5. split_dataset()        — Chia train/test phân tầng theo nhãn (mặc định 80/20).

Lí do bỏ trùng TRƯỚC khi chia:
  Bộ dữ liệu có ~56% dòng trùng lặp ("Tốt" ~450 lần, "Ok" ~340 lần...).
  Nếu chia trước rồi mới lọc, cùng một câu sẽ nằm ở cả train lẫn test
  → mô hình chỉ cần "học thuộc" là đoán đúng (data leakage).

Nhãn: label = 1 → spam,  label = 0 → ham.

Chạy: python preprocess/preprocess.py [--input ...] [--output ...] [--test-size 0.2] [--seed 42]
"""

import re
import unicodedata
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Đường dẫn mặc định
# ---------------------------------------------------------------------------

ROOT     = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "labeled" / "1000_sample_dau_may_gan_nhan.csv"
OUT_PATH = ROOT / "data" / "processed" / "processed.csv"

# ---------------------------------------------------------------------------
# Hằng số / regex dùng trong làm sạch văn bản
# ---------------------------------------------------------------------------

# Teencode / viết tắt hay gặp trong bình luận thương mại điện tử tiếng Việt
TEENCODE: dict[str, str] = {
    "ko": "không", "k": "không", "kh": "không", "khong": "không",
    "hok": "không", "hông": "không", "kg": "không", "khg": "không",
    "dc": "được", "đc": "được", "duoc": "được", "dk": "được", "đk": "được",
    "sp": "sản phẩm",
    "oke": "ok", "okie": "ok", "okela": "ok", "oki": "ok",
    "okay": "ok", "okê": "ok",
    "nv": "nhân viên", "dmx": "điện máy xanh", "bh": "bảo hành",
    "j": "gì", "mk": "mình", "mik": "mình", "mn": "mọi người",
    "vs": "với", "r": "rồi", "rùi": "rồi",
    "tks": "cảm ơn", "thanks": "cảm ơn", "thank": "cảm ơn",
    "ib": "inbox", "inbox": "liên hệ",
    "ship": "giao hàng", "shipper": "người giao hàng",
    "sd": "sử dụng", "sdung": "sử dụng",
    "chat": "nhắn tin", "msg": "tin nhắn",
    "ad": "admin", "mod": "quản trị viên",
    "cmt": "bình luận", "rep": "trả lời",
    "mua": "mua", "ban": "bán", "hang": "hàng",
    "ntn": "như thế nào", "sao": "sao", "lm": "làm",
    "cx": "cũng", "cx": "cũng", "vd": "ví dụ",
    "nc": "nói chuyện", "ny": "người yêu",
    "dt": "điện thoại", "đt": "điện thoại",
    "bth": "bình thường", "bt": "bình thường",
}

# --- Biểu thức chính quy ---
URL_RE = re.compile(
    r"(https?://\S+|www\.\S+|\b\S+\.(?:com|vn|net|vip|xyz|info|org|shop|top|online)\S*)",
    flags=re.IGNORECASE,
)
EMAIL_RE      = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE      = re.compile(r"(?:\+?84|0)(?:[\s.\-]?\d){8,10}")          # SĐT Việt Nam
NUMBER_RE     = re.compile(r"\d+(?:[.,]\d+)*(?:k|tr|đ|d|vnd|vnđ)?\b")   # 500k, 5tr, 30…
REPEAT_CHR_RE = re.compile(r"(\w)\1{2,}")                                # "tốttttt" → "tốt"
PUNCT_RE      = re.compile(r"([^\w\s])")                                  # tách dấu câu
MULTI_SPC_RE  = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 1. Đọc & chuẩn hóa dữ liệu
# ---------------------------------------------------------------------------

def load_data(path: str | Path) -> pd.DataFrame:
    """
    Đọc file CSV gốc, chuẩn hóa tên cột sang tiếng Anh, loại bỏ
    các dòng thiếu nội dung bình luận hoặc thiếu nhãn.

    Tên cột đầu ra: category, product, reviewer, rating, review, link, label

    Parameters
    ----------
    path : str | Path
        Đường dẫn tới file CSV gốc (có thể có BOM utf-8-sig).

    Returns
    -------
    pd.DataFrame
        DataFrame đã chuẩn hóa tên cột và loại dòng không hợp lệ.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[load]  Đọc dữ liệu gốc: {len(df):,} dòng, {df.shape[1]} cột")

    # Chuẩn hóa tên cột (hỗ trợ cả file gốc tiếng Việt lẫn file đã đổi tên)
    col_map = {
        "phan_loai":     "category",
        "san_pham":      "product",
        "nguoi_danh_gia":"reviewer",
        "so_sao":        "rating",
        "noi_dung":      "review",
        "link":          "link",
        "label":         "label",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Báo cáo giá trị thiếu
    missing = df.isna().sum()
    if missing.any():
        print(f"[load]  Giá trị thiếu:\n{missing[missing > 0].to_string()}")

    # Loại bỏ dòng không có nội dung hoặc không có nhãn
    n_before = len(df)
    df = df.dropna(subset=["review", "label"]).copy()
    df["label"] = df["label"].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[load]  Loại {n_dropped:,} dòng thiếu review/label → còn {len(df):,} dòng")

    ham_n  = int((df.label == 0).sum())
    spam_n = int((df.label == 1).sum())
    print(f"[load]  Phân bố nhãn: ham={ham_n:,}  spam={spam_n:,}  "
          f"(tỉ lệ spam {spam_n / len(df):.1%})")
    return df


# ---------------------------------------------------------------------------
# 2. Làm sạch từng bình luận
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Chuẩn hóa chuỗi Unicode về dạng NFC (đồng nhất hai kiểu gõ dấu tiếng Việt)
    và chuyển toàn bộ về chữ thường.
    """
    return unicodedata.normalize("NFC", text).lower()


def replace_special_tokens(text: str) -> str:
    """
    Thay thế URL, email, số điện thoại và số tiền/số lượng bằng các token
    chung để mô hình học được tín hiệu "có link / có SĐT / có số"
    mà không học thuộc giá trị cụ thể.

    Thứ tự quan trọng: URL → email → phone → number.
    """
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    text = PHONE_RE.sub(" phonetoken ", text)
    text = NUMBER_RE.sub(" numtoken ", text)
    return text


def normalize_characters(text: str) -> str:
    """
    - Rút gọn ký tự lặp quá 2 lần ("tốttttt" → "tốt").
    - Tách dấu câu ra thành token riêng ("ok." → "ok .") để giữ lại
      tín hiệu spam từ các bình luận chỉ có "....." hay "!!!".
    """
    text = REPEAT_CHR_RE.sub(r"\1", text)
    text = PUNCT_RE.sub(r" \1 ", text)
    return text


def normalize_teencode(tokens: list[str]) -> list[str]:
    """
    Chuẩn hóa teencode / viết tắt bằng bảng TEENCODE.
    Trả về danh sách token sau khi tra bảng.
    """
    return [TEENCODE.get(tok, tok) for tok in tokens]


def clean_text(text) -> str:
    """
    Pipeline làm sạch hoàn chỉnh cho một bình luận:
      1. Chuẩn hóa Unicode + chữ thường
      2. Thay token đặc biệt (URL, email, phone, number)
      3. Rút gọn ký tự lặp + tách dấu câu
      4. Chuẩn hóa teencode
      5. Thu gọn khoảng trắng thừa

    Parameters
    ----------
    text : any
        Nội dung bình luận thô (chuỗi hoặc NaN).

    Returns
    -------
    str
        Chuỗi đã được làm sạch, các token cách nhau bởi dấu cách.
        Trả về "emptytoken" nếu kết quả rỗng.
    """
    # Xử lý giá trị không phải chuỗi
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = normalize_unicode(text)
    text = replace_special_tokens(text)
    text = normalize_characters(text)

    tokens = normalize_teencode(text.split())
    text   = MULTI_SPC_RE.sub(" ", " ".join(tokens)).strip()
    return text if text else "emptytoken"


def apply_clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Áp dụng hàm clean_text() lên toàn bộ cột 'review' và
    ghi kết quả vào cột 'clean_text'.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có cột 'review'.

    Returns
    -------
    pd.DataFrame
        DataFrame với cột 'clean_text' được thêm vào.
    """
    df = df.copy()
    df["clean_text"] = df["review"].map(clean_text)
    print(f"[clean] Đã làm sạch {len(df):,} bình luận")

    n_tok = df["clean_text"].str.split().str.len()
    print(f"[clean] Độ dài token sau làm sạch — "
          f"trung bình: {n_tok.mean():.1f}, "
          f"trung vị: {n_tok.median():.0f}, "
          f"lớn nhất: {n_tok.max()}")
    return df


# ---------------------------------------------------------------------------
# 3. Giải quyết nhãn mâu thuẫn
# ---------------------------------------------------------------------------

def resolve_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tìm các bình luận có cùng clean_text nhưng được gán khác nhãn,
    sau đó thống nhất nhãn theo đa số (majority vote).

    Trường hợp hòa (50/50) → giữ nhãn ham (0) theo nguyên tắc "nghi ngờ → ham".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có cột 'clean_text' và 'label'.

    Returns
    -------
    pd.DataFrame
        DataFrame với nhãn đã được giải quyết mâu thuẫn.
    """
    df = df.copy()
    n_unique_labels = df.groupby("clean_text")["label"].nunique()
    conflict_texts  = n_unique_labels[n_unique_labels > 1].index

    if len(conflict_texts) == 0:
        print("[conflict] Không có nhãn mâu thuẫn.")
        return df

    # Tính nhãn đa số cho từng câu xung đột (hòa → 0 = ham)
    majority_label = (
        df[df["clean_text"].isin(conflict_texts)]
        .groupby("clean_text")["label"]
        .agg(lambda s: s.value_counts().idxmax())
    )

    mask = df["clean_text"].isin(conflict_texts)
    df.loc[mask, "label"] = df.loc[mask, "clean_text"].map(majority_label)
    df["label"] = df["label"].astype(int)

    print(f"[conflict] {len(conflict_texts):,} câu có nhãn mâu thuẫn → đã thống nhất theo đa số")
    return df


# ---------------------------------------------------------------------------
# 4. Loại bỏ bản ghi trùng lặp
# ---------------------------------------------------------------------------

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các dòng có cùng clean_text (giữ dòng đầu tiên xuất hiện).

    Bắt buộc thực hiện TRƯỚC khi chia train/test để tránh data leakage:
    nếu chia trước, cùng một câu sẽ xuất hiện ở cả hai tập, làm điểm
    test bị thổi phồng do mô hình chỉ cần "học thuộc".

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có cột 'clean_text'.

    Returns
    -------
    pd.DataFrame
        DataFrame đã loại trùng và reset index.
    """
    n_before = len(df)
    df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"[dedup] {n_before:,} → {len(df):,} dòng "
          f"(loại {n_removed:,} = {n_removed / n_before:.1%} bản ghi trùng)")
    return df


# ---------------------------------------------------------------------------
# 5. Chia train / test
# ---------------------------------------------------------------------------

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Thêm cột 'split' với giá trị 'train' hoặc 'test' vào DataFrame,
    sử dụng stratified split theo cột 'label' để đảm bảo tỉ lệ
    spam/ham nhất quán ở cả hai tập.

    Chia một lần duy nhất tại bước tiền xử lý để BoW, TF-IDF và
    Word Embedding dùng chung cùng một phân chia → so sánh công bằng.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã làm sạch và loại trùng, có cột 'label'.
    test_size : float
        Tỉ lệ dữ liệu dành cho tập test (mặc định 0.2 = 20%).
    random_state : int
        Seed để tái lập kết quả (mặc định 42).

    Returns
    -------
    pd.DataFrame
        DataFrame với cột 'split' được thêm vào.
    """
    df = df.copy()
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state,
    )
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    print(f"[split] Chia train/test (test_size={test_size}, seed={random_state}):")
    for split_name in ["train", "test"]:
        part = df[df["split"] == split_name]
        ham_n  = int((part["label"] == 0).sum())
        spam_n = int((part["label"] == 1).sum())
        print(f"  {split_name:5s}: {len(part):5,} mẫu | "
              f"ham={ham_n:,}  spam={spam_n:,}  "
              f"(tỉ lệ spam {part['label'].mean():.1%})")
    return df


# ---------------------------------------------------------------------------
# 6. Lưu kết quả
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame, out_path: str | Path) -> None:
    """
    Lưu DataFrame đã xử lý ra file CSV với encoding utf-8-sig.

    Chỉ giữ lại các cột cần thiết theo thứ tự chuẩn:
        category, product, review, rating, label, clean_text, split

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầy đủ sau tất cả các bước xử lý.
    out_path : str | Path
        Đường dẫn file CSV đầu ra.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Chỉ xuất các cột tồn tại trong df (tránh lỗi khi cột tùy chọn vắng mặt)
    desired_cols = ["category", "product", "review", "rating", "label", "clean_text", "split"]
    output_cols  = [c for c in desired_cols if c in df.columns]

    df[output_cols].to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✔ Đã lưu dữ liệu sau tiền xử lí: {out_path}")
    print(f"  ({len(df):,} dòng × {len(output_cols)} cột: {', '.join(output_cols)})")


# ---------------------------------------------------------------------------
# 7. Hiển thị ví dụ trước / sau làm sạch
# ---------------------------------------------------------------------------

def show_examples(df: pd.DataFrame, n: int = 6, random_state: int = 1) -> None:
    """
    In ra một vài ví dụ so sánh bình luận gốc và bình luận sau làm sạch,
    phân loại theo nhãn spam/ham để kiểm tra trực quan.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame có cột 'review', 'clean_text', 'label'.
    n : int
        Số ví dụ hiển thị (mặc định 6).
    random_state : int
        Seed cho hàm sample (mặc định 1).
    """
    print(f"\n{'─'*70}")
    print(f"Ví dụ trước / sau làm sạch ({n} mẫu ngẫu nhiên):")
    print(f"{'─'*70}")
    sample = df.sample(min(n, len(df)), random_state=random_state)
    for _, row in sample.iterrows():
        label_str = "SPAM" if row["label"] == 1 else "HAM "
        print(f"  [{label_str}] gốc   : {str(row['review'])[:80]!r}")
        print(f"           sạch  : {row['clean_text'][:80]!r}")
        print()


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def run_pipeline(
    input_path: str | Path  = RAW_PATH,
    output_path: str | Path = OUT_PATH,
    test_size: float        = 0.2,
    seed: int               = 42,
    show_sample: bool       = True,
) -> pd.DataFrame:
    """
    Chạy toàn bộ pipeline tiền xử lý dữ liệu spam/ham:
      load → clean → resolve_conflicts → deduplicate → split → save

    Parameters
    ----------
    input_path : str | Path
        Đường dẫn file CSV đầu vào.
    output_path : str | Path
        Đường dẫn file CSV đầu ra.
    test_size : float
        Tỉ lệ tập test (mặc định 0.2).
    seed : int
        Random seed (mặc định 42).
    show_sample : bool
        Nếu True, in ví dụ trước/sau làm sạch ra terminal.

    Returns
    -------
    pd.DataFrame
        DataFrame đã xử lý hoàn chỉnh (bao gồm cột 'split').
    """
    print("=" * 70)
    print("  PIPELINE TIỀN XỬ LÍ DỮ LIỆU SPAM/HAM")
    print("=" * 70)

    # Bước 1 — Đọc dữ liệu
    df = load_data(input_path)

    # Bước 2 — Làm sạch văn bản
    df = apply_clean_text(df)

    # Bước 3 — Giải quyết nhãn mâu thuẫn
    df = resolve_label_conflicts(df)

    # Bước 4 — Loại bỏ bản ghi trùng lặp
    df = deduplicate(df)

    # Bước 5 — Chia train/test
    df = split_dataset(df, test_size=test_size, random_state=seed)

    # Lưu kết quả
    save_data(df, output_path)

    # Hiển thị ví dụ
    if show_sample:
        show_examples(df)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
