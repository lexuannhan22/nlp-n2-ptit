"""Cài thư viện: pip install pandas emoji underthesea
Chạy: python tien_xu_ly_data.py
Đặt CSV và script cùng thư mục rồi chạy lệnh trong thư mục đó.
"""

import html
import re
import unicodedata

import emoji
import pandas as pd
from underthesea import word_tokenize


FILE_INPUT = "C:/Users/Admin/nlp-n2-ptit/data/labeled/raw_7k2_deepseek_labeled.csv"
FILE_OUTPUT = "C:/Users/Admin/nlp-n2-ptit/data/processed/new_processed.csv"

TEENCODE = {
    "ko": "không", "k": "không", "hok": "không", "hông": "không",
    "dc": "được", "đc": "được", "sp": "sản phẩm", "nv": "nhân viên",
    "dmx": "điện máy xanh", "ib": "inbox", "oke": "ok",
}

# Các token này được giữ lại khi xóa dấu câu.
TOKENS = {"urltoken", "emailtoken", "phonetoken", "numtoken",
          "emojitoken", "symboltoken"}
TOKEN_PATTERN = r"\b(urltoken|emailtoken|phonetoken|numtoken|emojitoken|symboltoken)\b"


def doc_du_lieu(duong_dan):
    """Đọc CSV và lấy hai cột cần dùng, không thay đổi nhãn hay lọc dòng."""
    df = pd.read_csv(duong_dan, encoding="utf-8-sig", dtype=str,
                     keep_default_na=False)
    df = df[["noi_dung", "label"]].copy()
    print("Số dòng đọc được:", len(df))
    return df


def chuan_hoa_van_ban(text):
    """Chuẩn hóa Unicode, chuyển chữ thường và bỏ HTML, khoảng trắng thừa."""
    text = unicodedata.normalize("NFC", text).lower()
    text = html.unescape(text)
    text = re.sub(r"<!--[\s\S]*?-->|</?[a-z][^>]*>", " ", text)
    return " ".join(text.split())


def thay_thong_tin(text):
    """Thay email, URL, số điện thoại, số và emoji bằng token chung."""
    # Xử lý email trước URL và số điện thoại trước số thường để tránh nhận nhầm.
    text = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " emailtoken ", text)
    text = re.sub(
        r"(?:https?://|www\.)[^\s<>]+|"
        r"\b[\w-]+(?:\.[\w-]+)*\.(?:com|vn|net|org|info|io|me|ly|xyz)"
        r"(?![\w-])(?:/[^\s<>]*)?",
        " urltoken ", text,
    )
    text = re.sub(
        r"(?<!\w)(?:\+?84|0)(?:[ .-]*\d){9}(?!\w)", " phonetoken ", text
    )
    # Xử lý emoji trước số để nhận diện nguyên vẹn emoji như 1️⃣.
    text = emoji.replace_emoji(text, replace=" emojitoken ")
    text = re.sub(r"\d+(?:[.,]\d+)*", " numtoken ", text)
    return " ".join(text.split())


def bo_dau_cau_va_sua_teencode(text):
    """Bỏ dấu câu và thay các từ viết tắt có trong bảng TEENCODE."""
    text_goc = text
    text = re.sub(r"[^\w\s]|_", " ", text)
    ket_qua = []
    for tu in text.split():
        ket_qua.append(TEENCODE.get(tu, tu))
    text = " ".join(ket_qua)

    # Nếu bình luận chỉ có dấu câu hoặc ký hiệu, giữ lại một token báo hiệu.
    if text == "" and text_goc.strip() != "":
        text = "symboltoken"
    return text


def tach_tu(text):
    """Tách từ tiếng Việt và giữ nguyên các token đặc biệt."""
    ket_qua = []
    for phan in re.split(TOKEN_PATTERN, text):
        if phan in TOKENS:
            ket_qua.append(phan)
        elif phan.strip():
            ket_qua.append(word_tokenize(phan, format="text"))
    return " ".join(" ".join(ket_qua).split())


def loai_bo_dong_trung(df):
    """Giữ lại dòng đầu tiên của mỗi cặp nội dung và nhãn trùng nhau."""
    df = df.drop_duplicates(subset=["noi_dung", "label"], keep="first")
    print("Sau bỏ dòng trùng:", len(df), "dòng")
    return df.reset_index(drop=True)


def main():
    # Đọc dữ liệu rồi lần lượt xử lý nội dung.
    df = doc_du_lieu(FILE_INPUT)

    df["noi_dung"] = df["noi_dung"].apply(chuan_hoa_van_ban)
    print("Sau chuẩn hóa cơ bản:", len(df), "dòng")
    df["noi_dung"] = df["noi_dung"].apply(thay_thong_tin)
    print("Sau thay thông tin bằng token:", len(df), "dòng")
    df["noi_dung"] = df["noi_dung"].apply(bo_dau_cau_va_sua_teencode)
    print("Sau bỏ dấu câu và sửa teencode:", len(df), "dòng")
    df["noi_dung"] = df["noi_dung"].apply(tach_tu)
    print("Sau tách từ:", len(df), "dòng")

    # Loại các dòng trùng trước khi lưu kết quả.
    df = loai_bo_dong_trung(df)
    df.to_csv(FILE_OUTPUT, index=False, encoding="utf-8-sig")
    print("Đã lưu:", FILE_OUTPUT)
    print("Số mẫu mỗi nhãn:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
