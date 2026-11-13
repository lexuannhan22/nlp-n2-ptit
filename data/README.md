# 📂 Dữ Liệu Dự Án — Phân Loại Review Spam/Ham Điện Máy Xanh (Kho 7.200 Mẫu)

Tài liệu này cung cấp **thống kê chi tiết, cấu trúc dữ liệu và quy chuẩn kỹ thuật** cho kho dữ liệu chính thức gồm **7.200 đánh giá** được thu thập từ website [Điện máy XANH](https://www.dienmayxanh.com) phục vụ bài toán phát hiện đánh giá rác (Spam) và đánh giá thật (Ham).

Toàn bộ dữ liệu được quản lý theo luồng xử lý 3 giai đoạn:
$$\text{Thu thập thô (Raw: 7.200)} \longrightarrow \text{Gán nhãn tự động & QC (Labeled: 7.200)} \longrightarrow \text{Tiền xử lý & Khử trùng (Processed: 5.347)}$$

---

## 🗂️ Cấu Trúc Thư Mục Dữ Liệu

```text
data/
├── raw/
│   └── raw_7k2.csv                                # 7.200 đánh giá thô thu thập từ web (1,76 MB)
│
├── labeled/
│   ├── raw_7k2_deepseek_labeled.csv               # 7.200 đánh giá đã gán nhãn DeepSeek (1,77 MB)
│   ├── raw_7k2_deepseek_labeled.csv.checkpoint.jsonl # Checkpoint chi tiết: mã Rule, lý do gán từng dòng (1,27 MB)
│   ├── raw_7k2_deepseek_test20.csv                # Lô chạy thử nghiệm 20 mẫu (3,56 KB)
│   ├── 1000_sample_dau_tu_gan_nhan.csv            # Lô 1.000 mẫu người gán thủ công (Ground Truth QC - 272 KB)
│   ├── 1000_sample_dau_may_gan_nhan.csv           # Lô 1.000 mẫu máy gán độc lập (272 KB)
│   ├── test.py                                    # Script kiểm định độ tương đồng Người vs Máy (1,2 KB)
│   ├── label_with_deepseek.py                     # Script gọi DeepSeek API gán nhãn tự động theo rule (20,2 KB)
│   ├── RULE_LABELING.md                           # Bộ hướng dẫn quy tắc gán nhãn chi tiết (3,6 KB)
│   └── README_DEEPSEEK.md                         # Hướng dẫn thực thi pipeline gán nhãn DeepSeek (2,1 KB)
│
└── processed/
    ├── new_processed.csv                          # Dữ liệu sạch 5.347 mẫu sau khử trùng & tách từ Underthesea (626 KB)
    ├── processed.csv                              # Dữ liệu tiền xử lý từ lô 1.000 mẫu chia Train/Test 80/20 (309 KB)
    └── tien_xu_ly_data.py                         # Pipeline tiền xử lý văn bản tiếng Việt & token hóa (4,5 KB)
```

---

## 1. `raw/` — Dữ Liệu Thu Thập Thô (7.200 Mẫu)

**Tệp lưu trữ:** `data/raw/raw_7k2.csv` · Dung lượng: **1,76 MB** · Định dạng: **UTF-8-SIG**.

Dữ liệu được thu thập trực tiếp từ các trang sản phẩm trực tuyến của Điện máy XANH thông qua Selenium WebDriver, mô phỏng hành vi người dùng và vượt qua các cơ chế tải trang động (AJAX / JavaScript).

### 1.1. Cấu trúc trường dữ liệu (Schema)

Tệp gồm **6 cột** phản ánh đầy đủ thông tin bối cảnh của đánh giá:

| Tên cột | Kiểu dữ liệu | Giá trị khuyết | Mô tả | Ví dụ minh họa |
|---|:---:|:---:|---|---|
| `phan_loai` | String | 0 (0%) | Tên danh mục ngành hàng | `Nồi cơm điện`, `Tủ lạnh`, `Máy lọc nước` |
| `san_pham` | String | 0 (0%) | Tên chính thức của sản phẩm | `Tủ lạnh Samsung Inverter 236 lít RT22FARBDSA/SV` |
| `nguoi_danh_gia` | String | 8 (0,11%) | Tên hiển thị người đánh giá | `Nguyễn Văn A` *(8 bản ghi để trống do người dùng ẩn danh)* |
| `so_sao` | Integer | 0 (0%) | Điểm số đánh giá từ 1 đến 5 | `5` |
| `noi_dung` | String | 0 (0%) | Toàn văn nội dung đánh giá | `"Nồi cơm nấu dẻo ngon, giữ ấm lâu, ship nhanh"` |
| `link` | String | 0 (0%) | URL gốc trang đánh giá sản phẩm | `https://www.dienmayxanh.com/noi-com-dien/...` |

---

### 1.2. Thống kê tổng quan dữ liệu thô

| Tiêu chí thống kê | Giá trị | Nhận xét chuyên môn |
|---|:---:|---|
| **Tổng số bản ghi** | **7.200** | Quy mô dữ liệu cào thực tế |
| **Số đánh giá duy nhất (Unique)** | **5.554** | Đánh giá có văn bản khác biệt |
| **Số đánh giá trùng lặp (Duplicates)** | **1.646 (22,86%)** | Đặc thù review ngắn ("ok", "tốt", "hàng đẹp") xuất hiện lặp lại nhiều lần |
| **Số ngành hàng bao quát** | **10** | Trải đều từ gia dụng nhỏ đến điện lạnh, thiết bị nghe nhìn |
| **Độ dài theo số từ (Word Count)** | Min: 1 · Max: 604 · **Mean: 15,4** · **Median: 8** | Đa số là đánh giá ngắn gọn (50% review từ 8 từ trở xuống) |
| **Độ dài theo ký tự (Char Count)** | Min: 1 · Max: 2.712 · **Mean: 66,2** · **Median: 32** | Khoảng 30–70 ký tự chiếm tỷ trọng lớn nhất |

---

### 1.3. Phân bố theo Danh mục sản phẩm (`phan_loai`)

| STT | Danh mục sản phẩm | Số lượng review | Tỷ lệ % | Đặc điểm nội dung đánh giá |
|:---:|---|:---:|:---:|---|
| 1 | **Nồi cơm điện** | 1.637 | 22,74% | Cơm ngon, chống dính, nắp nồi, thời gian nấu |
| 2 | **Tủ lạnh** | 1.622 | 22,53% | Khả năng làm lạnh, ồn máy nén, ngăn đông mềm, bảo quản rau |
| 3 | **Máy lọc nước** | 1.277 | 17,74% | Vị ngọt của nước, rò rỉ vòi, dịch vụ thay lõi lọc định kỳ |
| 4 | **Tivi** | 943 | 13,10% | Độ nét màn hình 4K, âm thanh, kết nối mạng, remote giọng nói |
| 5 | **Máy sấy quần áo** | 736 | 10,22% | Thời gian sấy, độ ồn, độ khô, nhăn quần áo |
| 6 | **Khác** | 404 | 5,61% | Các thiết bị phụ trợ, phụ kiện gia dụng |
| 7 | **Máy lọc không khí** | 231 | 3,21% | Cảm biến bụi mịn, khử mùi, độ êm khi ngủ |
| 8 | **Camera giám sát** | 180 | 2,50% | Góc nhìn, cảnh báo chuyển động, kết nối ứng dụng điện thoại |
| 9 | **Máy giặt** | 155 | 2,15% | Vắt khô, rung lắc, tiết kiệm điện nước |
| 10 | **Điều hòa** | 15 | 0,21% | Lô 15 mẫu bẫy thử nghiệm spam/edge-cases chọn lọc |
| | **Tổng cộng** | **7.200** | **100,0%** | |

---

### 1.4. Phân bố theo Số sao đánh giá (`so_sao`)

| Số sao | Số lượng | Tỷ lệ % | Xu hướng phản hồi |
|:---:|:---:|:---:|---|
| ⭐ **1 sao** | 602 | 8,36% | Phàn nàn máy lỗi, ồn, hỏng hóc, dịch vụ bảo hành/giao hàng chậm |
| ⭐⭐ **2 sao** | 348 | 4,83% | Sản phẩm không đúng kỳ vọng, dùng nhanh nóng, tính năng hạn chế |
| ⭐⭐⭐ **3 sao** | 546 | 7,58% | Tạm ổn, chất lượng bình thường so với tầm giá |
| ⭐⭐⭐⭐ **4 sao** | 1.266 | 17,58% | Hài lòng, sản phẩm dùng tốt, đóng gói cẩn thận |
| ⭐⭐⭐⭐⭐ **5 sao** | 4.438 | 61,64% | Rất hài lòng, đánh giá tích cực ngắn gọn ("tốt", "ok", "rất thích") |
| **Tổng cộng** | **7.200** | **100,0%** | **Đánh giá tích cực (4–5 sao) chiếm 79,22%** |

---

## 2. `labeled/` — Dữ Liệu Đã Gán Nhãn & Kiểm Định Chất Lượng

Dữ liệu được gán nhãn phân loại nhị phân:
* **`0` = HAM**: Đánh giá thật của khách hàng (kể cả khen, chê gay gắt, hỏi kỹ thuật, teencode).
* **`1` = SPAM**: Đánh giá rác (quảng cáo, link, SĐT/Zalo dẫn khách, nội dung vô nghĩa, sai lệch mặt hàng).

### 2.1. Phân phối nhãn trên toàn bộ 7.200 mẫu (`raw_7k2_deepseek_labeled.csv`)

| Nhãn phân loại | Mã nhãn | Số lượng mẫu | Tỷ lệ phần trăm |
|---|:---:|:---:|:---:|
| **HAM** (Đánh giá thật) | `0` | 7.041 | **97,79%** |
| **SPAM** (Đánh giá rác) | `1` | 159 | **2,21%** |
| **Tổng cộng** | | **7.200** | **100,0%** |

> ⚠️ **Hiện tượng mất cân bằng dữ liệu tự nhiên (Natural Class Imbalance):** Tỷ lệ SPAM tự nhiên trên sàn thương mại điện tử Điện máy XANH chỉ chiếm khoảng **~2,21%**. Đây là đặc điểm khách quan của các sàn có cơ chế kiểm duyệt trước khi hiển thị đánh giá công khai.

---

### 2.2. Thống kê chi tiết theo từng Quy tắc gán nhãn (Rule Breakdown)

Thông tin được trích xuất từ tệp nhật ký checkpoint `raw_7k2_deepseek_labeled.csv.checkpoint.jsonl`:

| Nhóm | Mã Rule | Điều kiện phân loại | Số lượng | Tỷ lệ / Nhóm | Tỷ lệ tổng |
|:---:|:---:|---|:---:|:---:|:---:|
| **HAM (0)** | **H1** | Đánh giá thực tế về chất lượng, công năng, nhân viên, giao hàng, bảo hành | 3.609 | 51,26% | 50,13% |
| | **H2** | Nhận xét ngắn, chung chung nhưng đúng ngữ cảnh ("tốt", "ok", "máy êm") | 2.997 | 42,57% | 41,63% |
| | **H4** | Câu hỏi thắc mắc kỹ thuật, lắp đặt, hướng dẫn sử dụng, bảo hành | 231 | 3,28% | 3,21% |
| | **H3** | Viết không dấu, sử dụng teencode, lỗi chính tả nhưng nội dung có nghĩa | 185 | 2,63% | 2,57% |
| | **H5** | So sánh khách quan giá bán/nơi khác mà không có liên kết/mời mua | 18 | 0,26% | 0,25% |
| | **H6** | Mặc định: Nghi ngờ nhưng thiếu bằng chứng rõ ràng để gán spam | 1 | 0,01% | 0,01% |
| **SPAM (1)** | **S1** | Chứa SĐT, Zalo, link ngoài whitelist, mời inbox, kéo khách sang gian hàng khác | 118 | 74,21% | 1,64% |
| | **S2** | Chuỗi ký tự vô nghĩa (`"asdfgh"`), cờ bạc, nội dung không liên quan mua bán | 40 | 25,16% | 0,56% |
| | **S3** | Sai sản phẩm hoàn toàn (ví dụ: review ốp lưng điện thoại trong mục tủ lạnh) | 1 | 0,63% | 0,01% |
| **Tổng cộng** | | | **7.200** | **100%** | **100,0%** |

---

### 2.3. Phân bố nhãn chéo theo Danh mục sản phẩm

| Danh mục sản phẩm | HAM (0) | SPAM (1) | Tổng số | Tỷ lệ SPAM (%) |
|---|:---:|:---:|:---:|:---:|
| **Điều hòa** *(tập 15 mẫu bẫy thử nghiệm)* | 1 | 14 | 15 | 93,33% |
| **Máy giặt** | 141 | 14 | 155 | 9,03% |
| **Camera giám sát** | 168 | 12 | 180 | 6,67% |
| **Máy lọc không khí** | 218 | 13 | 231 | 5,63% |
| **Khác** | 392 | 12 | 404 | 2,97% |
| **Nồi cơm điện** | 1.607 | 30 | 1.637 | 1,83% |
| **Máy sấy quần áo** | 724 | 12 | 736 | 1,63% |
| **Tivi** | 928 | 15 | 943 | 1,59% |
| **Máy lọc nước** | 1.260 | 17 | 1.277 | 1,33% |
| **Tủ lạnh** | 1.602 | 20 | 1.622 | 1,23% |
| **Toàn bộ kho 7.200** | **7.041** | **159** | **7.200** | **2,21%** |

> 💡 **Giải thích danh mục Điều hòa:** 15 mẫu ngành Điều hòa được đưa vào có chủ đích gồm các bình luận dẫn khách sửa chữa ngoài (`"Em nhận vệ sinh máy lạnh..."`), bán ống đồng giá thợ, và chuỗi ký tự ngẫu nhiên nhằm kiểm tra độ nhạy của thuật toán gán nhãn tự động.

---

### 2.4. Phân bố nhãn chéo theo Số sao đánh giá

| Số sao | HAM (0) | SPAM (1) | Tổng số | Tỷ lệ SPAM (%) | Phân tích đặc điểm Spam |
|:---:|:---:|:---:|:---:|:---:|---|
| ⭐ **1 sao** | 598 | 4 | 602 | 0,66% | Spam phá hoại, spam ký tự vô nghĩa |
| ⭐⭐ **2 sao** | 338 | 10 | 348 | 2,87% | Spam so sánh lôi kéo sang sàn khác |
| ⭐⭐⭐ **3 sao** | 532 | 14 | 546 | 2,56% | Dẫn khách số điện thoại / Zalo |
| ⭐⭐⭐⭐ **4 sao** | 1.236 | 30 | 1.266 | 2,37% | Lời mời kết bạn, dịch vụ lắp đặt ngoài |
| ⭐⭐⭐⭐⭐ **5 sao** | 4.337 | 101 | 4.438 | 2,28% | Chiếm số lượng spam tuyệt đối lớn nhất (khen ảo để chèn SĐT/link) |
| **Tổng cộng** | **7.041** | **159** | **7.200** | **2,21%** | |

---

### 2.5. Kiểm định chất lượng gán nhãn: Đối sánh Người vs Máy (Lô 1.000 mẫu)

Để bảo đảm độ tin cậy tuyệt đối trước khi áp dụng quy mô lớn, một lô chuẩn **1.000 mẫu** thuộc ngành hàng Tủ lạnh đã được thẩm định độc lập song song:
* **Người gán thủ công:** `1000_sample_dau_tu_gan_nhan.csv` (HAM: 985, SPAM: 15)
* **Máy gán tự động:** `1000_sample_dau_may_gan_nhan.csv` (HAM: 984, SPAM: 16)
* **Script thực thi kiểm định:** `python data/labeled/test.py`

#### Ma trận nhầm lẫn (Confusion Matrix)

| | Máy gán HAM (0) | Máy gán SPAM (1) | Tổng thực tế (Người) |
|---|:---:|:---:|:---:|
| **Người gán HAM (0)** | **984** | 1 | 985 |
| **Người gán SPAM (1)** | 0 | **15** | 15 |
| **Tổng Máy gán** | 984 | 16 | **1.000** |

#### Các chỉ số độ đồng thuận (Inter-Annotator Reliability)

| Chỉ số kiểm định | Kết quả đạt được | Tiêu chuẩn chất lượng (QC) |
|---|:---:|---|
| **Số mẫu kiểm thử** | 1.000 mẫu | Lô kiểm soát chất lượng |
| **Số mẫu khớp hoàn toàn** | **999 / 1.000** | Tỷ lệ đồng thuận đạt **99,90%** |
| **Hệ số Cohen's Kappa ($\kappa$)** | **0,9672** | Mức độ thỏa thuận hoàn hảo (Near-perfect, vượt xa ngưỡng khuyến nghị $\kappa \ge 0,75$) |
| **Số mẫu không khớp** | 1 mẫu | Index `[90]`: nội dung `"T"` (Máy gán `1` do quy tắc S2 - ký tự vô nghĩa; Người gán `0`) |

---

### 2.6. Tóm lược Quy tắc Gán Nhãn ([`RULE_LABELING.md`](labeled/RULE_LABELING.md))

1. **Nguyên tắc cốt lõi:**
   - Phân loại dựa trên **mục đích thực chất** của đánh giá, không dựa vào cảm xúc. Review chê gay gắt, teencode, viết không dấu vẫn là HAM (0).
   - Nguyên tắc suy đoán vô tội: *Nếu nghi ngờ nhưng không có bằng chứng spam rõ ràng thì mặc định gán HAM (0)*.
2. **Whitelist Điện máy XANH (Quy tắc S1 - Không coi là spam):**
   - **Tên miền hợp lệ:** `dienmayxanh.com`, `thegioididong.com`, `bachhoaxanh.com`, `mwg.vn`.
   - **Đầu số tổng đài chính thức:** `1900 232 461` (Tư vấn), `1900 232 465` (Kỹ thuật/Bảo hành), `1800 1063` (Khiếu nại), `1900 232 463` (Thợ Điện máy XANH).

---

## 3. `processed/` — Dữ Liệu Đã Tiền Xử Lý (5.347 Mẫu)

**Tệp lưu trữ chính:** `data/processed/new_processed.csv` · Dung lượng: **626 KB** · Số mẫu: **5.347**.

Tệp được sinh ra trực tiếp từ tập 7.200 mẫu đã gán nhãn thông qua pipeline tiền xử lý [`tien_xu_ly_data.py`](processed/tien_xu_ly_data.py).

### 3.1. Thống kê trước và sau tiền xử lý

| Tiêu chí | Trước tiền xử lý (`raw_7k2`) | Sau tiền xử lý (`new_processed`) | Mức độ thay đổi |
|---|:---:|:---:|:---:|
| **Tổng số bản ghi** | 7.200 | **5.347** | Giảm 1.853 bản ghi (**-25,74%**) |
| **Số lượng HAM (0)** | 7.041 (97,79%) | **5.207 (97,38%)** | Giảm 1.834 bản ghi |
| **Số lượng SPAM (1)** | 159 (2,21%) | **140 (2,62%)** | Giảm 19 bản ghi |
| **Số thuộc tính (Cột)** | 7 cột đầy đủ | **2 cột (`noi_dung`, `label`)** | Tối ưu hóa đầu vào cho mô hình |
| **Độ dài từ trung bình** | 15,4 từ | **17,0 từ** | Tăng do tách từ ghép (`ngăn_đông_mềm`) |
| **Độ dài ký tự trung bình** | 66,2 ký tự | **88,6 ký tự** | Tăng do chuyển đổi các token chuẩn |

> 📌 **Lý do giảm 25,74% số mẫu:** Bước khử trùng lặp (`drop_duplicates`) loại bỏ toàn bộ các câu đánh giá trùng lặp nội dung và nhãn (ví dụ: các đánh giá lặp như `"Tốt"`, `"Ok"`, `"Hàng xài tốt"`). Việc này giúp ngăn chặn triệt để hiện tượng rò rỉ dữ liệu (data leakage) giữa tập Train và Test khi huấn luyện mô hình.

---

### 3.2. Quy trình tiền xử lý 5 bước chuẩn hóa

Script [`tien_xu_ly_data.py`](processed/tien_xu_ly_data.py) thực thi 5 công đoạn tự động:

1. **Chuẩn hóa Unicode & Làm sạch cơ bản:**
   - Chuyển mã về Unicode dựng sẵn chuẩn (NFC).
   - Chuyển toàn bộ văn bản về chữ thường (lowercase).
   - Giải mã thực thể HTML (`html.unescape`) và loại bỏ hoàn toàn mã HTML thừa (`re.sub`).
2. **Quy chuẩn thông tin thực thể bằng Token đặc biệt:**
   - Email ➔ `emailtoken` (ví dụ: `hotro@gmail.com` ➔ `emailtoken`)
   - Đường dẫn URL ➔ `urltoken` (ví dụ: `https://shopee.vn/...` ➔ `urltoken`)
   - Số điện thoại cá nhân ➔ `phonetoken` (các dạng số 10 chữ số bắt đầu bằng `0` hoặc `+84`)
   - Biểu tượng cảm xúc ➔ `emojitoken` (nhận diện emoji qua thư viện `emoji`)
   - Số lượng / Giá tiền ➔ `numtoken` (ví dụ: `2 năm`, `5 sao`, `500k` ➔ `numtoken năm`, `numtoken sao`, `numtoken k`)
3. **Lược bỏ dấu câu & Chuẩn hóa từ điển Teencode:**
   - Bỏ các ký tự đặc biệt và dấu câu gây nhiễu từ vựng.
   - Thay thế từ điển viết tắt/teencode tiếng Việt thông dụng:
     * `ko`, `k`, `hok`, `hông` ➔ `không`
     * `dc`, `đc` ➔ `được`
     * `sp` ➔ `sản phẩm`
     * `nv` ➔ `nhân viên`
     * `dmx` ➔ `điện máy xanh`
     * `ib` ➔ `inbox`
   - Nếu câu chỉ chứa toàn ký tự lạ hoặc dấu câu ➔ gán nhãn dự phòng `symboltoken`.
4. **Phân tách từ ghép tiếng Việt (Word Tokenization):**
   - Sử dụng thư viện `underthesea` (`word_tokenize(..., format="text")`) để nối các từ ghép có nghĩa bằng dấu gạch dưới (ví dụ: `nhiệt_độ`, `ngăn_đông`, `nhân_viên`, `bảo_hành`), đồng thời bảo toàn nguyên vẹn các token đặc biệt (`urltoken`, `numtoken`...).
5. **Khử trùng lặp (Deduplication):**
   - Lọc trùng theo cặp `(noi_dung, label)`, chỉ giữ lại mẫu đại diện duy nhất đầu tiên.

---

## 4. 📊 Bảng Tổng Kết Luồng Biến Đổi Dữ Liệu (Pipeline Summary)

| Giai đoạn | Tệp tin dữ liệu | Kích thước | Số dòng | Số nhãn HAM (0) | Số nhãn SPAM (1) | Tỷ lệ SPAM | Trạng thái xử lý |
|:---:|---|:---:|:---:|:---:|:---:|:---:|---|
| **1. Thu thập thô** | `raw/raw_7k2.csv` | 1,76 MB | 7.200 | Chưa gán | Chưa gán | — | Dữ liệu gốc 10 ngành hàng cào từ web |
| **2. Gán nhãn AI** | `labeled/raw_7k2_deepseek_labeled.csv` | 1,77 MB | 7.200 | 7.041 (97,79%) | 159 (2,21%) | 2,21% | Gán nhãn DeepSeek API + Checkpoint JSONL |
| **3. Tiền xử lý** | `processed/new_processed.csv` | 626 KB | 5.347 | 5.207 (97,38%) | 140 (2,62%) | 2,62% | Khử trùng (-25,74%), token hóa, tách từ Underthesea |

---

## 5. ⚠️ Khuyến Nghị Kỹ Thuật Về Mất Cân Bằng Dữ Liệu (Class Imbalance)

Do tỷ lệ nhãn SPAM trong môi trường thực tế chỉ chiếm **2,21% – 2,62%**, các bài toán huấn luyện mô hình học máy cần tuân thủ các nguyên tắc sau:

1. **Không dùng Accuracy làm chỉ số chính:**
   - Một mô hình chỉ cần dự đoán tất cả là HAM (0) đã có thể đạt độ chính xác ~97,4%, nhưng hoàn toàn vô dụng vì không phát hiện được bất kỳ đánh giá spam nào.
2. **Ưu tiên các chỉ số chuyên biệt:**
   - **Precision trên lớp SPAM (1):** Đảm bảo hạn chế tối đa việc chặn nhầm đánh giá thật của khách hàng.
   - **Recall trên lớp SPAM (1):** Đảm bảo phát hiện và quét sạch nhiều review rác nhất có thể.
   - **F1-Score lớp SPAM** và **Macro-averaged F1**.
   - **PR-AUC (Precision-Recall AUC):** Chỉ số tối ưu nhất cho tập dữ liệu cực kỳ mất cân bằng.
3. **Kỹ thuật mô hình hóa khuyến nghị:**
   - Kích hoạt tham số `class_weight='balanced'` trong các thuật toán như Logistic Regression, SVM.
   - Tinh chỉnh ngưỡng quyết định phân loại (classification threshold tuning) thay vì giữ cố định ngưỡng mặc định 0.5.
   - Xem xét kỹ thuật oversampling có kiểm soát (SMOTE) trên tập Train.

---

## 6. 🛠️ Lệnh Thao Tác & Tái Hiện (Reproducibility)

### 6.1. Kiểm tra độ khớp nhãn Người vs Máy trên 1.000 mẫu
```bash
python data/labeled/test.py
```
*Kết quả in ra số dòng khớp (999/1000) và vị trí mẫu lệch duy nhất.*

### 6.2. Chạy pipeline tiền xử lý tạo tệp `new_processed.csv`
```bash
python data/processed/tien_xu_ly_data.py
```
*Đầu vào: `data/labeled/raw_7k2_deepseek_labeled.csv` ➔ Đầu ra: `data/processed/new_processed.csv`.*

### 6.3. Kiểm tra hoặc chạy lại pipeline gán nhãn tự động DeepSeek
Xem hướng dẫn chi tiết các tùy chọn và cấu hình model tại [`data/labeled/README_DEEPSEEK.md`](labeled/README_DEEPSEEK.md):
```bash
# Chạy thử nghiệm kiểm tra kết nối API
python data/labeled/label_with_deepseek.py --dry-run
```

---

## 📌 Bản Quyền & Mục Đích Học Thuật

* Bộ dữ liệu được thu thập từ website [Điện máy XANH](https://www.dienmayxanh.com) và phục vụ **độc quyền cho mục đích học tập, nghiên cứu khoa học và thực nghiệm phi thương mại** tại Học viện Công nghệ Bưu chính Viễn thông (PTIT).
* Mọi thông tin danh tính cá nhân (tên người dùng, số điện thoại, tài khoản email) đều đã được ẩn danh hoặc thay thế bằng các token chung nhằm bảo vệ quyền riêng tư người dùng.
