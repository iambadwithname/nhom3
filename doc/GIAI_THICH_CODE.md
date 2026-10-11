# Giải thích

- Phần A: tổng quan project, đề bài, luồng chạy, cách làm từng task
- Phần B: giải thích từng file / thư mục

---

# PHẦN A. Tổng quan project
**Project làm gì?** 

Dạy máy tính nhìn ảnh X-quang phổi rồi đoán: phổi **bình thường**, **viêm phổi**, hay **COVID**.
Giúp nhận biết bệnh qua việc nhìn ảnh.

1. **Chuẩn bị (Task 1 – xử lý dữ liệu)**
   - Có 5.228 ảnh, nhưng 1.111 ảnh bị **photo trùng**. Nếu để nguyên, ảnh ở tập test sẽ có cái giống hệt ở tập train → máy nhớ lại trong quá trình train nên mô hình cho ra kết quả rất cao. Nên phải clean ảnh thì còn 4.177 ảnh. Đây là phần data leakage giống phần Hiền làm khi lọc chưa kĩ dữ liệu.
   - Chia ảnh làm 3 phần: **70% bài tập** (train, để học), **15% bài kiểm tra thử** (val, để biết học tới đâu), **15% đề thi thật** (test, chỉ dùng 1 lần để chấm điểm).
   - Ảnh nào cũng resize về cùng cỡ cho dễ học.
   - **Augmentation** = xoay nhẹ, dịch nhẹ, chỉnh sáng tối ảnh bài tập → máy gặp nhiều kiểu ảnh hơn, không học vẹt.

2. **4 mô hình**
   - **Simple CNN (Task 2):** Đơn giản thôi
   - **Complex CNN (Task 3):** Ghép từ các "khối" có sẵn ý tưởng (Inception, Residual).
   - **ResNet-50, MobileNetV3 (Task 4 – Transfer Learning):** **đã học rất nhiều ảnh khác từ trước** (1,2 triệu ảnh). Mình chỉ dạy thêm phần nhận biết X-quang. Giống người đã giỏi nhìn ảnh, chỉ cần học thêm môn mới.

3. **Học (train)**
   - Máy xem bài tập, đoán, sai thì sửa. Lặp lại nhiều vòng (mỗi vòng = 1 **epoch**).
   - Sau mỗi vòng làm bài kiểm tra thử. Thấy không tiến bộ nữa thì **dừng** (early stopping), giữ lại lúc giỏi nhất.
   - Thử vài tốc độ học (**learning rate**) và số bài mỗi lần (**batch size**) xem cái nào tốt nhất → chọn lr 0,001, batch 64 cho Simple CNN.

4. **Chấm thi (Task 5 – đánh giá)**
   - Cho cả 4 học sinh làm đề thi thật (627 ảnh).
   - **Accuracy** = đúng bao nhiêu %. 
   - **Confusion matrix** = bảng xem hay nhầm bệnh nào thành bệnh nào.
   - **Kết quả:** Mô hình "đã học sẵn" thắng: **MobileNetV3 đúng 97,6%**, ResNet-50 97,3%, Complex CNN 95,7%, Simple CNN 94,9%.
   - Hay nhầm nhất: **viêm phổi bị đoán là bình thường**.
   - **Grad-CAM** = xem mô hình nhìn vào chỗ nào trên ảnh. Có lúc nhìn đúng vào phổi, có lúc lại nhìn ra bụng, cổ → chưa hoàn hảo.

**Code chạy theo thứ tự 4 lệnh:**

| Lệnh | Giống như |
|---|---|
| `uv run nhom3 split` | Chuẩn bị đề: bỏ ảnh trùng, chia bài tập / kiểm tra / thi |
| `uv run nhom3 tune` | Thử vài cách học cho Simple CNN, chọn cách tốt nhất |
| `uv run nhom3 train` | Cho 4 mô hình học rồi chấm đánh giá |
| `uv run nhom3 figures` | Vẽ biểu đồ, bảng điểm phục vụ cho báo cáo |

**Thư mục nào chứa gì:**

| Thư mục | Chứa |
|---|---|
| `src/nhom3/` | Code (train và test) |
| `data/` | Danh sách ảnh đã lọc, ảnh nào thuộc phần nào |
| `results/` | Điểm thi + biểu đồ của từng model |
| `doc/` | Tài liệu (file này) |
| `.venv/` | Đồ nghề Python, tự sinh, kệ nó |

---

# PHẦN A. TỔNG QUAN PROJECT

## A1. Đề bài

Lớp chia nhóm 4 người, mỗi nhóm chọn 1 bộ ảnh (không trùng nhóm khác), làm bài toán phân loại với **5 yêu cầu**:

1. Tiền xử lý dữ liệu: làm sạch, chuẩn hóa, tăng cường (augmentation)
2. Tự thiết kế **CNN đơn giản** (conv, pooling, fully connected)
3. Tự thiết kế **CNN phức tạp** có dùng các khối (blocks)
4. CNN dùng **Transfer Learning / Fine-tuning**
5. **Đánh giá** mô hình bằng độ đo phù hợp

**Đề tài nhóm:** cho 1 ảnh X-quang ngực → đoán thuộc lớp **COVID-19 / Normal / Pneumonia**.
**Dataset:** Kaggle `sachinkumar413/covid-pneumonia-normal-chest-xray-images`, 5.228 ảnh PNG (COVID 1.626, NORMAL 1.802, PNEUMONIA 1.800).


| File trong `src/nhom3/` | Vai trò |
|---|---|
| `__init__.py` | Nhận lệnh split / tune / train / figures và gọi đúng hàm |
| `data.py` | Dữ liệu: làm sạch, chia tập, đọc ảnh, augmentation, chia batch |
| `models.py` | 4 mô hình |
| `train.py` | Train, đánh giá, danh sách thí nghiệm |
| `figures.py` | Vẽ hình, bảng, Grad-CAM |

## A3. Task 1 – Tiền xử lý dữ liệu (`data.py`, lệnh `split`)

### Làm sạch (`build_splits`)
Dataset có **rất nhiều ảnh trùng**. Nếu cùng 1 ảnh nằm cả ở train lẫn test thì mô hình chỉ cần "nhớ" là đoán đúng → điểm test cao giả (**data leakage**). Làm sạch 4 bước:

| Bước | Cách làm | Bị loại |
|---|---|---|
| 1. Ảnh lỗi | Mở thử từng ảnh, lỗi thì bỏ | 0 |
| 2. Trùng hoàn toàn | So mã MD5 của file, giống nhau từng byte thì giữ 1 | 229 |
| 3. Gần trùng | Thu ảnh về "dấu vân tay" 16×16 (`dhash`), khác nhau ≤ 4 bit coi là 1 ảnh; `group_duplicates` gom nhóm, mỗi nhóm giữ 1 | 819 |
| 4. Trùng mà khác nhãn | Nhóm trùng có cả COVID lẫn NORMAL → không biết nhãn nào đúng, bỏ cả nhóm | 3 |
| **Còn lại** | | **4.177** (COVID 1.077, NORMAL 1.425, PNEUMONIA 1.675) |

Bằng chứng: notebook `x-ray.ipynb` (chia trên dữ liệu chưa sạch) có **188/785 ảnh test (24%)** có bản sao trong train, riêng COVID là 43%.

### Chia tập
70 / 15 / 15, chia riêng từng lớp (stratified), seed 42, lưu ở `data/splits.csv`.

| Tập | COVID | NORMAL | PNEUMONIA | Tổng | Dùng để |
|---|---|---|---|---|---|
| Train | 754 | 997 | 1.172 | 2.923 | Cho mô hình học |
| Val | 162 | 214 | 251 | 627 | Chọn tham số, quyết định khi nào dừng |
| Test | 161 | 214 | 252 | 627 | Chấm điểm cuối, chỉ dùng 1 lần |

### Chuẩn hóa (`load_images`, `XrayDataset`)
- Ảnh xám, resize **224×224** (Simple CNN là **128×128**), chia 255 về [0, 1].
- 2 CNN tự xây: thêm `(x − 0.5) / 0.25`, 1 kênh.
- ResNet / MobileNet: nhân thành 3 kênh, chuẩn hóa theo mean/std của ImageNet.
- Ảnh đã resize lưu vào `data/cache_*.pt` để lần sau load nhanh.

### Augmentation (`train_augment`)
Chỉ cho tập train, mỗi epoch biến đổi ngẫu nhiên: xoay ±10°, dịch 5%, zoom 90–110%, sáng/tương phản ±20%. **Không lật ngang** vì tim nằm bên trái.

### Mất cân bằng lớp (`class_weights` trong train.py)
COVID ít ảnh nhất → trong loss mỗi lớp có trọng số tỉ lệ nghịch với số ảnh, để mô hình không bỏ qua COVID.

## A4. Task 2 – Simple CNN (`models.py → SimpleCNN`)

Kiến trúc nhóm tự thiết kế trong notebook (Keras), chuyển sang PyTorch, giữ nguyên comment:

```
Input 1×128×128
→ Conv 32 filter 3×3 + ReLU   → 32×126×126
→ Conv 32 filter 3×3 + ReLU   → 32×124×124
→ MaxPool 2×2                 → 32×62×62
→ Flatten                     → 123.008
→ Dense 128 + ReLU → Dropout 0.5
→ Dense 3 (Softmax)
```

- **15.755.107 tham số** (khớp `model.summary()` trong notebook), gần hết nằm ở lớp Dense sau Flatten.
- Không gọi softmax trong model vì `CrossEntropyLoss` của PyTorch đã tự làm.

**Chọn cỡ ảnh:** thử 128 và 224 → val F1 gần như bằng nhau (0,924 vs 0,927), bản 224 nặng gấp 3 (49,6M tham số) và chậm gấp đôi → chọn **128**.

**Cách train (chung cho mọi model, hàm `train()`):**
- Optimizer **AdamW**, lr tự giảm (× 0,3) khi val loss không giảm sau 3 epoch.
- **Early stopping:** val loss không giảm sau 8 epoch thì dừng, giữ model tốt nhất vào `best.pt`.
- Mixed precision trên GPU RTX 3050.

**Dò hyperparameter (lệnh `tune`):** 3 lr × 2 batch size, chấm trên **tập val**:

| lr | batch | Val F1 |
|---|---|---|
| 0,003 | 32 | 0,19 (không học được) |
| 0,003 | 64 | 0,937 |
| 0,001 | 32 | 0,924 |
| **0,001** | **64** | **0,956** ← chọn |
| 0,0003 | 32 | 0,942 |
| 0,0003 | 64 | 0,945 |

**Train 2 bản** để xem tác dụng của augmentation: `simple_cnn` (không aug) và `simple_cnn_aug` (có aug).

## A5. Task 3 – Complex CNN (`models.py → ComplexCNN`)

Kết hợp 2 loại khối:
- **Inception** (GoogLeNet): 4 nhánh song song (conv 1×1; 1×1→3×3; 1×1→5×5; maxpool→1×1) rồi ghép theo kênh → nhìn chi tiết nhiều kích thước cùng lúc.
- **Residual** (ResNet): `đầu ra = F(x) + x`, đường tắt giúp gradient truyền ngược dễ, train được mạng sâu hơn.

```
Input 1×224×224
→ Stem: Conv 7×7 → MaxPool → Conv 1×1 → Conv 3×3 → MaxPool   (→ 28×28)
→ 2 khối Inception (128, 224 kênh) → MaxPool                  (→ 14×14)
→ 4 khối Residual (256, 256, 384, 384 kênh)                   (→ 7×7)
→ Global AvgPool → Dropout → Dense 128 → Dropout → Dense 3
```
Mỗi conv có BatchNorm + ReLU. ~**7,6 triệu** tham số. Train lr 0,001, batch 32, có aug.

## A6. Task 4 – Transfer Learning (`resnet50_transfer`, `mobilenet_transfer`)

Dùng model đã train sẵn trên ImageNet (1,2 triệu ảnh):
- **ResNet-50**: 23,5 triệu tham số
- **MobileNetV3-Large**: 4,2 triệu tham số, nhẹ và nhanh hơn nhiều

Thay lớp cuối bằng lớp mới 3 đầu ra, train **2 giai đoạn**:
1. **5 epoch đầu (feature extraction):** đóng băng toàn bộ thân, chỉ train lớp mới, lr 0,001.
2. **Fine-tuning:** mở khóa tầng cuối (`unfreeze_last_stages`: layer4 của ResNet-50, 4 khối cuối MobileNetV3), train tiếp với lr nhỏ 0,0001 để không phá những gì đã học.

## A7. Task 5 – Đánh giá (`evaluate`, `figures.py`)

Mỗi model chấm **1 lần** trên tập test (627 ảnh): Accuracy, Precision / Recall / F1 (trung bình macro 3 lớp), Confusion matrix → lưu vào `metrics_test.json`.

| Mô hình | Accuracy | F1 | Tham số |
|---|---|---|---|
| Simple CNN | 0,9490 | 0,9504 | 15,8M |
| Simple CNN + Aug | 0,9474 | 0,9498 | 15,8M |
| Complex CNN + Aug | 0,9569 | 0,9596 | 7,6M |
| ResNet-50 TL | 0,9729 | 0,9752 | 23,5M |
| **MobileNetV3 TL** | **0,9761** | **0,9781** | 4,2M |

**Đọc kết quả:**
- **Transfer learning thắng rõ**: chỉ ~2.900 ảnh train, quá ít để mạng train từ đầu bằng được mạng đã học sẵn từ 1,2 triệu ảnh.
- **Complex CNN tốt hơn Simple CNN** dù ít tham số hơn một nửa.
- **Augmentation không tăng điểm test** nhưng chống overfit: không aug thì val loss ngừng giảm từ epoch 4; có aug thì còn cải thiện tới epoch 22.

**3 câu hỏi của đề tài:**
1. *Có nhầm COVID với Pneumonia không?* Gần như không. Recall COVID ≥ 0,96 ở mọi model, ResNet-50 đúng cả 161/161.
2. *Lớp nào khó nhất?* Normal ↔ Pneumonia. Lỗi hay gặp nhất: Pneumonia bị đoán thành Normal (10–18/252 ảnh). Đây là âm tính giả, nguy hiểm nhất.
3. *Mô hình nhìn vào đâu?* **Grad-CAM** tô màu vùng ảnh ảnh hưởng tới quyết định. Ảnh Pneumonia đoán đúng thì sáng trong phổi; nhưng nhiều ảnh (nhất là ảnh sai) lại sáng ở bụng, cơ hoành, cổ.

**Hạn chế:**
- Ảnh NORMAL/PNEUMONIA là ảnh trẻ em, ảnh COVID là người lớn từ nguồn khác → mô hình có thể học "đường tắt" (phân biệt nguồn ảnh chứ không phải bệnh).
- Không có mã bệnh nhân, không chắc 2 ảnh của cùng 1 người không rơi vào 2 tập khác nhau.
- Mỗi model chỉ train 1 lần với 1 seed.

---

# PHẦN B. GIẢI THÍCH TỪNG FILE

Cây thư mục của project:

```
nhom3/
├── .gitignore            danh sách file không đưa lên git
├── .python-version       phiên bản Python (3.13)
├── README.md             giới thiệu project trên GitHub
├── pyproject.toml        cấu hình project + thư viện
├── uv.lock               khóa phiên bản thư viện (tự sinh)
├── .venv/                môi trường Python (tự sinh, không lên git)
├── data/                 danh sách ảnh sau khi làm sạch + cache ảnh
├── doc/                  tài liệu (file này + đề xuất đề tài)
├── results/              kết quả train, số liệu, hình cho báo cáo
└── src/nhom3/            toàn bộ code
```

Ký hiệu: **[git]** = có trên GitHub, **[máy]** = chỉ có trên máy (tự sinh, không lên git).

---

## B1. Các file ở thư mục gốc

### .gitignore [git]
Liệt kê những thứ git bỏ qua, không đưa lên GitHub:
- `.venv/`: môi trường Python, mỗi người tự tạo bằng `uv sync`
- `__pycache__/`: file Python tự dịch sẵn
- `data/cache_*.pt`: ảnh đã resize (~280 MB)
- `results/**/best.pt`: trọng số mô hình (nặng, có file 94 MB)
- `results/**/*.npy`: xác suất dự đoán
- `*.log`: file log

### .python-version [git]
Chỉ có 1 dòng `3.13`, báo cho `uv` dùng Python 3.13.

### README.md [git]
Trang GitHub hiện ra đầu tiên khi mở repo: tên bài toán, link dataset, cách cài (`uv sync`), 4 lệnh chạy.
Không xoá được vì `pyproject.toml` có dòng `readme = "README.md"`, thiếu file thì `uv sync` lỗi.

### pyproject.toml [git]
File cấu hình chính, `uv` đọc file này để biết cài gì, chạy gì:
- `[project]`: tên `nhom3`, mô tả, 4 thành viên, cần Python ≥ 3.13.
- `dependencies`: 7 thư viện code dùng thật:
  - `torch`, `torchvision`: xây và train mạng CNN, có sẵn ResNet-50 / MobileNetV3
  - `numpy`: tính toán mảng
  - `pillow`: đọc ảnh PNG
  - `scikit-learn`: tính accuracy, precision, recall, F1, confusion matrix
  - `matplotlib`: vẽ hình
  - `kagglehub`: tải dataset từ Kaggle
- `[project.scripts]`: dòng `nhom3 = "nhom3:main"` tạo ra lệnh `uv run nhom3 ...`, gọi hàm `main()` trong `src/nhom3/__init__.py`.
- `[build-system]`: dùng `uv_build` đóng gói `src/nhom3` thành package.
- `[tool.uv.sources]`, `[[tool.uv.index]]`: tải torch bản CUDA 12.8 (chạy GPU NVIDIA) thay vì bản thường.

Sửa file này xong phải chạy `uv sync` để `uv.lock` cập nhật theo.

### uv.lock [git] — tự sinh, không cần để tâm
Ghi chính xác phiên bản của toàn bộ thư viện (kể cả thư viện phụ) để cả nhóm `uv sync` ra môi trường giống hệt nhau, tránh "máy t chạy được máy m không". Không sửa tay.

### .venv/ [máy] — tự sinh, không cần để tâm
Môi trường Python riêng của project (~4,5 GB), do `uv sync` tạo ra. Xoá đi thì chạy `uv sync` là có lại.
- `Lib/`: chứa các thư viện đã cài, nặng nhất (torch chiếm phần lớn)
- `Scripts/`: file chạy được: `python.exe`, `nhom3.exe` (chính là lệnh `uv run nhom3`), `activate` (bật môi trường bằng tay)
- `etc/`, `share/`: file phụ vài thư viện tự thêm vào
- `pyvenv.cfg`: ghi môi trường dùng Python bản nào, ở đâu
- `.gitignore` (chứa `*`), `.lock`, `CACHEDIR.TAG`: file đánh dấu của `uv`

---

## B2. data/

### splits.csv [git] — cần nộp
Danh sách **4.177 ảnh đã làm sạch**, mỗi dòng 3 cột: tên file, lớp, thuộc tập nào. Ví dụ:
```
file,label,split
COVID/COVID_10.png,COVID,train
```
Tạo bởi lệnh `split`. Mọi model đều đọc file này để lấy ảnh → tất cả dùng chung một cách chia, ai chạy lại cũng ra giống nhau.

### cleaning_summary.json [git] — cần nộp
Kết quả làm sạch, tạo bởi lệnh `split`:
- `raw`: số ảnh gốc mỗi lớp (1.626 / 1.802 / 1.800)
- `corrupt`: 0 ảnh lỗi
- `exact_dups`: 229 ảnh trùng hoàn toàn
- `near_dups`: 819 ảnh gần trùng
- `label_conflicts`: 3 ảnh trùng nhưng khác nhãn
- `clean`: số ảnh còn lại mỗi lớp (1.077 / 1.425 / 1.675)
- `split`: số ảnh mỗi lớp trong train / val / test

Hình `class_distribution.png` và số liệu phần dataset trong báo cáo lấy từ file này.

### cache_train_224.pt, cache_val_224.pt, cache_test_224.pt, cache_*_128.pt [máy] — tự sinh, không cần để tâm
Ảnh đã chuyển xám và resize sẵn, lưu dạng tensor PyTorch. Lần đầu chạy, code đọc 4.177 file PNG (chậm) rồi lưu ra đây; các lần sau load thẳng file này (nhanh).
- `_224`: cho Complex CNN, ResNet-50, MobileNetV3
- `_128`: cho Simple CNN

Xoá đi thì lần chạy sau tự tạo lại.

---

## B3. doc/

### GIAI_THICH_CODE.md [git]
File bạn đang đọc: tổng quan project + giải thích từng file.

### readme.md [git]
Đề xuất đề tài do nhóm viết (tiếng Việt): bài toán, các việc chính, điểm hay của topic (3 câu hỏi phân tích), hướng phát triển (Transfer Learning, Grad-CAM), lý do chọn, mục tiêu.

---

## B4. src/nhom3/ — code

### __init__.py — "cửa vào" chương trình
Hàm `main()` đọc lệnh gõ sau `uv run nhom3`:
- `split` → gọi `build_splits()` (data.py), ghi `splits.csv` + `cleaning_summary.json`
- `tune` → train 6 cấu hình trong `TUNING`, in accuracy / F1 trên tập val
- `train` → chạy `run()` cho 5 thí nghiệm trong `EXPERIMENTS` (hoặc chỉ cái được gọi tên, vd `uv run nhom3 train resnet50_tl`)
- `figures` → gọi `make_all()` (figures.py)
- gõ sai → in dòng hướng dẫn

### data.py — dữ liệu
Hằng số đầu file: `CLASSES` (3 lớp, thứ tự COVID = 0, NORMAL = 1, PNEUMONIA = 2), `KAGGLE_ID`, đường dẫn `data/`, `IMG_SIZE = 224`, `SEED = 42`.

| Hàm / class | Làm gì |
|---|---|
| `raw_dir()` | Tìm thư mục ảnh gốc: dùng bản đã tải trong `~/.cache/kagglehub`, chưa có thì tải từ Kaggle |
| `dhash(path)` | Tạo "dấu vân tay" 16×16 của ảnh: thu nhỏ, so độ sáng 2 pixel cạnh nhau → 256 bit. Ảnh giống nhau thì dấu vân tay gần giống |
| `group_duplicates(hashes, threshold)` | Gom các ảnh có dấu vân tay khác nhau ≤ 4 bit vào cùng nhóm (thuật toán union-find) |
| `build_splits()` | Làm sạch 4 bước (ảnh lỗi → trùng MD5 → gần trùng → trùng khác nhãn) rồi chia 70/15/15 theo từng lớp, ghi `splits.csv`, trả về số liệu cho `cleaning_summary.json` |
| `read_split(split)` | Đọc `splits.csv`, trả danh sách đường dẫn ảnh + nhãn của 1 tập |
| `load_images(paths, size)` | Mở từng ảnh, chuyển xám, resize, gom thành 1 tensor |
| `train_augment()` | Phép augmentation: xoay ±10°, dịch 5%, zoom 0,9–1,1, sáng / tương phản ±20%, không lật ngang |
| `XrayDataset` | Trả từng ảnh + nhãn khi train: augmentation (nếu bật) → chia 255 → (nếu model pretrained) nhân 3 kênh → chuẩn hóa |
| `get_split(split, size)` | Lấy ảnh của 1 tập; có cache thì load, chưa có thì đọc ảnh rồi lưu cache |
| `make_loader(...)` | Tạo DataLoader chia ảnh thành batch; tập train thì xáo trộn và augmentation |

### models.py — 4 mô hình

| Hàm / class | Làm gì |
|---|---|
| `SimpleCNN` | Model của nhóm: Conv 32 → Conv 32 → MaxPool → Flatten → Dense 128 → Dropout 0,5 → Dense 3. Ảnh 128×128, 15,8M tham số. `features` = phần conv, `classifier` = phần dense |
| `conv_bn(...)` | Viết tắt cho Conv + BatchNorm + ReLU, dùng nhiều lần trong Complex CNN |
| `InceptionBlock` | 4 nhánh song song (`b1`–`b4`) rồi ghép theo kênh |
| `ResidualBlock` | 2 lớp conv 3×3 (`body`) + đường tắt (`shortcut`), đầu ra = body(x) + shortcut(x) |
| `ComplexCNN` | `stem` (conv đầu) → `inception` (2 khối) → `residual` (4 khối) → `head` (pool + dense). 7,6M tham số |
| `resnet50_transfer()` | Tải ResNet-50 đã train trên ImageNet, đóng băng hết, thay lớp `fc` cuối thành 3 đầu ra |
| `mobilenet_transfer()` | Như trên với MobileNetV3-Large |
| `unfreeze_last_stages(m)` | Mở khóa tầng cuối để fine-tune (layer4 của ResNet-50, 4 khối cuối MobileNetV3) |
| `count_params(m)` | Đếm tổng số tham số và số tham số đang được train |

### train.py — train và đánh giá

| Hàm / class | Làm gì |
|---|---|
| `Config` | Cấu hình 1 lần chạy: tên, loại model, có aug không, lr, batch size, số epoch tối đa, patience (early stopping), cỡ ảnh, số epoch chỉ train lớp cuối (TL)... |
| `build_model(cfg)` | Tạo model theo `cfg.arch` |
| `class_weights()` | Tính trọng số mỗi lớp nghịch với số ảnh, COVID ít ảnh nên nặng hơn |
| `run_epoch(...)` | Chạy 1 vòng qua dữ liệu. Có optimizer → train (tính loss, lan truyền ngược, cập nhật trọng số); không có → chỉ tính loss / accuracy |
| `make_optimizer(...)` | Tạo AdamW + bộ giảm lr (giảm × 0,3 khi val loss không giảm 3 epoch) |
| `train(cfg)` | Vòng train chính: mỗi epoch train rồi tính val loss; TL thì tới epoch 5 mở khóa và đổi lr nhỏ; val loss tốt nhất thì lưu `best.pt`; 8 epoch không cải thiện thì dừng; cuối cùng ghi `history.json` + `meta.json` |
| `load_trained(cfg)` | Tạo model và nạp `best.pt` |
| `predict(...)` | Chạy model trên 1 tập, trả nhãn thật + xác suất mỗi lớp |
| `evaluate(cfg, split)` | Tính accuracy, precision, recall, F1, confusion matrix, ghi `metrics_<split>.json` + `probs_<split>.npy` |
| `run(cfg)` | train → evaluate test → evaluate val → in kết quả |
| `EXPERIMENTS` | 5 thí nghiệm chính: `simple_cnn`, `simple_cnn_aug`, `complex_cnn_aug`, `resnet50_tl`, `mobilenetv3_tl` |
| `TUNING` | 6 cấu hình dò lr (0,003 / 0,001 / 0,0003) × batch (32 / 64) cho Simple CNN |
| `by_name(name)` | Tìm config theo tên (dùng cho `uv run nhom3 train <tên>`) |

### figures.py — vẽ hình cho báo cáo

| Hàm | Vẽ / tạo ra |
|---|---|
| `class_distribution()` | `class_distribution.png`: số ảnh mỗi lớp trước / sau làm sạch |
| `sample_grid()` | `samples.png`: 5 ảnh mẫu mỗi lớp |
| `augmentation_examples()` | `augmentation.png`: ảnh gốc và 4 bản augmentation |
| `_stack()` | Hàm phụ vẽ các lớp thành ô xếp chồng |
| `architectures()` | 4 sơ đồ kiến trúc `arch_*.png` |
| `curves()` | `training_curves.png`: loss / accuracy theo epoch của 5 model |
| `confusion_matrices()` | `confusion_matrices.png`: confusion matrix của 5 model |
| `tables()` | `results_tables.md`: 3 bảng kết quả |
| `_target_layer()`, `gradcam()` | Tính Grad-CAM: lấy gradient của lớp được đoán theo feature map của lớp conv cuối → heatmap |
| `gradcam_figures()` | `gradcam_complex_cnn_aug.png`, `gradcam_resnet50_tl.png`: mỗi lớp 3 ảnh đúng + tối đa 6 ảnh sai |
| `make_all()` | Chạy tất cả (lệnh `figures`) |

### __pycache__/ [máy] — tự sinh, không cần để tâm
Python tự lưu bản dịch sẵn của file `.py` để lần sau chạy nhanh hơn.

---

## B5. results/

### Thư mục của 5 model chính: simple_cnn, simple_cnn_aug, complex_cnn_aug, resnet50_tl, mobilenetv3_tl
Mỗi thư mục có:

| File | Nội dung | |
|---|---|---|
| `history.json` | Train loss / acc, val loss / acc, lr và giai đoạn (`full` / `head` / `finetune`) theo từng epoch. Dùng để vẽ đường cong train | [git] |
| `meta.json` | Cấu hình đã dùng, epoch tốt nhất, số epoch đã chạy, thời gian train, số tham số | [git] |
| `metrics_test.json` | Kết quả trên tập test: accuracy, precision, recall, F1 (macro), confusion matrix, kết quả từng lớp. **Số trong báo cáo lấy từ đây** | [git] |
| `metrics_val.json` | Như trên nhưng cho tập val | [git] |
| `best.pt` | Trọng số model tốt nhất (nặng, 2–94 MB) | [máy] |
| `probs_test.npy`, `probs_val.npy` | Xác suất model đoán cho từng ảnh, Grad-CAM dùng để chọn ảnh đúng / sai | [máy] |

### 6 thư mục tune_simple_lr*_bs* (vd `tune_simple_lr0.001_bs64`)
Kết quả 6 lần dò lr × batch size cho Simple CNN. Có `history.json`, `meta.json`, `metrics_val.json` [git] và `best.pt`, `probs_val.npy` [máy]. Chỉ chấm trên tập val (không đụng tập test). Bảng dò tham số trong báo cáo lấy từ đây.

### figures/ [git] — 11 hình cho báo cáo (tạo bởi lệnh `figures`)

| Hình | Nội dung | Mục báo cáo |
|---|---|---|
| `samples.png` | Ảnh mẫu mỗi lớp | II |
| `class_distribution.png` | Số ảnh trước / sau làm sạch | III.1 |
| `augmentation.png` | Ví dụ augmentation | III.1 |
| `arch_simple_cnn.png` | Sơ đồ Simple CNN (báo cáo dùng sơ đồ nhóm tự vẽ) | III.2 |
| `arch_complex_cnn.png` | Sơ đồ Complex CNN + khối Inception + khối Residual | III.3 |
| `arch_resnet50_tl.png`, `arch_mobilenetv3_tl.png` | Sơ đồ 2 model TL (xám = đóng băng, xanh = fine-tune) | III.4 |
| `training_curves.png` | Đường cong loss / accuracy | IV |
| `confusion_matrices.png` | Confusion matrix 5 model | IV |
| `gradcam_resnet50_tl.png` | Grad-CAM của ResNet-50 | IV |
| `gradcam_complex_cnn_aug.png` | Grad-CAM của Complex CNN (dự phòng) | IV |

### results_tables.md [git]
3 bảng gom sẵn: kết quả test 5 model, kết quả từng lớp, bảng dò lr × batch. Tiện để chép vào báo cáo.

### train_log.txt [git]
Log in ra khi train (từng epoch của mọi model), bằng chứng nhóm đã chạy thật. Phần cuối là log train lại Simple CNN (128×128) ngày 11/10.
