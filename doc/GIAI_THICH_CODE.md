# Giải thích project – ôn vấn đáp

Tài liệu ôn cho buổi vấn đáp, gồm 2 phần như cô yêu cầu:
- **Phần code** (mục 1 → 6): dataset + tiền xử lý, Simple CNN, Complex CNN, Transfer Learning, cách train / đánh giá, kết quả.
- **Phần lý thuyết** (mục 7): trả lời 15 câu hỏi.

Mục 8 là danh sách file trong repo, mục 9 là các câu cô có thể hỏi thêm.

---

## 0. Tóm tắt 1 phút

- **Bài toán:** phân loại ảnh X-quang ngực thành 3 lớp **COVID-19 / Normal / Pneumonia** (phân loại đa lớp, mỗi ảnh đúng 1 nhãn).
- **Dữ liệu:** 5.228 ảnh từ Kaggle → bỏ ảnh trùng còn **4.177 ảnh** → chia train / val / test = 70 / 15 / 15.
- **4 model:** Simple CNN (tự thiết kế), Simple CNN + augmentation, Complex CNN (khối Inception + Residual, tự thiết kế), MobileNetV3 (transfer learning).
- **Kết quả trên tập test (627 ảnh):**

| Model | Accuracy | Precision | Recall | F1 | Số tham số |
|---|---|---|---|---|---|
| Simple CNN | 0,9490 | 0,9492 | 0,9519 | 0,9504 | 15,76M |
| Simple CNN + Aug | 0,9474 | 0,9491 | 0,9515 | 0,9498 | 15,76M |
| Complex CNN + Aug | 0,9569 | 0,9605 | 0,9594 | 0,9596 | 7,56M |
| **MobileNetV3 TL + Aug** | **0,9761** | **0,9780** | **0,9785** | **0,9781** | 4,21M |

- **Chạy khi vấn đáp:** `uv run python src/5_predict.py` → nạp model đã lưu, in bảng trên + confusion matrix, không train lại.

---

## 1. Chạy code

| Lệnh | Làm gì | Thời gian (GPU RTX 3050) |
|---|---|---|
| `uv sync` | Cài thư viện | vài phút lần đầu |
| `uv run python src/1_preprocess.py` | Làm sạch + chia tập | vài phút |
| `uv run python src/2_simple_cnn.py` | Dò lr/batch + train Simple CNN | ~10 phút |
| `uv run python src/3_complex_cnn.py` | Train Complex CNN | ~6 phút |
| `uv run python src/4_transfer_learning.py` | Train MobileNetV3 | ~5 phút |
| **`uv run python src/5_predict.py`** | **Nạp `best.pt`, dự đoán tập test (dùng khi vấn đáp)** | ~20 giây |
| `uv run python src/5_predict.py anh.png` | Dự đoán 1 ảnh bất kỳ | vài giây |
| `uv run python src/6_figures.py` | Vẽ hình + bảng cho báo cáo | ~1 phút |

File dùng chung: `src/utils.py` (đọc ảnh, vòng train, đánh giá) và `src/models.py` (định nghĩa model).

**Model được lưu thế nào?** Lúc train, mỗi khi val loss tốt nhất thì lưu trọng số bằng `torch.save(model.state_dict(), "best.pt")` (utils.py dòng 152). Lúc dự đoán: tạo model rỗng rồi `model.load_state_dict(torch.load("best.pt"))` (5_predict.py dòng 24). Giống `pickle` bên sklearn nhưng PyTorch chỉ lưu trọng số, nên phải có code tạo model (models.py) mới nạp lại được.

---

## 2. Nội dung 1 – Dataset, bài toán, tiền xử lý

### 2.1 Dataset
- **Nguồn:** Kaggle `sachinkumar413/covid-pneumonia-normal-chest-xray-images`.
- **Ảnh gì:** ảnh X-quang ngực chụp thẳng, file PNG, 3 thư mục COVID / NORMAL / PNEUMONIA.
- **Số ảnh:** 5.228 (COVID 1.626, NORMAL 1.802, PNEUMONIA 1.800).
- **Số lớp:** 3. Thứ tự nhãn trong code: `CLASSES = ["COVID", "NORMAL", "PNEUMONIA"]` (utils.py dòng 14) → COVID = 0, NORMAL = 1, PNEUMONIA = 2.
- **Kích thước ảnh gốc:** 232×232 (4.228 ảnh) hoặc 256×256 (1.000 ảnh), lưu dạng RGB nhưng nội dung là ảnh xám.

### 2.2 Bài toán
- **Input:** 1 ảnh X-quang ngực. **Output:** 1 trong 3 nhãn + xác suất mỗi lớp (softmax).
- Phân loại đa lớp (multi-class), các lớp loại trừ nhau.
- Phân tích thêm: model có nhầm COVID với viêm phổi không, lớp nào khó nhất, model nhìn vào vùng nào (Grad-CAM).

### 2.3 Làm sạch dữ liệu – `src/1_preprocess.py`

**Vì sao phải làm sạch:** dataset có rất nhiều ảnh trùng. Nếu 1 ảnh vừa ở train vừa ở test, model chỉ cần "nhớ" là đoán đúng → điểm test cao giả (**data leakage – rò rỉ dữ liệu**). Kiểm tra cách chia ngẫu nhiên trên dữ liệu chưa làm sạch: 188/785 ảnh test (24%) có bản sao trong tập train.

| Dòng | Code làm gì |
|---|---|
| 10–14 | `raw_dir()` tìm thư mục ảnh (bản đã tải về máy, chưa có thì tải từ Kaggle). Liệt kê mọi file `.png` trong 3 thư mục, nhãn = tên thư mục |
| 16–23 | **Bỏ ảnh lỗi:** `Image.open(p).verify()`, lỗi thì bỏ → 0 ảnh |
| 25–31 | **Bỏ ảnh trùng hoàn toàn:** tính mã MD5 của file; 2 file cùng MD5 = giống nhau từng byte → giữ 1 → bỏ 229 ảnh |
| 33–37 | **dHash (dấu vân tay ảnh):** thu ảnh về 17×16 xám, so mỗi pixel với pixel bên phải (sáng hơn hay không) → 16×16 = 256 bit. Ảnh giống nhau (dù khác độ tương phản, dịch nhẹ) có dHash gần giống |
| 39–51 | Tính số bit khác nhau giữa mọi cặp ảnh (khoảng cách Hamming). Cặp nào ≤ 4 bit thì nối vào cùng nhóm bằng **union-find** (hàm `find` tìm "gốc" của nhóm) |
| 53–62 | Gom ảnh theo nhóm. Mỗi nhóm giữ 1 ảnh; nhóm nào có cả 2 nhãn khác nhau (vd COVID và NORMAL) thì không biết nhãn nào đúng → bỏ cả nhóm → bỏ 819 ảnh gần trùng + 3 ảnh mâu thuẫn nhãn |
| 67–76 | **Chia tập:** riêng từng lớp, xáo trộn (seed 42) rồi cắt 70% train, 15% val, 15% test (chia phân tầng – stratified, để mỗi tập giữ tỉ lệ lớp) |
| 78–88 | Ghi `data/splits.csv` (ảnh nào thuộc tập nào) và `data/cleaning_summary.json` (số liệu) |

**Kết quả:**

| Bước | Số ảnh |
|---|---|
| Ảnh gốc | 5.228 |
| Ảnh lỗi | 0 |
| Trùng hoàn toàn (MD5) | 229 |
| Gần trùng (dHash ≤ 4 bit) | 819 |
| Trùng nhưng khác nhãn | 3 |
| **Còn lại** | **4.177** (COVID 1.077, NORMAL 1.425, PNEUMONIA 1.675) |

| Tập | COVID | NORMAL | PNEUMONIA | Tổng | Dùng để |
|---|---|---|---|---|---|
| Train | 754 | 997 | 1.172 | 2.923 | Cho model học |
| Validation | 162 | 214 | 251 | 627 | Chọn tham số, early stopping |
| Test | 161 | 214 | 252 | 627 | Chấm điểm cuối, chỉ dùng 1 lần |

### 2.4 Đọc ảnh và chuẩn hóa – `src/utils.py`

| Hàm (dòng) | Làm gì |
|---|---|
| `load_split` (66–83) | Đọc ảnh của 1 tập theo `splits.csv`: chuyển xám (`convert("L")`), resize (`224×224` hoặc `128×128` cho Simple CNN, nội suy bilinear), gom thành tensor `uint8` kích thước (N, 1, H, W). Lưu cache `data/cache_*.pt` để lần sau load nhanh |
| `XrayDataset.__getitem__` (53–60) | Mỗi lần lấy 1 ảnh: augmentation (nếu bật) → chia 255 về [0, 1] → (nếu model pretrained) nhân thành 3 kênh → chuẩn hóa |
| Chuẩn hóa (45–48) | CNN tự xây: `(x − 0,5) / 0,25`. MobileNetV3: mean/std của ImageNet `[0,485; 0,456; 0,406] / [0,229; 0,224; 0,225]` vì model được train sẵn với chuẩn hóa này |
| `make_loader` (86–89) | Chia ảnh thành batch (`DataLoader`); tập train thì xáo trộn (`shuffle=True`) và bật augmentation, val/test thì không |

**Vì sao chuẩn hóa:** đưa giá trị pixel về cùng thang, quanh 0 → gradient ổn định, train nhanh hơn.

### 2.5 Tăng cường dữ liệu (augmentation) – `train_augment` (utils.py 32–36)
Chỉ áp dụng cho **tập train**, mỗi epoch ảnh bị biến đổi ngẫu nhiên khác nhau:
- `RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1))`: xoay ±10°, dịch tối đa 5%, phóng to / thu nhỏ 90–110%.
- `ColorJitter(brightness=0.2, contrast=0.2)`: đổi độ sáng, độ tương phản ±20%.
- **Không lật ngang:** tim nằm bên trái, lật ngang tạo ra ảnh sai giải phẫu.

**Mục đích:** model thấy nhiều biến thể hơn → học đặc trưng tổng quát, giảm overfitting.

### 2.6 Mất cân bằng lớp – utils.py 99–102
COVID ít ảnh nhất (754 so với 997 và 1.172) → mỗi lớp được gán trọng số trong hàm loss:
`w = tổng số ảnh / (3 × số ảnh của lớp)` → COVID 1,29; NORMAL 0,98; PNEUMONIA 0,83.
Đoán sai ảnh COVID bị phạt nặng hơn → model không "lười" bỏ qua lớp ít ảnh.

---

## 3. Nội dung 2 – Simple CNN

### 3.1 Kiến trúc – `SimpleCNN` (models.py 6–23)

| Lớp | Cấu hình | Output | Tham số |
|---|---|---|---|
| Input | ảnh xám 128×128 | 1×128×128 | 0 |
| Conv2d + ReLU | 32 kernel 3×3, stride 1, không padding | 32×126×126 | 320 |
| Conv2d + ReLU | 32 kernel 3×3 | 32×124×124 | 9.248 |
| MaxPool2d | 2×2 | 32×62×62 | 0 |
| Flatten | | 123.008 | 0 |
| Linear + ReLU | 128 neuron | 128 | 15.745.152 |
| Dropout | p = 0,5 | 128 | 0 |
| Linear | 3 neuron (3 lớp) | 3 | 387 |
| **Tổng** | | | **15.755.107** |

**Cách tính:**
- Kích thước: `W_out = (W − K + 2P) / S + 1` → (128 − 3 + 0)/1 + 1 = 126 → 124 → MaxPool 2×2 → 62. Trong code: `size = (img_size − 4) // 2` (dòng 14).
- Tham số conv 1: (3×3×1 + 1) × 32 = 320. Conv 2: (3×3×32 + 1) × 32 = 9.248.
- Linear 1: 123.008 × 128 + 128 = 15.745.152 → **gần hết tham số nằm ở lớp Dense sau Flatten**.
- Lớp cuối **không có Softmax** trong code vì `nn.CrossEntropyLoss` của PyTorch đã tự tính softmax bên trong. Khi dự đoán thì gọi `torch.softmax` để ra xác suất (utils.py dòng 177).

### 3.2 Train – `src/2_simple_cnn.py`
1. **Dò hyperparameter (dòng 6–13):** thử lr ∈ {0,003; 0,001; 0,0003} × batch size ∈ {32; 64} = 6 lần, có augmentation, tối đa 25 epoch, chấm trên **tập val** (không đụng tập test).

| lr | batch | Val acc | Val F1 |
|---|---|---|---|
| 0,003 | 32 | 0,4003 | 0,1906 (không học được) |
| 0,003 | 64 | 0,9346 | 0,9374 |
| 0,001 | 32 | 0,9219 | 0,9236 |
| **0,001** | **64** | **0,9537** | **0,9558** ← chọn |
| 0,0003 | 32 | 0,9394 | 0,9418 |
| 0,0003 | 64 | 0,9426 | 0,9451 |

   lr 0,003 + batch 32 bị hỏng (chỉ đoán 1 lớp) vì lớp Dense 15,7 triệu tham số không ổn định với bước nhảy lớn.

2. **Train model chính (dòng 15–23):** lr 0,001, batch 64, tối đa 40 epoch, early stopping patience 8, train 2 bản: **không augmentation** (`simple_cnn`) và **có augmentation** (`simple_cnn_aug`) để so sánh tác dụng của augmentation.

### 3.3 Kết quả

| | Accuracy | F1 | Epoch tốt nhất / đã chạy | Thời gian |
|---|---|---|---|---|
| Simple CNN | 0,9490 | 0,9504 | 4 / 12 | 31 s |
| Simple CNN + Aug | 0,9474 | 0,9498 | 22 / 30 | 142 s |

Confusion matrix (hàng = nhãn thật, cột = đoán; thứ tự COVID / NORMAL / PNEUMONIA):
- Simple CNN: `[157 3 1] [3 202 9] [4 12 236]`
- Simple CNN + Aug: `[157 2 2] [2 206 6] [3 18 231]`

**Nhận xét:**
- Augmentation **không tăng điểm test** (gần như bằng nhau) nhưng **giảm overfitting**: không aug thì train loss về gần 0 còn val loss ngừng giảm từ epoch 4 (dừng ở epoch 12); có aug thì 2 đường cong sát nhau, còn cải thiện tới epoch 22.
- Model nhỏ về số lớp nhưng nặng về tham số vì Flatten nối thẳng vào Dense. Muốn nhẹ hơn: thêm conv/pooling hoặc dùng Global Average Pooling.

---

## 4. Nội dung 3 – Complex CNN

### 4.1 Ý tưởng
Ghép 2 loại khối nổi tiếng:
- **Khối Inception** (từ GoogLeNet) – `InceptionBlock` (models.py 34–44): 4 nhánh song song trên cùng input rồi **nối theo chiều kênh** (`torch.cat(..., dim=1)`):
  - b1: conv 1×1
  - b2: conv 1×1 (giảm kênh) → conv 3×3
  - b3: conv 1×1 (giảm kênh) → conv 5×5
  - b4: maxpool 3×3 (stride 1) → conv 1×1
  → nhìn chi tiết ở nhiều kích thước cùng lúc; conv 1×1 giảm số kênh trước conv lớn để bớt tham số.
- **Khối Residual** (từ ResNet) – `ResidualBlock` (models.py 47–60): 2 conv 3×3 có BatchNorm (`body`) + **đường tắt** (`shortcut`): `output = ReLU(F(x) + x)`. Khi số kênh / kích thước thay đổi thì đường tắt dùng conv 1×1 để khớp kích thước. Đường tắt giúp gradient truyền ngược dễ → train mạng sâu không bị suy giảm.
- `conv_bn` (models.py 26–31): Conv + **BatchNorm** + ReLU, dùng ở mọi nơi. `padding = k // 2` để giữ nguyên kích thước (với stride 1).

### 4.2 Kiến trúc – `ComplexCNN` (models.py 63–89)

| Phần | Cấu hình | Output | Tham số |
|---|---|---|---|
| Input | ảnh xám 224×224 | 1×224×224 | |
| **stem** | conv 7×7 stride 2 (32) → maxpool 3×3/2 → conv 1×1 (64) → conv 3×3 (96) → maxpool 3×3/2 | 96×28×28 | 59.296 |
| Inception A | nhánh 32 / 48→64 / 8→16 / 16 | 128×28×28 | 41.200 |
| Inception B | nhánh 64 / 64→96 / 16→32 / 32 | 224×28×28 | 91.232 |
| MaxPool | 3×3 stride 2 | 224×14×14 | 0 |
| Residual 1 | 224→256 (shortcut conv 1×1) | 256×14×14 | 1.164.800 |
| Residual 2 | 256→256 (shortcut identity) | 256×14×14 | 1.180.672 |
| Residual 3 | 256→384, stride 2 | 384×7×7 | 2.312.448 |
| Residual 4 | 384→384 | 384×7×7 | 2.655.744 |
| **head** | Global AvgPool → Dropout 0,4 → Linear 128 + ReLU → Dropout 0,3 → Linear 3 | 3 | 49.667 |
| **Tổng** | | | **7.555.059** |

- Số kênh ra của Inception = tổng kênh 4 nhánh: A = 32 + 64 + 16 + 16 = 128; B = 64 + 96 + 32 + 32 = 224.
- **Global Average Pooling** lấy trung bình mỗi feature map 7×7 → 1 số → vector 384 → lớp Dense rất nhỏ. Nhờ vậy Complex CNN sâu hơn nhiều nhưng ít tham số hơn Simple CNN.

### 4.3 Train – `src/3_complex_cnn.py`
Ảnh 224×224, có augmentation, lr 0,001, batch 32, tối đa 40 epoch, early stopping patience 8.

### 4.4 Kết quả
- Accuracy **0,9569**, F1 **0,9596**, epoch tốt nhất 26 / 34, 370 s.
- Confusion matrix: `[156 0 5] [1 209 4] [0 17 235]` → nhầm 5 ảnh COVID thành PNEUMONIA, 17 ảnh PNEUMONIA thành NORMAL.
- Tốt hơn Simple CNN (F1 0,9596 so với 0,9498) dù ít tham số hơn một nửa. Val loss dao động mạnh mấy epoch đầu → model lớn, dữ liệu ít.

---

## 5. Nội dung 4 – Transfer Learning / Fine-tuning (MobileNetV3)

### 5.1 Ý tưởng
Dùng **MobileNetV3-Large đã train sẵn trên ImageNet** (1,2 triệu ảnh, 1.000 lớp). Các lớp đầu đã biết nhận cạnh, góc, kết cấu → dùng lại được cho X-quang, chỉ cần dạy lại phần cuối. Phù hợp khi dữ liệu ít (2.923 ảnh train).

MobileNetV3 dùng **depthwise separable convolution** và khối **inverted residual** → nhẹ (4,2 triệu tham số) và nhanh.

### 5.2 Kiến trúc – models.py 92–103
- `mobilenet_transfer` (92–97): tải model với trọng số `IMAGENET1K_V2`, **đóng băng toàn bộ** (`requires_grad = False`), thay lớp cuối `classifier[-1]` thành `Linear(1280, 3)`.
- Cấu trúc: `features` (17 khối, ra 960×7×7) → Global AvgPool → classifier: Linear 960→1280 → Hardswish → Dropout 0,2 → **Linear 1280→3 (mới)**.
- Ảnh xám được **nhân thành 3 kênh** (utils.py dòng 59) và chuẩn hóa theo ImageNet vì model gốc nhận ảnh màu.
- `unfreeze_last_stages` (100–103): mở khóa **4 khối cuối** của `features` (3 khối inverted residual + conv 1×1 ra 960 kênh).

### 5.3 Train 2 giai đoạn – `src/4_transfer_learning.py` + utils.py 113–123
1. **Feature extraction (5 epoch đầu, `head_epochs=5`):** chỉ train lớp mới Linear 1280→3 (**3.843 tham số**), lr 0,001. Giai đoạn này không dừng sớm.
2. **Fine-tuning (từ epoch 6):** `train_model` gọi `unfreeze(model)` mở khóa 4 khối cuối, tạo optimizer mới với **lr nhỏ 0,0001** (`finetune_lr`), số tham số được train tăng lên **2.183.307**. Tối đa 30 epoch tổng, early stopping patience 8.

**Vì sao lr nhỏ khi fine-tune:** trọng số đã tốt sẵn, lr lớn sẽ phá hỏng những gì model đã học. **Vì sao chỉ mở tầng cuối:** tầng đầu học đặc trưng chung (cạnh, vân) dùng tốt cho mọi ảnh; tầng cuối đặc thù cho ImageNet nên cần chỉnh lại; dữ liệu ít nên mở hết dễ overfit.

### 5.4 Kết quả
- Accuracy **0,9761**, Precision 0,9780, Recall 0,9785, F1 **0,9781**, epoch tốt nhất 16 / 24, 278 s.
- Confusion matrix: `[160 1 0] [0 210 4] [0 10 242]` → đúng 160/161 ảnh COVID, không nhầm COVID ↔ PNEUMONIA; lỗi chính: 10 ảnh PNEUMONIA bị đoán NORMAL.
- **Tốt nhất trong 4 model**, cao hơn 2 CNN tự xây khoảng 2–3 điểm, vì đã học đặc trưng từ 1,2 triệu ảnh.

---

## 6. Train và đánh giá (dùng chung cho cả 3 phần)

### 6.1 Vòng train – `train_model` (utils.py 92–169)
PyTorch không có sẵn `model.fit()` như Keras nên tự viết:

| Dòng | Làm gì |
|---|---|
| 96–97 | Tạo DataLoader train (có aug, xáo trộn) và val |
| 99–102 | Hàm loss **CrossEntropy có trọng số lớp** |
| 103 | `GradScaler`: mixed precision trên GPU cho nhanh |
| 105–107 | Optimizer **AdamW** (Adam + weight decay 1e-4) + **ReduceLROnPlateau** (val loss không giảm 3 epoch → lr × 0,3) |
| 113–123 | Mỗi epoch: xác định giai đoạn (`full` / `head` / `finetune`); TL tới epoch `head_epochs` thì mở khóa tầng cuối, đổi lr nhỏ |
| 126–143 | Chạy tập train (`model.train(True)`: bật Dropout, BatchNorm cập nhật) rồi tập val (`train(False)`, không tính gradient). Mỗi batch: forward → loss → `zero_grad` → `backward` (tính gradient) → `optimizer.step` (cập nhật trọng số) |
| 145–148 | Cập nhật lr theo val loss, lưu lịch sử, in 1 dòng log |
| 150–156 | **Early stopping:** val loss tốt nhất từ trước tới giờ → lưu `best.pt`; không cải thiện `patience` epoch liền → dừng |
| 158–168 | Ghi `meta.json`, `history.json`; nạp lại `best.pt` (bản tốt nhất) |

### 6.2 Đánh giá – `predict`, `evaluate_model` (utils.py 172–197)
- `predict`: `model.eval()`, không tính gradient (`@torch.no_grad()`), chạy cả tập, `softmax` ra xác suất 3 lớp.
- Lớp đoán = lớp có xác suất lớn nhất (`argmax`).
- `evaluate_model`: tính bằng sklearn và lưu `metrics_test.json`:
  - **Accuracy** = số ảnh đúng / tổng số ảnh.
  - **Precision (lớp X)** = trong các ảnh đoán là X, bao nhiêu % đúng là X.
  - **Recall (lớp X)** = trong các ảnh thật là X, tìm ra bao nhiêu %.
  - **F1** = 2·P·R / (P + R).
  - **Macro** = tính cho từng lớp rồi lấy trung bình cộng (3 lớp quan trọng như nhau).
  - **Confusion matrix**: hàng = nhãn thật, cột = nhãn đoán, đường chéo = đoán đúng.

### 6.3 Phân tích chung
- **Nhầm COVID với Pneumonia?** Gần như không: recall COVID ≥ 0,96 ở mọi model, MobileNetV3 đúng 160/161.
- **Lớp khó nhất:** NORMAL ↔ PNEUMONIA. Lỗi hay gặp nhất: PNEUMONIA bị đoán NORMAL (10–18/252 tùy model) = âm tính giả, nguy hiểm nhất trong y khoa.
- **Grad-CAM** (6_figures.py): tô màu vùng ảnh ảnh hưởng nhiều tới quyết định (dùng gradient của lớp được đoán theo feature map lớp conv cuối). Ảnh đoán đúng thường sáng trong phổi; một số ảnh (nhất là ảnh sai) sáng ở cổ, bụng, cơ hoành → model đôi khi dựa vào chi tiết không liên quan tới bệnh.

### 6.4 Hạn chế
- Ảnh COVID có vẻ đến từ nguồn khác ảnh NORMAL / PNEUMONIA (kiểu chụp, chữ đánh dấu khác nhau); nhiều ảnh NORMAL / PNEUMONIA trông như ảnh trẻ em, nhưng dataset không có thông tin tuổi / nguồn nên không kiểm tra được từng ảnh → model có thể một phần học "nguồn ảnh" thay vì bệnh.
- Không có mã bệnh nhân → 2 ảnh khác nhau của cùng 1 người vẫn có thể rơi vào 2 tập khác nhau.
- Mỗi model chỉ train 1 lần với 1 seed → chênh lệch nhỏ giữa các model chưa chắc chắn.
- Chỉ là bài tập học thuật, không dùng chẩn đoán thật.

---

## 7. Lý thuyết – trả lời 15 câu

### Part 1. Basic CNN

**Câu 1. Mục đích của lớp tích chập (convolutional layer), output tính thế nào?**
- Mục đích: tự học các đặc trưng cục bộ của ảnh (cạnh, góc, vân → các cấu trúc lớn hơn ở lớp sâu). Nhờ **chia sẻ trọng số** (1 kernel dùng cho mọi vị trí) nên ít tham số hơn lớp fully connected và nhận ra đặc trưng ở bất kỳ vị trí nào.
- Cách tính: trượt kernel K×K trên input; tại mỗi vị trí nhân từng phần tử của kernel với vùng ảnh tương ứng, cộng lại, cộng bias → 1 giá trị của feature map. Nhiều kênh vào thì cộng kết quả của tất cả các kênh.
- Kích thước output: `(W − K + 2P) / S + 1`. Ví dụ code: 128×128, kernel 3, P = 0, S = 1 → 126×126.

**Câu 2. Mục đích của lớp pooling, max pooling hoạt động thế nào?**
- Mục đích: giảm kích thước feature map → giảm tính toán và tham số lớp sau, giúp model ít nhạy với dịch chuyển nhỏ, giảm overfitting. Pooling **không có tham số học**.
- Max pooling: chia feature map thành các ô (vd 2×2, stride 2), mỗi ô lấy **giá trị lớn nhất** → kích thước giảm một nửa. Ví dụ code: 124×124 → 62×62.

**Câu 3. Max pooling khác average pooling thế nào?**
- Max pooling lấy giá trị **lớn nhất** trong ô → giữ đặc trưng nổi bật nhất (phản ứng mạnh nhất của bộ lọc).
- Average pooling lấy **trung bình** → làm mượt, giữ thông tin chung của cả vùng.
- Global average pooling: lấy trung bình cả feature map → 1 số mỗi kênh (dùng trong Complex CNN, MobileNetV3, NiN, GoogLeNet).

**Câu 4. Padding và stride là gì?**
- **Padding (P):** thêm viền (thường là số 0) quanh input để giữ kích thước output và không bỏ mất thông tin ở biên. Kernel K lẻ, stride 1 thì P = (K−1)/2 giữ nguyên kích thước (code `conv_bn`: `padding = k // 2`).
- **Stride (S):** bước nhảy của kernel mỗi lần trượt. S = 2 → output giảm khoảng một nửa (code: conv 7×7 stride 2 đưa 224 → 112).

**Câu 5. Lớp conv có n ma trận input và 1 ma trận output cần bao nhiêu ma trận kernel thành phần?**
- **n** ma trận kernel (mỗi kênh input 1 kernel). Tích chập từng kênh với kernel riêng của nó rồi **cộng n kết quả** lại (+ bias) → 1 ma trận output.

**Câu 6. n ma trận input, m ma trận output cần bao nhiêu?**
- **n × m** ma trận kernel (mỗi output cần n kernel, có m output). Tensor trọng số có kích thước m × n × K × K. Ví dụ code: conv 2 của Simple CNN có n = 32, m = 32 → 1.024 kernel 3×3 (= 9.216 trọng số + 32 bias = 9.248 tham số).

**Câu 7. LeNet-5 là gì, thiết kế cho bài toán nào?**
- Một trong những CNN đầu tiên (LeCun và cộng sự, 1998), dùng để **nhận dạng chữ số viết tay** (đọc mã bưu điện, séc ngân hàng, bộ MNIST).
- Kiến trúc: conv 5×5 (6 kênh) → pooling → conv 5×5 (16 kênh) → pooling → 3 lớp fully connected (120 → 84 → 10), dùng sigmoid. Đặt nền móng cho kiểu "conv – pool – … – fully connected". Simple CNN của nhóm cũng theo kiểu này.

**Câu 8. Training set, validation set, test set khác nhau thế nào?**
- **Train:** dữ liệu để model học, cập nhật trọng số.
- **Validation:** đánh giá trong lúc train để chọn hyperparameter (lr, batch size), quyết định dừng sớm, chọn model tốt nhất. Model không học trên tập này.
- **Test:** dữ liệu model chưa từng thấy, chỉ dùng **1 lần ở cuối** để báo cáo kết quả khách quan.
- Nhóm chia 70 / 15 / 15 theo từng lớp, sau khi bỏ ảnh trùng để 3 tập không có ảnh chung.

### Part 2. Modern CNN

**Câu 9. Đặc điểm chính của AlexNet?**
- Thắng ImageNet 2012, mở đầu thời kỳ deep learning cho thị giác máy tính.
- 8 lớp: 5 conv + 3 fully connected, sâu và rộng hơn LeNet nhiều; conv đầu 11×11 stride 4.
- Dùng **ReLU** thay sigmoid (train nhanh, giảm vanishing gradient), **Dropout** ở lớp fully connected, **data augmentation** (lật, cắt, đổi màu), max pooling chồng lấn, train trên **GPU**.

**Câu 10. Khối VGG là gì?**
- Một dãy các **conv 3×3** (padding 1, giữ kích thước) + ReLU, kết thúc bằng **max pooling 2×2 stride 2** (giảm một nửa kích thước).
- Mạng VGG = xếp nhiều khối như vậy + fully connected. Ý tưởng: dùng nhiều conv 3×3 nhỏ xếp chồng thay vì 1 conv lớn (2 conv 3×3 nhìn được vùng 5×5 mà ít tham số hơn, thêm phi tuyến). Thiết kế mạng theo khối lặp lại, đơn giản, có quy luật.

**Câu 11. Khối NiN (Network in Network) là gì?**
- 1 conv thường (K×K) theo sau là **2 conv 1×1** (mỗi cái + ReLU). Conv 1×1 đóng vai trò như lớp fully connected áp dụng cho từng pixel → thêm phi tuyến theo chiều kênh.
- Mạng NiN **bỏ hẳn lớp fully connected cuối**: khối cuối có số kênh = số lớp, rồi dùng **global average pooling** → giảm mạnh số tham số, ít overfit.

**Câu 12. Khối Inception trong GoogLeNet là gì?**
- 4 nhánh song song trên cùng input: conv 1×1; conv 1×1 → conv 3×3; conv 1×1 → conv 5×5; max pooling 3×3 → conv 1×1. Kết quả 4 nhánh được **nối theo chiều kênh**.
- Ý tưởng: trích đặc trưng ở nhiều kích thước cùng lúc; conv 1×1 trước conv lớn để giảm số kênh → bớt tính toán.
- Complex CNN của nhóm dùng đúng khối này (`InceptionBlock`).

**Câu 13. Mục đích của batch normalization?**
- Chuẩn hóa output của 1 lớp theo từng mini-batch về trung bình 0, phương sai 1, rồi nhân / cộng với 2 tham số học được (γ, β) để model tự điều chỉnh.
- Lợi ích: train **ổn định và nhanh hơn**, dùng được learning rate lớn hơn, ít phụ thuộc khởi tạo, có tác dụng điều chuẩn nhẹ.
- Lúc train dùng mean / variance của batch; lúc dự đoán dùng giá trị trung bình trượt đã lưu (vì vậy phải gọi `model.eval()`).
- Complex CNN dùng BatchNorm sau mọi conv (`conv_bn`, `ResidualBlock`).

**Câu 14. Ý tưởng chính của ResNet?**
- **Khối residual:** thay vì học trực tiếp hàm H(x), mỗi khối học phần chênh lệch F(x) = H(x) − x, output = **F(x) + x** (đường tắt – skip connection).
- Nếu lớp thêm vào không cần thiết, model dễ dàng học F(x) ≈ 0 (giữ nguyên x) → mạng sâu hơn không bị kém đi. Gradient đi thẳng qua đường tắt → tránh vanishing gradient, train được mạng hàng trăm lớp.
- Khi kích thước / số kênh khác nhau, đường tắt dùng conv 1×1 để khớp. Complex CNN của nhóm dùng khối này (`ResidualBlock`).

**Câu 15. DenseNet khác CNN thông thường thế nào?**
- CNN thông thường: mỗi lớp chỉ nhận output của lớp ngay trước.
- DenseNet: trong 1 dense block, mỗi lớp nhận **output của tất cả các lớp trước đó, nối theo chiều kênh (concatenate)**, và output của nó đi tới tất cả các lớp sau.
- Khác ResNet: ResNet **cộng** (F(x) + x), DenseNet **nối** ([x, F(x)]).
- Dùng **transition layer** (conv 1×1 + average pooling) giữa các dense block để giảm số kênh và kích thước.
- Lợi ích: tái sử dụng đặc trưng, gradient truyền tốt, ít tham số cho cùng độ chính xác.

---

## 8. Các file trong repo

| File / thư mục | Nội dung |
|---|---|
| `src/1_preprocess.py` | Làm sạch + chia tập (mục 2.3) |
| `src/2_simple_cnn.py` | Dò lr/batch, train Simple CNN (mục 3) |
| `src/3_complex_cnn.py` | Train Complex CNN (mục 4) |
| `src/4_transfer_learning.py` | Train MobileNetV3 (mục 5) |
| `src/5_predict.py` | Nạp `best.pt`, dự đoán tập test hoặc 1 ảnh |
| `src/6_figures.py` | Vẽ hình (phân bố lớp, ảnh mẫu, augmentation, kiến trúc, đường cong train, confusion matrix, Grad-CAM) + `results_tables.md` |
| `src/models.py` | Định nghĩa các model (mục 3.1, 4.2, 5.2) |
| `src/utils.py` | Đọc ảnh, augmentation, vòng train, đánh giá (mục 2.4–2.6, 6) |
| `data/splits.csv` | 4.177 ảnh sạch, mỗi ảnh thuộc tập nào |
| `data/cleaning_summary.json` | Số ảnh bị loại ở mỗi bước |
| `data/cache_*.pt` | Ảnh đã resize sẵn (tự sinh, không lên git) |
| `results/<model>/best.pt` | Trọng số model tốt nhất (không lên git, nặng) |
| `results/<model>/metrics_test.json` | Kết quả trên tập test (số trong báo cáo lấy từ đây) |
| `results/<model>/history.json`, `meta.json` | Loss/acc từng epoch; cấu hình, thời gian, số tham số |
| `results/tune_simple_*` | 6 lần dò lr × batch size |
| `results/figures/` | Hình cho báo cáo |
| `results/train_log.txt` | Log lúc train |
| `pyproject.toml`, `uv.lock` | Danh sách thư viện (`uv sync` để cài) |

---

## 9. Câu cô có thể hỏi thêm về code

| Câu hỏi | Trả lời ngắn |
|---|---|
| Vì sao Simple CNN dùng ảnh 128 mà model khác 224? | Simple CNN Flatten thẳng vào Dense: ảnh 224 sẽ có 49,6 triệu tham số. Thử cả 2 cỡ thì kết quả gần bằng nhau nên chọn 128 cho nhẹ và nhanh |
| Vì sao lớp cuối không có Softmax? | `CrossEntropyLoss` đã gồm softmax; khi dự đoán mới gọi `torch.softmax` |
| `model.train()` và `model.eval()` khác gì? | train: bật Dropout, BatchNorm dùng thống kê của batch. eval: tắt Dropout, BatchNorm dùng thống kê đã lưu |
| `optimizer.zero_grad()` để làm gì? | Xóa gradient của batch trước, vì PyTorch cộng dồn gradient |
| `loss.backward()` / `optimizer.step()`? | backward: tính gradient bằng lan truyền ngược. step: cập nhật trọng số theo gradient |
| Early stopping chọn model nào? | Model ở epoch có **val loss thấp nhất** (lưu `best.pt`), không phải epoch cuối |
| Vì sao không dùng tập test để chọn lr? | Như vậy test không còn khách quan (chọn theo test = "học" trên test) |
| AdamW là gì? | Adam (lr thích nghi cho từng tham số, có momentum) + weight decay tách riêng → chống overfit |
| ReduceLROnPlateau là gì? | Val loss không giảm 3 epoch → giảm lr còn 30% |
| Dropout 0,5 nghĩa là gì? | Mỗi bước train tắt ngẫu nhiên 50% neuron → giảm phụ thuộc vào vài neuron, giảm overfit |
| Vì sao không lật ngang ảnh? | Tim nằm bên trái, lật ra ảnh sai giải phẫu |
| Class weight tính sao? | Tổng ảnh / (số lớp × số ảnh lớp đó): COVID 1,29, NORMAL 0,98, PNEUMONIA 0,83 |
| Fine-tune mở bao nhiêu tham số? | Giai đoạn 1: 3.843 (chỉ lớp cuối). Giai đoạn 2: 2.183.307 (4 khối cuối + lớp cuối) |
| Mixed precision (`autocast`, `GradScaler`) là gì? | Tính một phần bằng số thực 16 bit trên GPU cho nhanh, ít bộ nhớ; GradScaler phóng to loss để gradient nhỏ không bị mất |
| Seed 42 để làm gì? | Cố định ngẫu nhiên → chia tập và khởi tạo trọng số giống nhau mỗi lần chạy |
| Lưu và nạp model thế nào? | `torch.save(model.state_dict(), "best.pt")` lúc train; `model.load_state_dict(torch.load(...))` lúc predict |
