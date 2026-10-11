# Giải thích code (nhánh main)

## uv.lock
File tự sinh bởi `uv`, khóa phiên bản thư viện để cả nhóm cài giống nhau, tránh conflict. Không cần quan tâm, không sửa tay.

## README.md
Trang giới thiệu project trên GitHub: bài toán, dataset, cách cài, 4 lệnh chạy.
Không xoá được vì `pyproject.toml` cần nó.

## pyproject.toml
Cấu hình project: tên, nhóm, thư viện cần cài, tạo lệnh `uv run nhom3 ...`.
Sửa xong chạy `uv sync`.

## .python-version
Báo cho `uv` dùng Python 3.13.

## .venv/
**Tự sinh ra khi chạy `uv sync`, không cần để tâm.**
Môi trường Python riêng của project (Python 3.13 + toàn bộ thư viện, ~4,5 GB).
Không đưa lên git, không sửa tay; xoá đi thì chạy `uv sync` là có lại.

- `etc/`, `share/`: file phụ vài thư viện tự thêm (cấu hình jupyter, tài liệu), không cần quan tâm.
- `Lib/`: chứa toàn bộ thư viện đã cài (torch, numpy, sklearn...), nặng nhất.
- `Scripts/`: file chạy được: `python.exe`, `nhom3.exe` (lệnh `uv run nhom3`), `activate` (bật môi trường).
- `.gitignore`: chứa `*`, bảo git bỏ qua cả thư mục.
- `.lock`: file khóa của `uv`, chống 2 lệnh sửa môi trường cùng lúc.
- `CACHEDIR.TAG`: đánh dấu đây là thư mục tạm để phần mềm backup bỏ qua.
- `pyvenv.cfg`: ghi môi trường dùng Python 3.13.12 nào, ở đâu.

## data/

### cache_*_128.pt, cache_*_224.pt (6 file) — tự sinh, không cần để tâm
- Ảnh đã resize sẵn rồi lưu lại, để các lần train sau khỏi phải đọc lại 4177 ảnh PNG → train nhanh hơn.
- `128` là cho Simple CNN, `224` là cho các model còn lại. Mỗi cỡ có 3 file: `train`, `val`, `test`.
- Tổng ~280 MB, không lên git (có trong `.gitignore`). Xoá đi thì lần chạy sau code tự tạo lại.

### cleaning_summary.json — cần nộp
- Kết quả bước làm sạch dữ liệu (lệnh `uv run nhom3 split`):
  - Ban đầu: 5228 ảnh (COVID 1626, NORMAL 1802, PNEUMONIA 1800)
  - Loại 0 ảnh lỗi, 229 ảnh trùng hoàn toàn, 819 ảnh gần trùng, 3 ảnh trùng nhưng khác nhãn
  - Còn lại: **4177** ảnh (COVID 1077, NORMAL 1425, PNEUMONIA 1675)
- Ghi luôn số ảnh từng lớp trong train (2923) / val (627) / test (627).
- Số liệu phần dataset trong báo cáo lấy từ file này.

### splits.csv — cần nộp
- Danh sách 4177 ảnh sạch, mỗi dòng: tên file, lớp, thuộc tập nào. VD: `COVID/COVID_10.png,COVID,train`.
- Mọi model đều đọc file này để lấy ảnh → tất cả train/test trên cùng một cách chia, ai chạy lại cũng ra giống nhau.

## src/

Code chính nằm trong `src/nhom3/`. Thứ tự chạy: `split` (data.py) → `tune`/`train` (models.py + train.py) → `figures` (figures.py).

### __pycache__/ (ở mọi thư mục) — tự sinh, không cần để tâm
Python tự lưu bản dịch sẵn của file `.py` để lần sau chạy nhanh hơn. Không lên git, xoá thoải mái.

### nhom3/__init__.py — "cửa vào" của chương trình
- Có hàm `main()`, chính là lệnh `uv run nhom3 <lệnh>` (khai báo trong `pyproject.toml`).
- 4 lệnh:
  - `split`: gọi `build_splits()` trong data.py → ghi `data/splits.csv` + `data/cleaning_summary.json`
  - `tune`: train 6 cấu hình Simple CNN (3 lr × 2 batch size), in kết quả trên tập val
  - `train`: train + chấm test 5 model (hoặc chỉ model ghi tên, vd `uv run nhom3 train resnet50_tl`)
  - `figures`: vẽ toàn bộ hình + bảng

### nhom3/data.py — dữ liệu
- `raw_dir()`: lấy thư mục ảnh gốc (dùng bản đã tải trong `~/.cache/kagglehub`, chưa có thì tải từ Kaggle).
- `build_splits()`: làm sạch + chia tập:
  1. bỏ ảnh lỗi không mở được
  2. bỏ ảnh trùng hoàn toàn (so mã MD5)
  3. bỏ ảnh gần trùng: thu ảnh về "dấu vân tay" 16×16 (`dhash`), 2 ảnh khác nhau ≤ 4 bit coi là một, giữ 1 (`group_duplicates` gom nhóm)
  4. nhóm trùng mà khác nhãn thì bỏ cả nhóm
  5. chia 70/15/15 theo từng lớp, seed 42
- `load_images()` / `get_split()`: đọc ảnh, chuyển xám, resize (224 hoặc 128), lưu cache `data/cache_*.pt`.
- `train_augment()`: augmentation cho tập train: xoay ±10°, dịch 5%, zoom 90–110%, đổi sáng/tương phản ±20%. Không lật ngang (tim nằm bên trái).
- `XrayDataset`: trả từng ảnh đã chuẩn hóa. CNN tự xây: 1 kênh, (x − 0.5)/0.25. ResNet/MobileNet: nhân thành 3 kênh, chuẩn hóa theo ImageNet.
- `make_loader()`: tạo DataLoader chia batch cho train/val/test.

### nhom3/models.py — 4 model
- `SimpleCNN` (của nhóm, chuyển từ notebook x-ray): ảnh 128×128 → Conv 32 → Conv 32 → MaxPool → Flatten → Dense 128 → Dropout 0.5 → Dense 3. ~15,8 triệu tham số.
- `ComplexCNN`: stem (các conv đầu) → 2 khối `InceptionBlock` (4 nhánh song song rồi ghép) → 4 khối `ResidualBlock` (đầu ra = F(x) + x) → Global AvgPool → FC. ~7,6 triệu tham số. `conv_bn` = Conv + BatchNorm + ReLU dùng chung.
- `resnet50_transfer`, `mobilenet_transfer`: lấy model đã train trên ImageNet, đóng băng hết, thay lớp cuối thành 3 đầu ra.
- `unfreeze_last_stages()`: mở khóa tầng cuối để fine-tune (layer4 của ResNet-50, 4 khối cuối MobileNetV3).
- `count_params()`: đếm số tham số.

### nhom3/train.py — train và đánh giá
- `Config`: cấu hình 1 lần chạy (tên, model, có aug không, lr, batch size, số epoch, cỡ ảnh...).
- `train()`: vòng train. Loss = CrossEntropy có trọng số lớp (COVID ít ảnh nên nặng hơn, `class_weights`), optimizer AdamW (`make_optimizer`), giảm lr khi val loss đứng yên, early stopping, lưu model tốt nhất `best.pt`. Với transfer learning: 5 epoch đầu chỉ train lớp cuối, sau đó mở khóa và fine-tune với lr nhỏ hơn.
- `evaluate()`: chấm trên val/test → accuracy, precision, recall, F1, confusion matrix, ghi `metrics_*.json` và `probs_*.npy`.
- `EXPERIMENTS`: 5 thí nghiệm chính (simple_cnn, simple_cnn_aug, complex_cnn_aug, resnet50_tl, mobilenetv3_tl).
- `TUNING`: 6 cấu hình dò lr × batch size cho Simple CNN.

### nhom3/figures.py — vẽ hình cho báo cáo
- Vẽ vào `results/figures/`: phân bố lớp, ảnh mẫu, ví dụ augmentation, sơ đồ kiến trúc 4 model, đường cong train, confusion matrix, Grad-CAM.
- `tables()`: tạo `results/results_tables.md` (bảng kết quả).
- `gradcam()`: tô màu vùng ảnh model nhìn vào khi ra quyết định.
- `make_all()`: chạy tất cả (lệnh `uv run nhom3 figures`).
## .gitignore
Những thứ không đưa lên git: `.venv`, `__pycache__`, cache ảnh, trọng số `best.pt`, file `.npy`, log.
