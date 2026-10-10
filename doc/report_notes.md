# Ghi chú số liệu cho báo cáo giữa kỳ (nhóm 3)

> Đây là các con số, hình và quan sát rút ra từ code trong repo, sắp theo mẫu báo cáo của cô.
> Mọi con số lấy từ `results/` và `data/cleaning_summary.json`; nếu chạy lại thì cập nhật theo file mới.

## Part 1 – Theory (các ý cần có)

- Ảnh số = tensor C×H×W; vì sao MLP không phù hợp (số tham số, mất cấu trúc không gian).
- Phép tích chập: kernel, stride, padding, công thức kích thước đầu ra `(W − K + 2P)/S + 1`; chia sẻ trọng số, receptive field cục bộ.
- Hàm kích hoạt ReLU; Pooling (max / average / global average).
- Lớp fully connected + Softmax; hàm mất mát Cross-Entropy (nhóm dùng thêm class weight).
- Batch Normalization, Dropout, weight decay (chống overfitting).
- Tối ưu: SGD → Adam/AdamW, learning-rate scheduling (ReduceLROnPlateau), early stopping.
- Các kiến trúc kinh điển: LeNet-5, AlexNet, VGG, GoogLeNet (Inception), ResNet (residual), MobileNet (depthwise separable, inverted residual).
- Transfer learning: feature extraction (đóng băng backbone) vs fine-tuning.
- Grad-CAM (Selvaraju et al., 2017): trọng số kênh = trung bình gradient, heatmap = ReLU(Σ wₖ·Aₖ).

## Part 2 – I. Problem Description

- Title: Automated Chest X-ray Classification for Respiratory Diseases (COVID-19 / Pneumonia / Normal).
- Input: 1 ảnh X-quang ngực thẳng (PNG), chuyển grayscale, resize 224×224.
- Output: 1 trong 3 nhãn + xác suất softmax; kèm heatmap Grad-CAM.
- Các việc đã làm: làm sạch dữ liệu → chia 70/15/15 → augmentation → dò lr/batch size → 5 mô hình → đánh giá trên test → Grad-CAM.

## II. Dataset Description

- Link: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images
- Gốc: 5.228 ảnh PNG — COVID 1.626, NORMAL 1.802, PNEUMONIA 1.800.
- Kích thước: 232×232 (4.228 ảnh) và 256×256 (1.000 ảnh); RGB (1 ảnh RGBA) nhưng nội dung là ảnh xám.
- Hình: `results/figures/samples.png`, `results/figures/class_distribution.png`.

## III.1 Preprocessing

Làm sạch (`src/nhom3/data.py`, kết quả ở `data/cleaning_summary.json`):

| Bước | Số ảnh bị loại |
|---|---|
| File lỗi không đọc được | 0 |
| Trùng lặp hoàn toàn (cùng MD5) | 229 |
| Gần trùng (dHash 16×16, Hamming ≤ 4) — giữ 1 ảnh/nhóm | 819 |
| Nhóm gần trùng nhưng khác nhãn (loại cả nhóm) | 3 |
| **Còn lại** | **4.177** (COVID 1.077, NORMAL 1.425, PNEUMONIA 1.675) |

- Lý do quan trọng: nếu không loại ảnh gần trùng thì cùng 1 ảnh có thể vừa ở train vừa ở test → accuracy bị thổi phồng (data leakage). COVID có nhiều bản sao nhất (đã kiểm tra bằng mắt: cùng 1 ảnh, chỉ khác độ tương phản / dịch nhẹ).
- Chia phân tầng theo lớp, seed 42:

| Tập | COVID | NORMAL | PNEUMONIA | Tổng |
|---|---|---|---|---|
| Train | 754 | 997 | 1.172 | 2.923 |
| Validation | 162 | 214 | 251 | 627 |
| Test | 161 | 214 | 252 | 627 |

- Chuẩn hóa: grayscale, resize 224×224, chia 255; custom CNN: (x−0.5)/0.25; mô hình pretrained: nhân 3 kênh + mean/std ImageNet.
- Augmentation (chỉ train): xoay ±10°, dịch 5 %, scale 0.9–1.1, độ sáng/tương phản ±20 %. Không lật ngang (tim nằm bên trái, lật sẽ tạo ảnh sai giải phẫu). Hình: `results/figures/augmentation.png`.
- Mất cân bằng lớp (COVID ít nhất): dùng class weight nghịch đảo tần suất trong Cross-Entropy.

## III.2 Simple CNN — `arch_simple_cnn.png`

- 3 × (Conv → ReLU → MaxPool 2×2): 16 (5×5), 32 (5×5), 64 (3×3) filter → AdaptiveAvgPool 7×7 → FC 128 + Dropout 0.5 → FC 3. Khoảng 0,43 triệu tham số.
- Dò hyperparameter (validation, 25 epoch tối đa):

| lr | batch | Val acc | Val macro-F1 |
|---|---|---|---|
| 3e-3 | 32 | 0.9569 | **0.9586** |
| 3e-3 | 64 | 0.9537 | 0.9555 |
| 1e-3 | 32 | 0.9537 | 0.9562 |
| 1e-3 | 64 | 0.9442 | 0.9462 |
| 3e-4 | 32 | 0.9346 | 0.9375 |
| 3e-4 | 64 | 0.9250 | 0.9262 |

  → chọn lr 3e-3, batch 32. lr nhỏ hội tụ chậm hơn, batch 64 kém hơn batch 32 một chút ở mọi lr.

## III.3 Complex CNN — `arch_complex_cnn.png`

- Stem: Conv 7×7/2 + BN → MaxPool → Conv 1×1 → Conv 3×3 → MaxPool (96×28×28).
- 2 Inception block (4 nhánh 1×1 / 1×1→3×3 / 1×1→5×5 / MaxPool→1×1, nối theo kênh): 128 rồi 224 kênh → MaxPool (14×14).
- 4 Residual block (2 × 256, 2 × 384, block thứ 3 stride 2 → 7×7).
- Global AvgPool → Dropout 0.4 → FC 128 → Dropout 0.3 → FC 3. Khoảng 7,56 triệu tham số. lr 1e-3.

## III.4 Transfer Learning — `arch_resnet50_tl.png`, `arch_mobilenetv3_tl.png`

- ResNet-50 (ImageNet, 23,5 triệu tham số) và MobileNetV3-Large (4,2 triệu).
- Giai đoạn 1 (5 epoch): đóng băng toàn bộ backbone, chỉ train lớp FC mới, lr 1e-3.
- Giai đoạn 2: mở khóa layer4 (ResNet) / 4 block cuối (MobileNet), fine-tune lr 1e-4.
- Chung: AdamW, weight decay 1e-4, mixed precision, ReduceLROnPlateau (×0.3, patience 3), early stopping patience 8 theo val loss, giữ checkpoint tốt nhất.
- Phần cứng: RTX 3050 Laptop 6 GB.

## IV. Experimental Results (test set, 627 ảnh, macro average)

| Model | Accuracy | Precision | Recall | F1-score | Params | Thời gian train |
|---|---|---|---|---|---|---|
| Simple CNN | 0.9553 | 0.9557 | 0.9599 | 0.9577 | 0.43M | 62 s |
| Simple CNN + Aug | 0.9537 | 0.9565 | 0.9570 | 0.9566 | 0.43M | 264 s |
| Complex CNN + Aug | 0.9569 | 0.9605 | 0.9594 | 0.9596 | 7.56M | 370 s |
| ResNet-50 TL + Aug | 0.9729 | 0.9744 | 0.9761 | 0.9752 | 23.51M | 467 s |
| MobileNetV3 TL + Aug | **0.9761** | **0.9780** | **0.9785** | **0.9781** | 4.21M | 278 s |

Bảng theo từng lớp: `results/results_tables.md`. Hình: `training_curves.png`, `confusion_matrices.png`, `gradcam_resnet50_tl.png`, `gradcam_complex_cnn_aug.png`.

Quan sát để nhóm phân tích:

1. Transfer learning tốt hơn rõ (~+2 điểm F1) so với 2 CNN tự xây; MobileNetV3 tốt nhất dù ít tham số hơn ResNet-50 ~5,6 lần.
2. Augmentation không làm Simple CNN tốt hơn trên test (0.9566 vs 0.9577), nhưng đường cong cho thấy bản không aug overfit sớm hơn (train loss tiếp tục giảm, val loss đi ngang từ epoch ~10, dừng ở epoch 18).
3. Complex CNN chỉ hơn Simple CNN rất ít; val loss dao động mạnh (BN + dữ liệu ít) — gợi ý: 7,5M tham số train từ đầu với ~2.900 ảnh là nhiều.
4. **"Có nhầm COVID với Pneumonia không?"** — gần như không: COVID recall ≥ 0.97 ở mọi mô hình, ResNet-50 đúng 161/161. Chỉ Complex CNN nhầm 5 ảnh COVID → Pneumonia.
5. **"Lớp nào khó nhất?"** — NORMAL vs PNEUMONIA. Lỗi chủ yếu là Pneumonia bị đoán Normal (10–17 ảnh/252 tùy mô hình). Về y khoa đây là lỗi âm tính giả, nguy hiểm hơn dương tính giả.
6. **"CNN nhìn vào đâu?"** (Grad-CAM ResNet-50) — với ảnh đúng, heatmap phần lớn nằm trong vùng phổi (thâm nhiễm hai bên ở Pneumonia, tổn thương khu trú ở COVID). Nhưng có nhiều ảnh (kể cả ảnh đúng và hầu hết ảnh sai) heatmap rơi vào vùng bụng/cơ hoành, cổ hoặc góc ảnh → mô hình có thể dùng "đường tắt" (shortcut learning) như chữ đánh dấu, khung ảnh, kích thước bệnh nhân. Xem `samples.png`: ảnh NORMAL và PNEUMONIA là ảnh trẻ em cùng một kiểu chụp (có chữ "R"), còn COVID là ảnh người lớn từ nguồn khác (có dây điện cực, chữ khác). Nên việc COVID gần như đúng 100 % có thể một phần do model phân biệt *nguồn ảnh / độ tuổi* chứ không chỉ do tổn thương COVID — đây là điểm nên nói rõ trong báo cáo (xem DeGrave et al., 2021).
7. Hạn chế: dataset ghép từ nhiều nguồn, không có ID bệnh nhân (không loại trùng bệnh nhân được), chỉ 1 lần chạy/1 seed, test 627 ảnh nên chênh lệch ~0,3 % (2 ảnh) không có ý nghĩa thống kê. Kết quả không có nghĩa là dùng được cho chẩn đoán thật.
8. Hướng phát triển: phân đoạn phổi trước khi phân loại (giảm shortcut), nhiều seed / k-fold, dataset bên ngoài để kiểm tra, web demo.

## V. Conclusion — ý chính

- Làm đủ 5 yêu cầu: tiền xử lý (làm sạch 1.051 ảnh trùng/xung đột), Simple CNN, Complex CNN (Inception + Residual), Transfer learning (ResNet-50, MobileNetV3), đánh giá bằng Accuracy/Precision/Recall/F1/Confusion matrix + Grad-CAM.
- Tốt nhất: MobileNetV3 TL, F1 0.978 trên test.

## References (gợi ý, nhóm kiểm tra lại trước khi đưa vào)

1. Y. LeCun, L. Bottou, Y. Bengio, P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, 1998.
2. C. Szegedy et al., "Going deeper with convolutions," CVPR 2015.
3. K. He, X. Zhang, S. Ren, J. Sun, "Deep residual learning for image recognition," CVPR 2016.
4. A. Howard et al., "Searching for MobileNetV3," ICCV 2019.
5. S. Ioffe, C. Szegedy, "Batch normalization," ICML 2015.
6. N. Srivastava et al., "Dropout: a simple way to prevent neural networks from overfitting," JMLR 2014.
7. I. Loshchilov, F. Hutter, "Decoupled weight decay regularization," ICLR 2019.
8. R. R. Selvaraju et al., "Grad-CAM: Visual explanations from deep networks via gradient-based localization," ICCV 2017.
9. A. Zhang, Z. C. Lipton, M. Li, A. J. Smola, *Dive into Deep Learning* (d2l.ai) — nguồn các hình LeNet/GoogLeNet trong mẫu báo cáo.
10. A. J. DeGrave, J. D. Janizek, S.-I. Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal," Nature Machine Intelligence, 2021.
11. Dataset: S. Kumar, "COVID-19 Pneumonia Normal Chest X-ray images," Kaggle.
