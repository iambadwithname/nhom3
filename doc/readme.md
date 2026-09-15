## 1. Bài toán

Đây là bài toán Automated Chest X-ray Classification for Respiratory Diseases
Link dataset: https://www.kaggle.com/datasets/sachinkumar413/covid-pneumonia-normal-chest-xray-images
Cho một ảnh X-quang phổi, mô hình CNN sẽ học các đặc trưng trên ảnh và dự đoán ảnh đó thuộc một trong 3 nhóm:
Normal – phổi bình thường
Pneumonia – viêm phổi
COVID-19 – COVID-19

## 2. Các công việc chính:

Thu thập và tiền xử lý dataset ảnh X-quang.
Chia dữ liệu thành training / validation / testing.
Xây dựng một CNN model để học đặc trưng hình ảnh.
Huấn luyện mô hình và điều chỉnh các hyperparameter.
Đánh giá bằng Accuracy, Precision, Recall, F1-score, Confusion Matrix.
Có thể thử Transfer Learning với ResNet/MobileNet để so sánh với CNN tự xây dựng.

## 3. Điểm hay của topic

Là bài toán trực quan và dễ hiểu, nhưng vẫn thể hiện được khá nhiều kiến thức của môn Deep Neural Network.
Nhóm không chỉ trả lời: “Model accuracy bao nhiêu?” mà có thể phân tích:
“Model có thường nhầm COVID-19 với Pneumonia hay không?”
“Class nào khó nhận diện nhất?”
“CNN thực sự đang nhìn vào vùng nào trên ảnh X-quang?”
Điều này giúp project có phần phân tích mô hình, thay vì chỉ train model rồi đưa ra accuracy.

## 4. Có thể phát triển ở đâu?

Project có thể phát triển theo nhiều mức, tùy thời gian của nhóm.
Cơ bản:
CNN → phân loại 3 class → đánh giá kết quả.
Nâng cao:
So sánh Custom CNN vs Transfer Learning như ResNet hoặc MobileNet.
Nên thử làm:
Sử dụng Grad-CAM để highlight vùng trên X-quang mà model tập trung khi đưa ra dự đoán. Đây là phần Explainable AI, giúp giải thích quyết định của CNN.

## 5. Vì sao chọn topic này?

Phù hợp với môn học: CNN được thiết kế đặc biệt để xử lý dữ liệu hình ảnh, nên topic thể hiện trực tiếp kiến thức Deep Learning.
Dataset phù hợp: Dataset có sẵn 3 class tương đối rõ ràng, quy mô vừa phải, phù hợp với nhóm chưa có nhiều kinh nghiệm.
Bài toán có tính thực tế: X-quang ngực được sử dụng trong việc hỗ trợ phát hiện các bệnh về phổi, nên bài toán có ý nghĩa ứng dụng.
Có khả năng mở rộng: Từ một CNN cơ bản có thể phát triển lên Transfer Learning, Explainable AI và web demo mà không cần thay đổi hoàn toàn bài toán.

## 6. Mục tiêu

Xây dựng và đánh giá một mô hình CNN có khả năng tự động phân loại ảnh X-quang ngực thành COVID-19, Pneumonia và Normal, đồng thời phân tích khả năng và hạn chế của mô hình.
