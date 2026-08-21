# Báo Cáo Lab Day 21 - CI/CD cho AI Systems

| | |
|---|---|
| Họ và tên | ___ |
| MSSV | ___ |
| Lớp / Khóa | K4 |
| Repo GitHub | https://github.com/___/___ |
| Ngày nộp | ___ |

## 1. Bộ Siêu Tham Số Đã Chọn và Lý Do

| Lần chạy | n_estimators | learning_rate | max_depth | f1_score | accuracy |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 0.1 | 3 | 0.7109 | 0.8780 |
| 2 | 50 | 0.05 | 2 | 0.6051 | 0.8460 |
| 3 | 200 | 0.1 | 5 | 0.7149 | 0.8740 |

**Bộ siêu tham số đã chọn:** `n_estimators=200`, `learning_rate=0.1`, `max_depth=5`.

**Lý do:** Bộ 200/0.1/5 cho F1 cao nhất (0.7149), vượt ngưỡng 0.65. Accuracy cao nhất là lần chạy 1 nhưng F1 vẫn thấp hơn lần chạy 3, vì vậy F1 phù hợp hơn để chọn mô hình. Khi giảm learning rate, thường cần tăng số cây để bù lại; bộ 50/0.05/2 vẫn chưa đạt yêu cầu.

## 2. Vì Sao Ngưỡng Chất Lượng Đặt Trên F1 Chứ Không Phải Accuracy

Chỉ khoảng 24,8% mẫu có thu nhập trên 50K, còn lớp thu nhập thấp chiếm đa số. Mô hình luôn dự đoán lớp thấp có thể đạt accuracy khoảng 0,752 nhưng F1 của lớp dương bằng 0, nên không có giá trị thực tiễn. F1 kết hợp precision và recall của lớp dương, đo được khả năng bắt đúng nhóm thu nhập cao. Vì vậy lab dùng `f1_score(y_eval, preds)` mặc định cho lớp 1, không dùng weighted hay macro để lớp đa số không che khuất chất lượng.

## 3. Khó Khăn Gặp Phải và Cách Giải Quyết

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| MLflow không import được | MLflow 2.13 dùng `pkg_resources`, setuptools 84 đã loại bỏ | Pin `setuptools<81` trong requirements và cài bản tương thích local. |
| DVC không cập nhật ngay | DVC bị vướng quyền ghi cache hệ thống và pytest cache | Loại pytest cache bằng `.dvcignore` và chạy lại `dvc add`; pointer đã cập nhật. |
| Chưa upload cloud/VM | Chưa có bucket, credentials và IP VM | Workflow đã dùng GitHub Secrets; push/deploy sẽ chạy sau khi cấu hình cloud thực tế. |

## 4. So Sánh Bước 2 và Bước 3

| | f1_score | accuracy |
|---|---:|---:|
| Bước 2 (chỉ `train_batch1`) | 0.7149 | 0.8740 |
| Bước 3 (thêm `train_batch2`) | 0.7354 | 0.8820 |

**Nhận xét:** Khi thêm `train_batch2`, F1 tăng 0.0205 và accuracy tăng 0.0080 trên cùng tập holdout. Kết quả này cho thấy dữ liệu bổ sung có ích trong thử nghiệm, đồng thời pipeline đã tự động đi từ cập nhật DVC pointer đến huấn luyện, quality gate và sản phẩm model mới.
