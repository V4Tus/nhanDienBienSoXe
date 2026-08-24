##Nhận diện biển số xe Việt Nam

#Dự án nhận diện biển số xe từ video bằng hai mô hình tự huấn luyện:

1.biensoxe.pt: mô hình YOLO phát hiện vị trí biển số.

2.plate_crnn_deploy.pth: mô hình CRNN đọc chữ và số trên biển số.

Hệ thống sử dụng ByteTrack để theo dõi biển số qua nhiều khung hình, crop vùng biển số, tổng hợp kết quả OCR và hiển thị kết quả lên video.

Phiên bản hiện tại chỉ xử lý file video, không sử dụng camera realtime.

#Chức năng

- Phát hiện biển số trong video.

- Theo dõi biển số bằng ByteTrack.

- Đọc biển số một dòng và hai dòng.

- Ổn định kết quả OCR qua nhiều khung hình.

- Lưu ảnh crop và thông tin biển số vào SQLite.

#Cấu trúc

├── main.py
├── ocr_crnn.py
├── biensoxe.pt
├── plate_crnn_deploy.pth
├── bienso.db
└── PICTURE/

#Công nghệ

Python, OpenCV, PyTorch, Ultralytics YOLO, CRNN, ByteTrack và SQLite.

#Tác giả
V4Tus
