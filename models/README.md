# Đặt model vào đây rồi bật webcam

```text
models/
├── rf-detr.onnx (RF-DETR ONNX, tối ưu cho CPU)
├── rf-detr.json (danh sách class của RF-DETR)
├── yolo26l.pt (khuyên dùng khi có GPU)
├── yolo26n.pt (bản nhẹ cho CPU)
└── relsgg-vits16plus/
    ├── model.pth
    ├── text_student.pt
    ├── tokenizer.json
    ├── tokenizer_config.json
    ├── calibration.json
    └── predicate_embeddings.npz
```

Detector lấy bounding box; RelateAnything chấm quan hệ trên các box đó.

## Link tải trực tiếp

- [yolo26l.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt) → `models/yolo26l.pt` (khuyên dùng khi có GPU)
- [yolo26n.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) → `models/yolo26n.pt` (bản nhẹ cho CPU)
- [model.pth](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/model.pth?download=true)
- [text_student.pt](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/text_student.pt?download=true)
- [tokenizer.json](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/tokenizer.json?download=true)
- [tokenizer_config.json](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/tokenizer_config.json?download=true)
- [calibration.json](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/calibration.json?download=true)
- [predicate_embeddings.npz](https://huggingface.co/maelic/relsgg-vits16plus/resolve/2db90096be5217bdc7a9003c042950f45723d105/predicate_embeddings.npz?download=true)

Tất cả file RelateAnything đặt chung trong `models/relsgg-vits16plus/`.
File chính khoảng 257 MB. File `.part`, `.incomplete` hoặc `.crdownload` chưa dùng được.
Giữ nguyên tên file; không cần clone repository hoặc train model.

## Bật camera thật

Trong PowerShell tại thư mục project:

```powershell
./start_camera.ps1
```

Lệnh kiểm tra file local, bật webcam số 0 và mở web http://127.0.0.1:8080.
Model được nạp từ đĩa; chế độ này không tự tải weights từ mạng.
Lần nạp CPU có thể chậm; đợi khung hình đầu tiên.

Thử trực tiếp: cầm balo/túi trong vùng camera cho tới khi sidebar có “Chủ #…”,
đặt đồ xuống rồi rời khỏi hình trong ít nhất 5 giây. Đồ phải còn được camera thấy.
Web sẽ bật modal; người khác đứng trong hình không tự chặn cảnh báo.
Nếu sidebar vẫn “Chưa xác định chủ”, model chưa trả đủ tương tác mạnh để gán chủ;
không phát cảnh báo OWNER_LEFT_CAMERA trong tình trạng này.

Webcam ngoài hoặc cổng web khác:

```powershell
./start_camera.ps1 -Camera 1 -Port 8081
```

Nếu chưa có thư viện, chạy `./setup.ps1` trước. Nhấn Ctrl+C để dừng camera.
Nếu camera không mở: đóng Teams/Zoom/app Camera đang chiếm webcam, kiểm tra quyền
Camera cho desktop apps trong Windows Settings, rồi thử index 0 hoặc 1.

Model tải local không được commit lên GitHub; chỉ README và thư mục mẫu được theo dõi.
