# Realtime và GPU

Máy kiểm tra: NVIDIA GeForce GTX 1650, 4 GB VRAM, driver 576.80.
Môi trường ban đầu là torch 2.13.0+cpu: có GPU nhưng ứng dụng chưa dùng CUDA.

## Chạy GPU

```powershell
./setup_gpu.ps1
./start_camera.ps1 -Device cuda -FPS 30 -RelationEvery 10
```

setup_gpu.ps1 cài torch 2.11.0+cu128 + torchvision 0.26.0+cu128 từ index chính thức
vào .venv, không gỡ torch của Python hệ thống. Bộ CUDA Windows tải lần đầu khoảng
2.75 GB. Sau khi cài, script chạy phép nhân tensor CUDA để kiểm tra phần cứng thật.
Đây là runtime inference riêng; không phải bộ pin tái lập training của upstream.

Không cần cài CUDA Toolkit để dùng wheel này. Cần driver NVIDIA hỗ trợ runtime CUDA.
Nếu chỉ dùng CPU, chọn `-Device cpu`. Nếu chọn cuda mà không có GPU, ứng dụng báo lỗi;
không âm thầm chuyển sang CPU rồi hiển thị như thể GPU đang chạy.
Sau khi thay torch, tắt tiến trình demo cũ bằng Ctrl+C và chạy lại.

## Những điểm gây lag đã xử lý

- Web cũ đổi ảnh mỗi 700 ms: giới hạn khoảng 1.4 ảnh/giây. Preview mới dùng MJPEG
  /stream.mjpg; cập nhật khi có frame mới, tách khỏi polling metadata/modal.
- Webcam dùng buffer một frame trên thread capture. Inference chậm thì bỏ frame cũ,
  lấy frame mới nhất; không tích hàng đợi rồi phát hình quá khứ.
- Launcher yêu cầu camera 30 FPS; tốc độ thực tế phụ thuộc webcam và inference.
- Detector và relation vẫn chạy tuần tự trên GPU, phù hợp hướng dẫn upstream;
  không chạy hai model cạnh tranh GPU đồng thời. Relation mặc định mỗi 10 frame
  và chỉ chạy khi có người cùng đồ vật cá nhân.
- Giới hạn 16 box để giữ chi phí relation có giới hạn. Không đưa các class ngoài
  person/đồ vật cá nhân vào logic; ví dụ nhầm nhãn donut không tạo track đồ cần cảnh báo.
- Launcher không encode MP4 khi test live; log và ảnh cảnh báo vẫn được lưu.
- UI và frame có device, FPS đo theo thời gian xử lý hoàn thành thực tế và relation ms.
  FPS yêu cầu camera không được dùng để giả số FPS đo được.

## Box cũ chồng lên điện thoại/người

ByteTrack có thể đổi ID khi đồ đưa sát camera, thay đổi hình dạng hoặc confidence thấp.
Adapter cũ chỉ lấy track đã xác nhận, khiến box mới bị bỏ trong thời gian xác nhận,
trong khi box cũ vẫn vẽ như thể bị che khuất. Adapter mới giữ observation mới ngay
từ frame đầu bằng internal ID của ByteTrack, confidence khởi tạo khớp với ngưỡng detector.

Giữ track mất dấu trong logic nhưng không vẽ người đã mất dấu như một người đang hiện diện.
Box đồ mất dấu chồng >= 50% diện tích box nhỏ hơn với observation mới cùng lớp
được ẩn ở overlay; không tự gộp ownership hoặc đổi ID người dựa vào việc chồng box.
Box mất dấu có chữ LOST màu vàng, nền nhãn tối dễ đọc; không khẳng định mọi mất dấu
đều là che khuất. Track di chuyển mất dấu > 1 giây không còn vẽ bóng dự đoán.
Đồ đứng yên thật vẫn được giữ theo hold-seconds; chỉ phát cảnh báo mới khi thấy lại.

## Đối chiếu upstream

- [Deployment](https://github.com/Maelic/RelateAnything/blob/main/docs/deployment.md)
- [Demo webcam](https://github.com/Maelic/RelateAnything/blob/main/deploy/demo_webcam.py)
- [PyTorch CUDA](https://pytorch.org/get-started/locally/)

Upstream có TensorRT/torch.compile và benchmark trên GPU khác. Không lấy FPS của
A40/RTX 3080 làm FPS cam kết cho GTX 1650. Bản này giữ PyTorch inference để dùng
checkpoint người dùng đã tải; chưa cài TensorRT hoặc thay đổi độ chính xác score.
