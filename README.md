# K4-Track4-Capstone-TeamSentinel

Mini project **RelateAnything + ByteTrack/Kalman**, giữ đồ vật qua che khuất ngắn.

Based on RelateAnything by Maëlic Neau — https://github.com/Maelic/RelateAnything

Luồng: detector mẫu → ByteTrack (Kalman) → RelateAnything → video và log quan hệ.
YOLO26 Nano mặc định cung cấp box; phần chính là relation model.
Không thêm RF-DETR riêng hoặc tối ưu các đồ khó detect.

## Cài đặt trên Windows

Cần Python 3.12+ và Git. Trong PowerShell tại thư mục dự án:

```powershell
./setup.ps1
```

Script tạo `.venv`, cài `requirements.txt` và RelateAnything tại commit
`197ac47d645f106e8beb79e49a3f62b91fcbe91f` bằng `--no-deps`.
Đây là bộ runtime của mini project, khác bộ pin tái lập training của upstream.
Lần chạy đầu cần mạng để tải weights YOLO và checkpoint RelateAnything.
Model và video đầu ra được giữ ở máy, không đưa lên GitHub.

Nếu tải từ Hugging Face bị đứng với Xet, có thể thử `$env:HF_HUB_DISABLE_XET='1'`
trước khi chạy. Nếu đã có model local, dùng
`--relation-model "C:/models/relsgg/model.pth"`; giữ `text_student.pt`, tokenizer,
calibration và các sidecar của upstream cùng thư mục checkpoint.

## Chạy

Webcam, video hoặc ảnh:

```powershell
./.venv/Scripts/python.exe demo.py --source 0
./.venv/Scripts/python.exe demo.py --source "C:/videos/demo.mp4" --output outputs/video-01
./.venv/Scripts/python.exe demo.py --source "C:/images/demo.jpg" --output outputs/image-01 --no-preview
```

Nhấn **Q** hoặc **Esc** để đóng preview. Dùng thư mục output mới cho mỗi lần chạy.

```powershell
./.venv/Scripts/python.exe demo.py --source 0 --hold-seconds 10 --relation-every 5 --threshold 0.3
```

- `--hold-seconds`: thời gian tối đa giữ track không được nhìn thấy, mặc định 10 giây.
- `--relation-every`: suy luận quan hệ mỗi N frame, mặc định 5; tăng nếu CPU chậm.
- `--device cpu`: chạy CPU; `--device cuda` nếu đã có PyTorch hỗ trợ GPU.
- `--max-frames 100`: chạy tối đa 100 frame để kiểm tra nhanh.
- `--no-preview`: chỉ ghi kết quả ra đĩa.

Kết quả: `annotated.mp4` hoặc `annotated.jpg`, `events.jsonl`, `summary.json`.
Log chứa ID, box, confidence detection, `observed`, quan hệ và score từ RelateAnything.
Không truyền `box_scores` vào relation model để giữ score quan hệ riêng;
không thêm fusion detection. Ngưỡng 0.3 chỉ là giá trị khởi đầu cho demo.

## Khi đồ bị che khuất

- ByteTrack ghép detection qua frame; Kalman dự đoán track đang chuyển động.
- Đồ gần như đứng yên giữ box cuối để hạn chế trôi vị trí.
- Box xám ghi `OCCLUDED`: track nhớ/dự đoán, không phải detection mới.
- Quan hệ cũ có nhãn `stale` và tuổi dữ liệu; không suy luận mới trên box dự đoán.
- Khi nhìn lại và ghép thành công, dùng lại ID và cập nhật quan hệ.
- Hết khoảng giữ thì bỏ track. ByteTrack không có appearance Re-ID nên có thể đổi ID
  khi che quá lâu hoặc nhiều đồ giao cắt.

Đây là demo quan hệ + tracking. Chưa có máy trạng thái xác nhận bỏ quên,
ownership bền vững hoặc hệ thống cảnh báo của PRD đầy đủ.
[Thiết kế thu gọn](docs/simple-demo-design.md) thay thế thiết kế SAOD cũ.

## Kiểm thử

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Test dùng ByteTrack/Kalman thật cho che khuất, chuyển động, hết hạn và tách lớp.
Test video dùng detector/relation giả để kiểm tra log và video I/O.
Xem [kết quả kiểm tra](docs/verification.md) để phân biệt với inference model thật.

Supervision 0.30.8 cảnh báo ByteTrack sắp được chuyển khỏi package; dự án khóa
phiên bản này vì truy cập cả lost tracks/Kalman. Không tự nâng lên 0.31 trước khi sửa adapter.

## Nguồn

- [RelateAnything](https://github.com/Maelic/RelateAnything)
- [ByteTrack qua Supervision](https://github.com/roboflow/supervision)
- [P-101](https://github.com/AI20K-Build-Phase-Cohort-4/P-101): tham khảo cách tách detector,
  tracking theo lớp và xuất log/video; không phụ thuộc đường dẫn P-101 khi chạy.

RelateAnything có giấy phép AGPL-3.0-only và điều khoản attribution trong NOTICE;
weights và detector có điều khoản riêng. Giữ attribution khi dùng lại demo.
