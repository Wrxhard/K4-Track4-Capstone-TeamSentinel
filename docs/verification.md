# Kết quả kiểm tra — 16/10/2026

Môi trường thực tế: Windows, Python 3.13.6, torch 2.13.0+cpu; không có CUDA.
Virtual environment local dùng system-site-packages sẵn có và cài riêng relsgg 1.0.0
từ source upstream commit `197ac47d645f106e8beb79e49a3f62b91fcbe91f` với `--no-deps`.
`setup.ps1` dành cho máy khác tạo venv riêng và cài các dependency được pin;
đã kiểm tra cú pháp và pip dry-run, chưa xác minh cài sạch trên máy thứ hai.

## Kiểm thử tự động

`python -m unittest discover -s tests -v`: 8/8 đạt.

- Mất detection rồi nhìn lại: giữ ID balo.
- Đồ đứng yên bị che: giữ box cuối, observed=false, không đưa vào relation inference mới.
- Đồ đang chuyển động: vị trí dự đoán Kalman tiếp tục thay đổi khi mất dấu.
- Hết hold timeout: bỏ track; đồ trở lại được cấp ID mới.
- Người và balo box chồng lấn: giữ hai ID khác nhau.
- Timestamp không tăng: từ chối.
- Video fixture: đọc và ghi đủ 6 frame, relation cũ gắn stale khi balo bị che,
  quan hệ được cập nhật sau khi balo xuất hiện lại.
- Không có quan hệ trả về: không tự tạo score 0; ảnh không có đồ vẫn được lưu.

Một số ca nằm chung trong cùng test method. Các test model/video dùng adapter giả;
tracker/Kalman là implementation thật của Supervision 0.30.8.

`ruff check demo.py sentinel.py tests`: đạt.
`python demo.py --help`: đạt.
`pip install --dry-run -r requirements.txt`: resolver hoàn tất, dependency có sẵn.

## Smoke test và kiểm tra hình ảnh

- Import `from relsgg import RelateAnything`: đạt.
- YOLO26 Nano thật trên ảnh horse.jpg của upstream: trả 2 box, horse và person.
- Fixture 20 frame: tạo `outputs/tracking-demo/annotated.mp4`, events.jsonl và summary.json;
  kiểm tra trực quan `occluded-frame.jpg` có box xám OCCLUDED và cạnh carrying gắn stale.
  Đây là dữ liệu tổng hợp với model giả, không phải độ chính xác model thật.

## Chưa xác minh

Inference RelateAnything thật chưa chạy xong: tải checkpoint Hugging Face bị đứng.
Đã thử Hugging Face Hub, tắt Xet và tải HTTP trực tiếp; chỉ có file tải dở.
Dừng các tiến trình thử tải, không báo thành công hoặc dùng score giả để thay thế.
Có thể chạy lại CLI sau khi tải được weights, hoặc dùng --relation-model trỏ tới model.pth local.
Không công bố FPS, độ chính xác relation/ownership hoặc tỷ lệ giảm false alarm.
Chưa thử webcam thực, preview tương tác và video người dùng.

Giới hạn phạm vi hiện tại: relation + tracking; chưa triển khai cảnh báo đồ bị bỏ quên.
