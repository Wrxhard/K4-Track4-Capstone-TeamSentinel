# Kết quả kiểm tra — 16/10/2026

## Khung cảnh báo trôi khi đồ không còn được nhìn thấy

Log local phiên 21:44 ghi phone #2 ở trạng thái ABANDONED, observed=false từ 23.43 giây;
tọa độ Kalman tiếp tục thay đổi. Lớp vẽ cảnh báo trước đây bỏ qua observed nên vẫn
vẽ khung đỏ trên box dự đoán, kể cả sau khi lớp box thông thường đã ẩn track.

Sửa: khung đỏ chỉ vẽ khi observed=true; box dự đoán của đồ đang di chuyển được ẩn
ngay từ frame mất dấu đầu. Giữ ID, ownership và lịch sử cảnh báo trong bộ nhớ.
Test video mô phỏng xác minh khung đỏ biến mất trong các frame 66–70 không có detection,
hiện lại khi đồ trở lại, giữ cùng object_id/alert_id và chỉ một sự kiện ALERT.
Kết quả: 41/41 unittest đạt; Ruff check và git diff --check đạt.

## Mất detection 1–2 frame rồi đổi ID liên tiếp

Tiến trình webcam mở lúc 21:33 chưa nạp bản sửa lưu lúc 21:41. Ngoài ra, replay đoạn
490–498 giây của log local phát hiện một nhánh còn lỗi: vật vừa di chuyển từ trạng thái
đứng yên và native track chưa xác nhận bị bỏ. Bản 4db310d sinh 7 ID phone trong đoạn này;
bản sửa tiếp giữ 1 ID, sử dụng cùng detection đã ghi, không chạy lại model.

Áp dụng cửa sổ nối 3 giây cho cả vật vừa di chuyển; giữ track ứng dụng ở observed=false
qua frame ByteTrack không còn native track, để bảo toàn owner và score. Test hồi quy
kiểm tra nhiều lần mất một frame, giữ owner, sau đó phát đúng một OWNER_LEFT_CAMERA.

## Điện thoại còn lại sau khi chủ rời khung hình

Log local `outputs/20261016-213326/events.jsonl` cho thấy phone #11 có owner #1,
nhưng khi di chuyển lúc chủ rời đi lại thành #14, UNASSIGNED. Không có score sở hữu
mới để gán chủ cho ID mới, nên không phát cảnh báo.

Phát lại detection và relation đã ghi ở đoạn 130–145 giây, ánh xạ endpoint relation
bằng box cùng lớp: bản trước không có alert; bản sửa giữ ID và phát đúng một
OWNER_LEFT_CAMERA tại 137.87 giây, sau khi chủ mất dấu từ 132.84 giây; score lịch sử
holding 0.6501 được giữ. Đây là replay kết quả model đã lưu, không chạy lại model/camera.

Sửa nối track chuyển động theo Kalman trong 3 giây, kiểm tra duy nhất hai chiều.
Test bao gồm giữ ID trong 3 giây, kết thúc cửa sổ nối chuyển động sau 3 giây, không
gộp hai điện thoại mơ hồ và giữ ownership để cảnh báo khi không có relation mới.
Kết quả: 37/37 unittest đạt; Ruff check và git diff --check đạt.

## Kiểm tra GPU và sửa mất ID sau che khuất

Đã kiểm tra thực torch 2.11.0+cu128 trên GTX 1650 4 GB; phép nhân tensor CUDA đạt.
RF-DETR ONNX khởi động bằng CUDAExecutionProvider với onnxruntime-gpu 1.20.2.

Benchmark ảnh horse.jpg upstream, YOLO26 Nano + RelateAnything mỗi frame:
CPU trước đó 1.45 FPS; GPU có các lần đo 5.0 và 3.85 FPS (10 iteration sau warmup).
Đây là benchmark ảnh cố định, không phải FPS webcam hoặc kiểm thử độ chính xác.
Tốc độ dao động theo trạng thái phần cứng; không công bố số cao nhất của sweep là cam kết.

Webcam thật với RF-DETR GPU: 90 frame trong 6.02 giây; cuối phiên window 30 frame
đạt 19.9 FPS. Cảnh đo không có đồ vật cá nhân, không có relation inference trong phiên này;
không dùng kết quả đó làm tốc độ full relation pipeline khi có nhiều đồ vật.

Sửa thêm: CPU dispatch threads cấu hình được; RF-DETR yêu cầu CUDA thì báo lỗi nếu
provider chỉ chạy CPU; giữ handle DLL search path; Kalman đồ đứng yên không trôi khi lost;
nối lại ID duy nhất và không lấy ID của object đang hiện diện; cho box yếu nối track cũ
nhưng không tạo track/ownership mới. Log thêm detections để chẩn đoán detector vs tracker.

Kiểm tra cuối sau bản sửa tracking: 33/33 unittest đạt; Ruff check và format check đạt.
Webcam RF-DETR CUDA chạy lại 90 frame trong 5.24 giây, cửa sổ cuối đạt 23.4 FPS;
relation_ms = 0 trong cảnh này, nên chưa đại diện tốc độ khi tính quan hệ người–vật.
Test tự động xác minh vật đứng yên bị che 9 giây rồi trở lại giữ ID, box yếu nối track cũ,
không lấy ID của vật đang thấy và không gộp hai ứng viên mơ hồ. Chưa tái hiện thao tác
che vật thực tế của người dùng trước webcam để đánh giá độ chính xác trên cảnh đó.

Phần phía dưới là các báo cáo lịch sử, trước khi xác minh GPU/model thật.

## Cập nhật cảnh báo + web + webcam launcher

Kết quả cuối: 19/19 unittest đạt; Ruff check và format check đạt.
start_camera.ps1 qua kiểm tra cú pháp và chạy thử trường hợp thiếu model:
báo đúng các file local còn thiếu, không mở camera hoặc âm thầm tải model.
yolo26n.pt đã được đặt ở models/ trên máy local; weights vẫn được gitignore.

Đã thêm ownership dựa trên kết quả relation mới, trạng thái ATTACHED/UNATTENDED/ABANDONED,
OWNER_LEFT_CAMERA/UNATTENDED_TIMEOUT, RESOLVED, chống lặp alert, evidence crop/frame,
dashboard HTTP local, modal và start_camera.ps1 nhận model local.

Kiểm thử mới gồm: chủ rời nhưng người khác còn lại; không cảnh báo trên box khuất;
chủ quay lại resolve; low/near/stale relation không gán chủ; hai ứng viên mơ hồ;
timeout khi chủ còn trong hình; đồ chuyển động/không biết chủ; HTTP page/status/frame;
video hai người phát đúng 1 alert và lưu evidence; camera index 0 và giải phóng capture.

Đã mở web bằng Chrome: modal hiển thị backpack #1, chủ #3, score 0.90,
OWNER_LEFT_CAMERA tại 5.0 giây trong scenario mô phỏng; bấm Đã xem đóng modal,
trạng thái Bỏ quên vẫn còn. Không dùng model thật cho scenario này.

Model thật end-to-end và webcam vật lý vẫn chưa xác minh vì checkpoint chưa tải đủ.
start_camera.ps1 kiểm tra đủ file trước khi mở camera, dùng offline mode.
Phần phía dưới là báo cáo của bản trước, trước khi có cảnh báo/web.

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
