# K4-Track4-Capstone-TeamSentinel

## Đề tài: Giám sát khoang hành khách VinBus / VinFast Smart Cabin
**Bài toán:** Hệ thống camera AI giám sát khoang xe thông minh — Tự động nhận diện chủ quyền đồ vật và cảnh báo hành khách bỏ quên tài sản (Lost & Found / Left Item Detection).
* **Đơn vị gắn liền trong hệ sinh thái Vin:** **VinBus** (Xe buýt điện thông minh) và **VinFast** (Smart Cabin / Xe điện dịch vụ Xanh SM).
* **Bối cảnh & Giá trị thực tế:**
  Hành khách đi xe buýt điện VinBus hoặc taxi điện VinFast thường mang theo các vật dụng cá nhân (balo, túi xách, điện thoại, laptop, ví tiền). Tình trạng để quên đồ khi vội bước xuống trạm/điểm đến diễn ra rất phổ biến, gây thất lạc tài sản cho khách hàng và tốn nhiều chi phí xử lý khiếu nại cho tổng đài.
  Hệ thống tận dụng **camera góc rộng trong khoang hành khách** để:
  1. Tự động nhận diện hành khách nào đang cầm/đeo/dùng đồ vật nào (`holding`, `carrying`, `wearing`, `using`) thông qua mô hình quan hệ thị giác (RelateAnything).
  2. Bám vết liên tục người và vật qua thời gian (ByteTrack + Kalman Filter), kiên cường trước việc người đi lại che khuất tầm nhìn.
  3. Kích hoạt cảnh báo tức thì lên màn hình khoang xe hoặc báo cho tài xế ngay khi hành khách bước xuống xe (chủ rời khỏi khung hình $\ge 3$ giây) mà đồ vật vẫn còn nằm lại trên ghế.

---

## Bảy trụ cột kiến thức Track 4 thể hiện trong Sentinel

Dự án tập trung chuyên sâu vào hai trụ cột cốt lõi: **THỜI GIAN** và **BẤT ĐỊNH**, cùng các minh chứng số đo thực nghiệm chặt chẽ:

### 1. THỜI GIAN (Trụ cột cốt lõi ⭐)
* **Khái niệm:** Đồng bộ cảm biến, tần số, độ trễ từ ảnh đến quyết định.
* **Thể hiện trong Sentinel:**
  * **Xử lý che khuất bằng Kalman Filter:** Trong khoang xe đông đúc, hành khách đi lại thường xuyên che khuất đồ vật. Bộ lọc Kalman duy trì quỹ đạo và vị trí dự đoán trong cửa sổ `hold_seconds = 10s` và cửa sổ nối track di chuyển 3 giây để không bị đứt đoạn nhận diện.
  * **Tách rời tần số xử lý (Decoupled Temporal Pipeline):** Camera và Detector chạy theo thời gian thực (30 FPS), trong khi mô hình quan hệ RelateAnything nặng hơn được phân bổ chạy chu kỳ mỗi 5 frame (`--relation-every 5`) nhằm giảm độ trễ pipeline và tối ưu tài nguyên tính toán.
  * **Bộ đếm thời gian thực cho quyết định:** Quyết định cảnh báo không dựa vào frame đơn lẻ mà kích hoạt theo bộ đếm thời gian: chủ nhân rời khỏi camera $\ge 3$ giây (`exit_seconds`), hoặc đồ vật đứng yên tách rời chủ $\ge 30$ giây (`unattended_seconds`).

### 2. BẤT ĐỊNH (Trụ cột cốt lõi ⭐)
* **Khái niệm:** Nhiễu cảm biến, độ tin cậy của đầu ra, khi nào nên nghi ngờ.
* **Thể hiện trong Sentinel:**
  * **Phân tầng ngưỡng tin cậy (Multi-stage Confidence):** Áp dụng ByteTrack 2 tầng: dùng box confidence cao để tạo track mới, và tận dụng box confidence thấp để duy trì track khi đồ vật bị mờ/nghiêng/khuất một phần mà không sinh ra track rác.
  * **Khử nhiễu gán chủ sở hữu (Ambiguity Resolution):** Để xác lập một người là chủ món đồ, hệ thống yêu cầu điểm tin cậy quan hệ $\ge 0.5$ tích lũy qua ít nhất 2 lần quan sát liên tiếp (`owner_samples = 2`). Nếu có từ 2 người cùng đạt điểm tương đương, hệ thống nghi ngờ và không vội gán chủ bừa bãi.
  * **Kiểm soát khi nghi ngờ (Doubt & Stale Handling):** Khi đồ vật bị che khuất (`observed = False`), quan hệ cũ bị đánh dấu `stale`; **tuyệt đối không chạy suy luận quan hệ mới trên box Kalman dự đoán**, và **khung cảnh báo đỏ `ABANDONED` chỉ hiển thị khi camera thực sự nhìn thấy đồ vật** (`observed = True`) để loại bỏ hoàn toàn cảnh báo ảo trôi nổi ngoài không gian.

### Các trụ cột hỗ trợ khác
* **BIỂU DIỄN:** Mã hóa thế giới thành Bounding Box, vector trạng thái động học Kalman `[x, y, a, h, vx, vy, va, vh]`, đồ thị quan hệ ngữ nghĩa `(Subject, Predicate, Object)` và máy trạng thái vòng đời vật thể (`UNASSIGNED` $\rightarrow$ `OWNED` $\rightarrow$ `OBSERVED / LOST` $\rightarrow$ `ABANDONED`).
* **BẰNG CHỨNG:** Đo lường FPS thực tế (CPU ~1.45 FPS, GPU NVIDIA GTX 1650 chạy RF-DETR CUDA đạt 19.9 – 23.4 FPS); kiểm thử tự động 41/41 unit tests; công bố rõ giới hạn hệ thống (ByteTrack chưa có Appearance Re-ID nên có thể đổi ID nếu bị che quá 10 giây).
* **HỌC:** Dual-model pipeline kết hợp Object Detection (RF-DETR ONNX / YOLOv8 học trên COCO) và Visual Relationship Detection (RelateAnything học trên tập quan hệ ngữ cảnh).
* **HÌNH HỌC:** Phân loại vật đứng yên dựa trên tỷ lệ dịch chuyển tâm chuẩn hóa theo chiều cao và delta time; tính khoảng cách Euclidean tương đối (`safe_distance = 1.5`) giữa tâm người và tâm vật.
* **DỮ LIỆU:** Ghi log chi tiết ra `events.jsonl` tách biệt detection thô và track; hỗ trợ replay chuỗi sự kiện để kiểm thử hồi quy không cần camera thật.

---

## Kiến trúc hệ thống
Mini project **RelateAnything + ByteTrack/Kalman**, giữ đồ vật qua che khuất ngắn.

Luồng: detector mẫu → ByteTrack (Kalman) → RelateAnything → video và log quan hệ.
Launcher webcam mặc định dùng RF-DETR ONNX local; có thể chọn YOLO bằng `-Detector`.
RelateAnything chấm quan hệ trên box của detector; ByteTrack/Kalman giữ lịch sử đồ vật.

Đã có logic bỏ quên theo từng đồ vật và **web local với modal cảnh báo**.
Hướng dẫn đầy đủ: [tải model và test cảnh báo](docs/model-and-alert-guide.md).

**Test webcam với model bạn tải:** đặt file theo [models/README.md](models/README.md),
rồi chạy `./start_camera.ps1`. Lệnh bật webcam thật, nạp model local và mở trang cảnh báo;
không cần đưa file qua form upload hoặc tải model lại khi chạy.

**GPU và preview realtime:** xem [hướng dẫn GPU](docs/realtime-gpu.md).
Launcher mới lấy frame webcam mới nhất, yêu cầu 30 FPS, dùng preview MJPEG
và không encode MP4 trong phiên test live. `./start_camera.ps1 -Device cuda` bắt buộc
dùng GPU; `auto` tự chọn nếu CUDA đã cài. Số FPS trên màn hình là số đo thực tế.

Thử modal ngay, không cần tải model:

```powershell
./.venv/Scripts/python.exe simulate.py
```

Mở http://127.0.0.1:8080. Đây là dữ liệu mô phỏng; sau khoảng 5 giây có cảnh báo
chủ rời đi trong khi người thứ hai vẫn còn.

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
snapshot ownership có score quan hệ và fused_score riêng. Ngưỡng 0.3 chỉ là giá trị khởi đầu cho demo.

## Khi đồ bị che khuất

- ByteTrack ghép detection qua frame; Kalman dự đoán track đang chuyển động.
- Đồ gần như đứng yên giữ box cuối để hạn chế trôi vị trí.
- Box vàng ghi `LOST`: vị trí cuối của đồ đứng yên, không phải detection mới.
- Ẩn box dự đoán của đồ đang di chuyển khi mất detection. Khung đỏ `ABANDONED`
  chỉ vẽ khi đang nhìn thấy đồ; lịch sử cảnh báo và ID vẫn được giữ qua mất dấu.
- Quan hệ cũ có nhãn `stale` và tuổi dữ liệu; không suy luận mới trên box dự đoán.
- Khi nhìn lại và ghép thành công, dùng lại ID và cập nhật quan hệ.
- Hết khoảng giữ thì bỏ track. ByteTrack không có appearance Re-ID nên có thể đổi ID
  khi che quá lâu hoặc nhiều đồ giao cắt.

Đã sửa nối lại ID sau che khuất: Kalman của đồ đứng yên được giữ tại box cuối;
chỉ phục hồi ID khi match cùng lớp, đủ chồng lấn và duy nhất theo cả hai chiều.
Vật đang di chuyển bị mất detection tối đa 3 giây được nối theo vị trí Kalman dự đoán;
khi nối lại, giữ ID, chủ sở hữu và score lịch sử để tiếp tục cảnh báo nếu chủ đã rời đi.
Box confidence thấp được giữ để nối track đã có, không dùng để xác lập ownership mới.
`events.jsonl` có cả detections và tracks để phân biệt detector bỏ sót với tracker mất dấu.

Đồ vật cần được xác lập chủ qua ít nhất 2 kết quả mới của holding/carrying/wearing/using,
score >= 0.5, và không có ứng viên ngang điểm. Quan hệ near/beside không gán chủ.
Sau đó chủ mất dấu >= 3 giây hoặc đồ đứng yên tách chủ >= 30 giây sẽ cảnh báo.
Người khác còn trong hình không chặn cảnh báo. Chủ cũ không tự bị thay bằng người gần nhất.
Đồ chưa có chủ chưa tự phát cảnh báo; box đang khuất không tạo cảnh báo mới.
Đây là rule demo, chưa phải ownership đã kiểm định; ByteTrack có thể đổi ID.
Xem [thiết kế thu gọn](docs/simple-demo-design.md) và phần mở rộng cảnh báo ở hướng dẫn.

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

RelateAnything có giấy phép AGPL-3.0-only và điều khoản attribution trong NOTICE;
weights và detector có điều khoản riêng. Giữ attribution khi dùng lại demo.
