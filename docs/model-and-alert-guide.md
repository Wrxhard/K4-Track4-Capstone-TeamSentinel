# Tải model và chạy cảnh báo

## 1. Mở đúng thư mục

Trong PowerShell:

```powershell
cd "C:\Users\wrxha\Desktop\VinAI20K\Lab_Hackathon\K4-Track4-Capstone-TeamSentinel"
./setup.ps1
```

Nếu đã có `.venv` và các thư viện từ phiên trước, có thể bỏ qua setup.

## 2. Tải model về local

Bạn muốn tự tải rồi đặt vào thư mục: xem [models/README.md](../models/README.md)
có link từng file và cấu trúc chính xác. Sau khi đặt đủ file, bật webcam ngay:

```powershell
./start_camera.ps1
```

Lệnh này chạy offline với weights local, tự mở web và mặc định chờ chủ mất dấu 5 giây.
Tải bằng script bên dưới là lựa chọn thêm, không bắt buộc.

```powershell
./.venv/Scripts/python.exe download_models.py
```

Script tải checkpoint `maelic/relsgg-vits16plus` từ Hugging Face, phiên bản
`2db90096be5217bdc7a9003c042950f45723d105`. File chính model.pth khoảng 257 MB;
text encoder và sidecar tải thêm. Có thể chạy lại cùng lệnh nếu bị ngắt;
Hugging Face giữ cache tải để tiếp tục khi có thể. Không cần token cho repo public này.
Script đã tắt Xet và đặt download timeout 30 giây trước khi import Hub.

Kết quả phải có trong `models/relsgg-vits16plus/`:

- `model.pth`
- `text_student.pt`
- `calibration.json`
- `predicate_embeddings.npz`
- Các file `tokenizer*` do repository cung cấp.

YOLO26 Nano (`models/yolo26n.pt`) tải riêng để lấy box. RelateAnything không tự phát hiện box.

Nếu script vẫn tải chậm, mở [trang file checkpoint](https://huggingface.co/maelic/relsgg-vits16plus/tree/2db90096be5217bdc7a9003c042950f45723d105),
bấm download từng file ở danh sách trên rồi đặt chung vào `models/relsgg-vits16plus/`.
Không đổi tên và không dùng file `.part` hoặc `.incomplete` làm checkpoint.

## 3. Chạy trên video thật

```powershell
./.venv/Scripts/python.exe demo.py --source "C:/videos/test.mp4" --relation-model models/relsgg-vits16plus/model.pth --web --no-preview --output outputs/test-01
```

Mở **http://127.0.0.1:8080**. Thư mục output phải chưa tồn tại; đổi test-01 thành
test-02 cho lần chạy sau. Khi video kết thúc, web vẫn mở để xem lịch sử.
Nhấn Ctrl+C trong terminal để dừng server. Server chỉ bind 127.0.0.1.

Webcam:

```powershell
./.venv/Scripts/python.exe demo.py --source 0 --relation-model models/relsgg-vits16plus/model.pth --web --no-preview
```

CPU được chọn nếu không có CUDA; không cam kết realtime. Có thể tăng
`--relation-every 10` để giảm số lần chạy relation; lúc đó xác nhận ownership cũng chậm hơn.

## 4. Ngưỡng cảnh báo

```powershell
./.venv/Scripts/python.exe demo.py --source "C:/videos/test.mp4" --relation-model models/relsgg-vits16plus/model.pth --web --no-preview --exit-seconds 5 --unattended-seconds 15 --owner-threshold 0.5 --owner-samples 2 --output outputs/test-02
```

- Ownership: ít nhất owner-samples kết quả relation mới có tương tác mạnh, score
  >= owner-threshold. Nếu top hai người chênh < 0.15 thì chưa gán chủ.
- Không dùng near/beside/has để xác nhận chủ. Không đếm lại kết quả stale.
- Chủ mất dấu đủ exit-seconds, đồ còn được quan sát đứng yên: OWNER_LEFT_CAMERA.
- Chủ vẫn còn nhưng đồ đứng yên cách xa > 1.5 lần chiều cao box chủ đủ
  unattended-seconds: UNATTENDED_TIMEOUT.
- Người lạ còn trong hình không reset timer hoặc đổi chủ của đồ.
- Đồ bị che: giữ ownership và timer, chờ nhìn thấy lại mới phát cảnh báo mới.
- Chủ quay lại gần đồ: RESOLVED; lần bỏ lại tiếp theo có alert_id mới.
- Đồ chưa từng xác định chủ: UNASSIGNED; không khẳng định ai bỏ quên.

Một lần cảnh báo cho mỗi đợt. Modal có loại đồ, ID, chủ, score và thời gian video.
Nút **Đã xem** chỉ đóng modal trong tab hiện tại, không xóa incident; tải lại trang
sẽ hiển thị lại cảnh báo còn hoạt động. Nếu nhiều đồ có cảnh báo, modal lần lượt từng đồ.

Đầu ra: annotated.mp4, events.jsonl, alerts.jsonl, summary.json và evidence/.
Evidence lưu crop người chủ lúc xác nhận và frame cảnh báo; chỉ ghi đường dẫn nếu lưu thành công.
Thời gian cảnh báo dùng giây video hoặc monotonic time webcam, không phải UTC.

## 5. Test mà chưa cần model

```powershell
./.venv/Scripts/python.exe simulate.py
```

Mở http://127.0.0.1:8080: chủ mang balo trong 2 giây đầu rồi rời cảnh,
người thứ hai ở lại; khoảng giây thứ 5 có modal. Đây là dữ liệu mô phỏng, không phải
inference detector hoặc RelateAnything thật. Dùng `--port 8081` nếu cổng 8080 đang bận.
Test tự động:

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Giới hạn bản đầu

Mất detection của người chủ quá exit-seconds được coi là mất dấu, không chứng minh
người thực sự ra khỏi xe. Tăng thời gian này nếu góc camera hay che khuất người.
ByteTrack không có appearance Re-ID; đổi ID có thể gây báo sai hoặc không tự resolve.
Quan hệ holding/using là bằng chứng tương tác, không chứng minh quyền sở hữu thực tế.
Ngưỡng model cần điều chỉnh theo video và vocabulary; chưa đánh giá độ chính xác end-to-end.
Không tự chuyển giao chủ hoặc suy luận đồ bị lấy bởi người khác trong bản đơn giản này.
