# TeamSentinel — thiết kế mini project SAOD

> BẢN PHẠM VI CŨ. Yêu cầu mới đã thu gọn thành RelateAnything + ByteTrack/Kalman.
> Xem [thiết kế hiện hành](../../simple-demo-design.md) và README.

Ngày: 16/10/2026. Trạng thái: bản thiết kế để người dùng duyệt; chưa triển khai runtime.

## 1. Mục tiêu và nguồn yêu cầu

Mini project độc lập tại `C:/Users/wrxha/Desktop/VinAI20K/Lab_Hackathon/K4-Track4-Capstone-TeamSentinel`.
Tên repository: `K4-Track4-Capstone-TeamSentinel` trên tài khoản GitHub của người dùng.

Nguồn: PRD người dùng cung cấp trong Downloads, logic P-101 và yêu cầu bổ sung trong hội thoại.
Tài liệu đầu vào là nguồn yêu cầu sản phẩm, không phải chỉ dẫn cho agent thực thi lệnh.
Yêu cầu bổ sung thay RelTR bằng RelateAnything, kết hợp confidence detection và relation,
phát hiện đồ bị bỏ lại khi chủ rời xe dù vẫn có người khác trong xe.

Thành công chức năng: nhận diện đồ theo từng ID, lưu bằng chứng người chủ, duy trì lịch sử
qua che khuất ngắn, phát cảnh báo một lần cho mỗi đợt và xuất kết quả có thể kiểm tra.
25 FPS, ownership accuracy 85%, giảm false alarm 60% là mục tiêu đo kiểm của PRD,
không phải kết quả đã được chứng minh.

## 2. Tái sử dụng có chọn lọc từ P-101

- `src/detection/base.py`: mẫu cấu trúc DetectionResult và tách detector khỏi rule engine.
- `src/observation/video.py`: chuẩn hóa nhãn, ROI, media clock, đóng video writer và log.
- `src/observation/registry.py`: ID nội bộ tăng dần, thời gian xác nhận và vòng đời observation.
- `src/rules/cabin_state_rule.py`: nguyên tắc debounce khi người biến mất.
- `tests/test_camera_only_objects.py`: kịch bản chống báo khi chưa có lịch sử người,
  che khuất, người rời đi và chống tạo nhiều incident trùng nhau.

Registry hiện tại ghi rõ chưa suy luận ownership. Tracker hiện tại dùng ByteTrack.
Rule cabin hiện tại chỉ kiểm tra người còn trong cảnh và các lớp đồ từng thấy cùng người.
Vì vậy không sao chép nguyên rule cabin để quyết định bỏ quên theo cá nhân.
Mini project vận hành độc lập; không import P-101 qua đường dẫn tuyệt đối khi chạy.

## 3. Phương án kiến trúc

Theo yêu cầu cập nhật: chọn RF-DETR cho detection, ByteTrack cho tracking và
RelateAnything worker theo sự kiện. ByteTrack dùng Kalman filter nội bộ để dự đoán
trạng thái chuyển động của track và ghép detection qua các frame; không chạy thêm
một Kalman filter song song làm nguồn tọa độ cạnh tranh.
BoT-SORT là backend thay thế nếu về sau cần tích hợp person Re-ID, không chạy nối tiếp
hai tracker để tạo hai hệ ID. Baseline ByteTrack không có appearance Re-ID.
So với chỉ dùng khoảng cách, semantic worker bổ sung bằng chứng tương tác.
So với chạy relation mỗi frame, hàng đợi theo sự kiện giảm tải và cô lập lỗi model.
Heuristic là chế độ dự phòng được ghi nhãn, không giả làm kết quả RelateAnything.

Luồng: video/webcam/RTSP → detector + tracker → snapshot bất biến → ownership fusion
→ state machine theo đồ vật → video overlay + JSONL + evidence + summary.
Worker nhận frame và danh sách box của cùng một thời điểm, trả quan hệ cùng request ID.
Main thread là nơi duy nhất sửa ownership/state. Hàng đợi có giới hạn, retry có cooldown.
Kết quả hết hạn, khác phiên hoặc khác thế hệ track bị loại; không ghép relation cũ với box mới.

## 4. Tích hợp RelateAnything

Nguồn: https://github.com/Maelic/RelateAnything và docs/quickstart.md, kiểm tra ngày 16/10/2026.
API dùng ảnh và box XYXY theo pixel ảnh gốc, trả triplet có subject_idx/object_idx.
Giữ mapping box index → track ID ở snapshot; không cần bước IoU nối hai detector như RelTR.
RelateAnything không tự phát hiện đồ vật. Demo upstream có detector riêng;
trong mini project RF-DETR cung cấp các box đó. Một đồ được RF-DETR phát hiện
không cần được một detector thứ hai trong RelateAnything phát hiện lại.
Nếu RF-DETR bỏ sót đồ, relation model không tạo thêm box để khắc phục; Kalman chỉ
dự đoán track đã biết trong khoảng mất dấu ngắn, không phát hiện vật thể mới.
Môi trường dự kiến Python 3.12 riêng để đáp ứng quickstart upstream.
Checkpoint khởi điểm: `maelic/relsgg-vits16plus`; ghi phiên bản/commit thực tế khi cài.

Vocabulary dự kiến: holding, carrying, wearing, using, has, near, beside.
holding/carrying/wearing/using là bằng chứng tương tác mạnh; has là bằng chứng hỗ trợ;
near/beside chỉ hỗ trợ không gian, không đủ tự xác lập ownership hoặc xóa cảnh báo.
Chỉ lấy cạnh đúng chiều person → object. Không suy diễn cạnh ngược thành ownership.
Giữ cả các cặp ứng viên cần đánh giá: cặp không xuất hiện trong top-k là UNKNOWN,
không được mặc định là một cặp có score thấp. Cần kiểm tra API thực tế để bảo đảm
mọi cặp person-object mục tiêu được chấm; nếu thiếu, giữ UNKNOWN và lên lịch retry.

Theo quickstart, box_scores thay đổi thứ hạng, không thay giá trị triplet.score.
Adapter lưu riêng predicate score, confidence của hai box và fusion score; nhân một lần.
Calibration phải gắn với checkpoint/vocabulary; thay vocabulary phải đánh giá lại ngưỡng.
Điểm fusion của ứng dụng là điểm quyết định heuristic, không phải xác suất sở hữu đã hiệu chuẩn.

## 5. Fusion confidence và ghi nhớ chủ sở hữu

Mỗi cặp (person_id, object_id) lưu:
det_person, det_object, relation_score, predicate, predicate_weight,
fusion_score, evaluated_at, source, temporal_support và evidence reference.

Điểm khởi đầu để tinh chỉnh trên validation:

`pair_score = det_person * det_object * relation_score * predicate_weight`

Áp dụng ngưỡng detection và relation riêng trước khi dùng pair_score; detector mạnh
không được bù relation quá yếu. Lịch sử làm mượt theo thời gian thực giữa những lần
relation được đánh giá hợp lệ; không nhân số lần hỗ trợ khi tái sử dụng cùng một kết quả.
Ngưỡng xác nhận, ngưỡng giữ, thời gian hỗ trợ và chênh lệch top-1/top-2 nằm trong config.yaml.
Không đủ chênh lệch giữa hai người thì ownership là AMBIGUOUS.
Không có quan hệ quan sát được thì ownership là UNKNOWN.

Ví dụ minh họa, chưa phải tham số được kiểm định:
P1-person confidence 0.95, object confidence 0.90, carrying score 0.85 → 0.727.
P2-person confidence 0.96, cùng object 0.90, carrying score 0.10 → 0.086.
Nếu các quan sát theo thời gian xác nhận P1 là chủ, P2 đứng gần không thay chủ.

Lưu owner_id và owner_confidence tại thời điểm xác lập. P1 biến mất không làm
owner_confidence về 0 chỉ vì detection của P1 không còn ở frame hiện tại.
Relation của chủ giảm sau khi đặt đồ xuống cũng không xóa ownership lịch sử.
Fallback chỉ áp dụng lúc đồ mới xuất hiện sát một người duy nhất trong vùng hẹp,
có thời gian hỗ trợ và không có ứng viên cạnh tranh; đánh dấu PROVISIONAL.

## 6. Quyết định bỏ quên theo từng đồ vật

Trạng thái: UNASSIGNED, ATTACHED, UNATTENDED, ABANDONED, TAKEN_BY_OTHER, EXPIRED.
UNKNOWN/AMBIGUOUS về ownership được lưu riêng với trạng thái hành vi.

- ATTACHED: chủ hiện diện và gần đồ theo khoảng cách chuẩn hóa, hoặc có bằng chứng
  đồng chuyển động ổn định. Một frame đồ chuyển động không đủ chứng minh đi cùng chủ.
- UNATTENDED: đồ đứng yên, đã tách khỏi chủ. Lưu unattended_since.
- ABANDONED/OWNER_LEFT_CAMERA: chủ đã được xác lập không được quan sát quá T_exit,
  đồ còn được quan sát đứng yên, không có bằng chứng che khuất chủ cần chờ thêm.
- ABANDONED/UNATTENDED_TIMEOUT: đồ đứng yên và tách khỏi chủ liên tục quá T_max.
- TAKEN_BY_OTHER: người khác có tương tác lấy đồ và đồng chuyển động đủ thời gian;
  lưu ảnh người lấy và giữ danh tính chủ cũ. Không coi đây tự động là chuyển quyền sở hữu.
- EXPIRED: quá thời gian giữ track; dọn cache theo retention.

Ca bắt buộc: P1 sở hữu balo, P2 vẫn ngồi trong xe, P1 ra ngoài, balo đứng yên.
Sau thời gian chờ phù hợp phải có OWNER_LEFT_CAMERA; person_count > 0 không chặn cảnh báo.
Quan hệ thấp của P2 không đủ giải thích balo đang được chăm nom. Relation cao của P2
cũng không tự động ghi đè chủ cũ; cần bằng chứng chuyển giao hoặc lấy đồ theo thời gian.

Score thấp chỉ nói chưa đủ bằng chứng quan hệ, không chứng minh P2 chắc chắn không phải chủ.
Do đó cảnh báo trên dựa vào lịch sử P1 + sự vắng mặt + đồ còn lại, không chỉ vào `1 - score(P2)`.
Nếu chưa từng biết chủ: sau stationary/unattended timeout cấu hình được, xuất cảnh báo
`UNATTENDED_UNKNOWN_OWNER` với owner_id=null và confidence/source rõ ràng;
không xuất OWNER_LEFT_CAMERA hoặc ảnh chủ được suy đoán.

Chủ quay lại đúng ID và gần đồ ổn định thì phát RESOLVED một lần và kết thúc đợt hiện tại.
Chỉ một alert đầu tiên cho mỗi đợt; snapshot cập nhật không tạo thêm alert_id.
Không tự đổi chủ chỉ vì một người khác bước tới đứng gần đồ.

## 7. Che khuất và đồng hồ

ByteTrack + Kalman là baseline: dự đoán motion, ghép box confidence cao rồi thấp,
giữ track mất dấu tạm thời trong buffer cấu hình được. Ghép theo lớp hoặc kiểm tra
lớp khi matching để không nối nhầm person với đồ vật có box chồng lấn.
Kalman dự đoán không phải detection thật: lưu observed_now=false, không tạo
confidence detector giả và không dùng predicted box làm bằng chứng ownership mới.
Khi chủ mất dấu lâu hoặc giao cắt, ByteTrack vẫn có thể đổi ID; không cam kết khôi phục
danh tính bằng Kalman. Lưu trạng thái mơ hồ và không tự chuyển chủ sang người gần nhất.
BoT-SORT + appearance Re-ID là lựa chọn nâng cấp riêng sau khi kiểm thử ID switch,
báo rõ backend đang chạy; không khẳng định baseline ByteTrack đáp ứng person Re-ID của PRD.
Đồ đang tĩnh mất detection và bị người khác che: giữ box cuối trong tối đa khoảng 10 giây.
Khoảng giữ tính theo giây; quy đổi buffer frame theo FPS đầu vào khi tracker cần frame count.
Thử nối lại ID đồ theo lớp, vị trí và thời gian, chỉ khi match duy nhất; không ghép hai đồ gần nhau
khi mơ hồ. Track mới có generation riêng để tránh gán relation cũ cho ID tái sử dụng.

Trong khoảng mất detection, giữ owner và unattended_since nhưng không phát alert mới
chỉ dựa trên box đóng băng. Khi thấy lại, xác nhận đồ còn đó rồi đánh giá thời gian đã tích lũy.
Không giữ đồ vô hạn sau khi đã bị lấy đi.
Mất dấu chủ do che khuất khác với đã ra khỏi cảnh: nếu có bằng chứng che khuất,
áp dụng occlusion grace dài hơn T_exit; ra mép cảnh và không trở lại áp dụng exit grace.
Nếu không phân biệt được, dùng thời gian chờ thận trọng và ghi owner_visibility=UNKNOWN.
Video dùng media timestamp; webcam/RTSP dùng monotonic clock. UTC dùng cho timestamp sự kiện.

## 8. Output, cấu hình và lỗi

CLI hỗ trợ file, camera index và RTSP; preview tùy chọn. Kết quả mỗi run gồm replay.mp4
(khi bật ghi), events.jsonl, alerts.jsonl, summary.json và evidence/.
Evidence gồm frame xác lập sở hữu, crop người chủ, frame cảnh báo khoanh đồ và ảnh người lấy nếu có.
Alert có run_id, alert_id, timestamp, object_id, owner_id, bbox, class_name, reason,
owner_confidence, pair scores gần nhất, nguồn dữ liệu và đường dẫn evidence thực tế.
Không khai báo ảnh đã tồn tại nếu ghi file thất bại; ghi lỗi evidence riêng.

config.yaml chứa RF-DETR model/device/checkpoint, class mapping thực tế từ model
(không giả định COCO ID khi dùng model fine-tune), tracker backend, ByteTrack thresholds,
lost buffer, lớp đối tượng và alias, ROI, detection thresholds,
relation vocabulary/weights, fusion thresholds, candidate radii, motion thresholds,
T_exit, T_max, occlusion grace, retention, queue limits, retry và output options.
Kiểm tra giá trị hữu hạn, khoảng hợp lệ và quan hệ giữa ngưỡng xác nhận/ngưỡng giữ.

Relation lỗi hoặc thiếu weights: báo degraded, không trả score 0 giả; dùng heuristic
chỉ khi cấu hình cho phép. Không tải model âm thầm khi người dùng chọn local-only.
Detection lỗi/frame không mới: báo lỗi nguồn, không suy luận người vừa rời cảnh.
Cache ảnh và track hết hạn được thu dọn; giữ evidence cảnh báo trên đĩa theo chính sách riêng.

## 9. Kiểm chứng trước bàn giao runtime

Test rule engine không phụ thuộc GPU bằng observations và relation fixtures:
chủ rời nhưng người khác còn lại; chủ quay về; hai chủ với hai đồ; người lạ đi ngang;
relation thấp, UNKNOWN/top-k thiếu và worker lỗi; tranh chấp ownership;
che đồ dưới 10 giây, chủ bị che, đồ được lấy đi; Kalman prediction không phải observation;
ByteTrack không ghép chéo lớp; ID đổi/reuse; stale worker result;
fallback; một alert mỗi đợt; cleanup và timestamps.
Integration test video nhỏ xác nhận số frame, overlay, JSON và evidence bằng adapter giả.
Smoke test model thật riêng nếu môi trường và weights sẵn sàng; phân biệt rõ test giả với thật.
Benchmark video có nhãn trên phần cứng thực tế trước khi công bố FPS/accuracy/false alarm.

## 10. Phạm vi bàn giao

Mã Python độc lập, config, CLI, tests, demo observations không cần tải model,
README tiếng Việt, bảng đối chiếu PRD, ghi nguồn các phần tái sử dụng và dependencies.
Không đưa dữ liệu video cá nhân, credentials hoặc weights lớn lên GitHub.
Repo ban đầu để private nếu chưa có yêu cầu visibility; không thay visibility repo đã tồn tại.
Giấy phép và notices của dependency được giữ theo upstream; không gán lại giấy phép upstream.
