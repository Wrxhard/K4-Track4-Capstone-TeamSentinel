# Bản đầu thu gọn — 16/10/2026

Thiết kế này thay thế phạm vi bản SAOD trước theo hai yêu cầu mới nhất:
chỉ tập trung RelateAnything, vẫn giữ tracking/Kalman khi đồ bị che khuất.

Luồng: detector YOLO trong cách dùng mẫu upstream → ByteTrack có Kalman →
RelateAnything trên box thực sự quan sát được → preview/video và JSONL.
Không tích hợp RF-DETR riêng, worker bất đồng bộ, fusion ownership, cảnh báo bỏ quên
hay giao diện web ở bản đầu. Không tối ưu lớp đồ nhỏ/khó detect.

Một tracker cho mỗi lớp để không ghép người với đồ. Mỗi track có ID toàn cục,
box quan sát cuối, thời gian cuối nhìn thấy. Khi mất detection, giữ tối đa 10 giây;
đồ gần như đứng yên dùng box cuối, đồ đang di chuyển dùng Kalman prediction.
Box dự đoán có nhãn OCCLUDED và không được đưa vào relation inference như observation mới.
Quan hệ cũ được giữ để hiển thị trong tối đa khoảng giữ track nhưng ghi rõ stale và tuổi dữ liệu.
ByteTrack không có appearance Re-ID, không bảo đảm giữ ID qua che khuất dài/giao cắt.

Đầu vào ảnh, file video hoặc webcam. Model/cache tải ở máy người dùng, không commit.
Output mới cho mỗi run: ảnh hoặc video annotated và events.jsonl.
CLI có model/device, source, hold-seconds, relation-every, threshold và no-preview.
Kiểm thử thực ByteTrack: mất dấu/nhìn lại giữ ID, hết hạn, ghép đúng lớp,
không dùng predicted box để tính relation. Kiểm thử video bằng adapter giả để kiểm tra I/O.

Nguồn API: https://github.com/Maelic/RelateAnything (commit 197ac47d645f106e8beb79e49a3f62b91fcbe91f).
