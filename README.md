# K4-Track4-Capstone-TeamSentinel

Mini project phát hiện đồ vật bị bỏ quên, phát triển từ logic chọn lọc của VinSense/P-101
và PRD SAOD do người dùng cung cấp.

Repository: https://github.com/Wrxhard/K4-Track4-Capstone-TeamSentinel

## Trạng thái

Hiện có bản thiết kế để duyệt. Chưa có runtime, model weights hoặc kết quả benchmark.

Đọc [thiết kế SAOD và fusion confidence](docs/superpowers/specs/2026-10-16-saod-design.md).

## Luồng dự kiến

Video/webcam/RTSP → YOLO + BoT-SORT → RelateAnything → fusion theo cặp người–đồ vật
→ state machine từng đồ vật → cảnh báo, ảnh bằng chứng và log.

Ca trọng tâm: chủ sở hữu rời xe nhưng người khác vẫn còn bên trong; sự hiện diện của
người khác không tự chặn cảnh báo đồ bị bỏ lại. Quan hệ yếu không đủ để đổi chủ,
còn thiếu kết quả model được ghi là UNKNOWN.

## Nguồn tham khảo

- [RelateAnything](https://github.com/Maelic/RelateAnything)
- [API và ý nghĩa score](https://github.com/Maelic/RelateAnything/blob/main/docs/quickstart.md)
- [P-101](https://github.com/AI20K-Build-Phase-Cohort-4/P-101)

Theo dõi giấy phép từng thành phần khi tích hợp mã và model; không coi giấy phép
của dự án tham khảo là giấy phép mặc định cho tất cả dependency.
