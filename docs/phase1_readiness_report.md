# Báo cáo đánh giá Phase 1: Physics Text + Image QA

**Kết luận:** Phase 1 đã làm tốt phần chuyển đổi, lọc dữ liệu theo trường bắt buộc và kiểm tra file ảnh; nhưng **chưa đủ điều kiện gọi là benchmark nghiên cứu đã được xác nhận**. Có thể chuyển sang Phase 2 như một *pilot benchmark* sau khi cách ly các nhóm lỗi đã xác định và dùng split mới. Việc xác minh thủ công toàn bộ gold label vẫn còn thiếu.

## 1. Những gì đã kiểm tra được bằng script

Trên dataset sạch trước khi cách ly nhóm lỗi nhìn thấy bằng mắt:

| Chỉ số | Kết quả |
|---|---:|
| Dòng dữ liệu đầu vào | 3.605 |
| Mẫu sau lọc Phase 1 | 1.731 |
| ID mẫu duy nhất | 1.731 |
| Trường bắt buộc bị thiếu trong dataset sạch | 0 |
| Ảnh được tham chiếu duy nhất | 245 |
| File ảnh có trong thư mục (kể cả file không được mẫu sạch dùng) | 255 |
| Mẫu có ảnh bị thiếu hoặc không đọc được | 0 |
| Nội dung ảnh khác biệt theo SHA-256 trong 255 file | 208 |
| Nhóm file ảnh trùng byte chính xác | 14 nhóm, 61 file thuộc các nhóm này |
| Nhóm câu hỏi trùng sau chuẩn hóa khoảng trắng/chữ hoa-thường | 20 nhóm, 42 bản ghi |
| Bản ghi đã có trạng thái human review | 0; tất cả ghi `unreviewed` |

Việc lọc Phase 1 giữ lại 1.731/3.605 mẫu. Lý do loại không loại trừ lẫn nhau: có thể một bản ghi cùng lúc thiếu ảnh và có nhãn cần rà soát. Những giá trị đáp án bị Excel đọc thành ngày tháng hoặc số sê-ri đã bị loại thay vì tự sửa.

## 2. Các vấn đề được phát hiện

### 2.1. Lệch giữa đề/lời giải và ảnh

Qua kiểm tra trực quan các sơ đồ gắn với những nhóm câu hỏi mạch điện, bốn ảnh dưới đây có dấu hiệu không khớp nội dung:

- `180.png`: câu hỏi/lời giải nói đến năm điện trở và ampe kế lý tưởng, trong khi hình biểu diễn bốn điện trở và không thấy ampe kế.
- `181.png`: câu hỏi có ampe kế lý tưởng, nhưng hình có mạng điện trở mà không thấy ampe kế; lời giải mô tả topology không thể xác nhận từ hình.
- `187.png`: câu hỏi/lời giải dùng `Rx` và `Ra`, nhưng hình chỉ ghi `R1`, `R2`, `R3`, `R4`.
- `188.png`: câu hỏi/lời giải cũng dùng `Rx` và `Ra`, nhưng hình chỉ thể hiện các nhánh `R1-R2` và `R3-R4`.

Tổng cộng **23 mẫu** liên quan đến bốn ảnh này được giữ trong dataset sạch gốc để truy vết, nhưng cách ly khỏi bộ dữ liệu ứng viên Phase 2 cho đến khi được sửa hoặc adjudicate. Danh sách ở `data/audit/manual_review_queue.csv`.

Đây là kiểm tra thủ công có mục tiêu, không phải đánh giá toàn bộ 245 ảnh; có thể còn những sai lệch khác chưa được phát hiện.

### 2.2. Rò rỉ giữa split cũ

Split cũ chỉ nhóm theo tên file ảnh. Vì vậy, các file ảnh có byte giống hệt nhau nhưng tên khác nhau có thể nằm ở các split khác nhau. Đối chiếu cho thấy:

- 11 nhóm nội dung ảnh giống hệt nhau xuất hiện ở nhiều split.
- 10 nhóm câu hỏi trùng chính xác xuất hiện ở nhiều split.

Đây không chứng minh model đã bị train trên test, vì chưa chạy training; nhưng thiết kế split cũ không đủ để bảo đảm tách biệt benchmark chặt chẽ. Phase 2 thay bằng split nhóm theo **SHA-256 ảnh + câu hỏi đã chuẩn hóa chính xác**.

## 3. Trạng thái bộ ứng viên Phase 2

Sau khi cách ly 23 mẫu có dấu hiệu lệch ảnh:

| Tập | Mẫu |
|---|---:|
| Train | 1.195 |
| Validation | 257 |
| Test | 256 |
| **Tổng** | **1.708** |

Split được tạo với seed `42`. Không có nhóm ảnh trùng byte chính xác hoặc nhóm câu hỏi trùng chính xác nào băng qua ranh giới train/validation/test theo các kiểm tra tự động đã chạy. Vì phải giữ nguyên nhóm, tỷ lệ split là xấp xỉ 70/15/15.

## 4. Vì sao vẫn chưa thể gọi là benchmark cuối cùng?

1. Các mẫu còn lại chưa được con người xác nhận tính đúng đắn của `answer`, `unit`, `cot` và quan hệ giữa cả ba với hình.
2. Chưa có kiểm định độc lập hoặc đo độ đồng thuận giữa người gán nhãn.
3. Dữ liệu chưa có nhãn vàng cho số cách diễn giải mờ, loại mâu thuẫn bằng chứng, hoặc sự phù hợp của lựa chọn theo ngữ cảnh. Do đó chỉ dùng dataset hiện tại để đo QA đầu cuối; không thể chỉ dựa vào `cot`/`answer` để tuyên bố đã đánh giá đầy đủ các thành phần fuzzy/multi-agent.
4. Nguồn gốc và quyền tái phân phối dữ liệu chưa được xác minh từ file Excel. Cần làm rõ trước khi công bố benchmark công khai.
5. Split tránh lặp chính xác nhưng chưa bảo đảm tách các template câu hỏi gần giống về ngữ nghĩa.

## 5. Quyết định

- **Phase 1 — đạt có điều kiện:** hoàn tất pipeline kỹ thuật và cấu trúc dữ liệu; chưa đạt xác nhận nội dung để công bố thành benchmark cuối cùng.
- **Phase 2 — bắt đầu được:** dùng `physics_dataset_candidate.json` (1.708 mẫu) cho thí nghiệm pilot, khởi đầu từ B0 direct MLLM/VLM.
- **Không được gọi smoke test là kết quả:** các dự đoán giả lập lấy đáp án gold chỉ kiểm tra I/O, và evaluator loại chúng mặc định.
- **Trước khi báo cáo kết quả chính thức:** tiếp tục human audit có lấy mẫu phân tầng, sửa/cách ly lỗi, đóng băng prompt/model/split và chỉ sau đó chạy test.
