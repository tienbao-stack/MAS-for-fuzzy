# Hướng dẫn human audit nhãn vàng

Dùng hai file để tách đánh giá độc lập khỏi nhãn tham chiếu:

- `data/audit/human_review_blind_sample_100.csv`: danh sách cần rà soát, **không có đáp án/cot gốc**.
- `data/audit/human_review_gold_key_100.csv`: đáp án, đơn vị và lời giải gốc; chỉ mở sau khi người rà soát hoàn thành câu trả lời độc lập. Hai file ghép bằng `sample_id`.

Ảnh tương ứng nằm trong `data/images/`, tên file ở cột `image_ref`. Mẫu lấy phân tầng theo train/validation/test và tiền tố ID, với seed cố định để tái lập.

## Giai đoạn 1: đánh giá độc lập

Với mỗi hàng trong file blind, điền:

- `independent_answer`: kết quả do người rà soát tự tính từ câu hỏi và ảnh.
- `independent_unit`: đơn vị do người rà soát tự xác định.
- `image_matches_question`: `yes`, `no`, hoặc `uncertain`.
- `reviewer_notes`: ghi dữ kiện, phép tính hoặc dấu hiệu không nhất quán.
- `review_status`: dùng `reviewed` khi hoàn tất.

Chưa mở file gold key trong giai đoạn này để giảm thiên lệch.

## Giai đoạn 2: đối chiếu nhãn

Sau khi hoàn tất độc lập, đối chiếu `independent_answer`/`independent_unit` với `reference_answer`/`reference_unit` trong gold key. Ghi lại kết quả so sánh trong bản sao của file blind hoặc trong một file adjudication mới; không sửa file nguồn. So sánh lời giải bằng tính hợp lệ vật lý/toán học, không yêu cầu diễn đạt trùng nguyên văn.

## Quy trình khuyến nghị

1. Rà soát ảnh và câu hỏi trước, sau đó tự giải; chỉ nhìn nhãn tham chiếu ở lượt hai.
2. Với nghiên cứu công bố, nên có hai người chấm độc lập trên ít nhất một phần dữ liệu, sau đó adjudicate bất đồng và báo cáo độ đồng thuận.
3. Chỉ sửa nhãn sau khi có lý do và audit trail. Không ghi đè dữ liệu nguồn; lưu phiên bản mới và tạo lại manifest/split nếu một mẫu bị loại hoặc thay đổi nhóm.

Mẫu 100 là kiểm tra chất lượng ban đầu, không chứng minh rằng mọi mẫu chưa được kiểm tra đều đúng.
