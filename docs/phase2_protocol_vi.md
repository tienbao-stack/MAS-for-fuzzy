# Protocol Phase 2 — Baseline đa phương thức trực tiếp B0

## Mục tiêu

Thiết lập một mốc tham chiếu thí nghiệm đầu tiên có thể tái lập cho bài toán hỏi đáp Vật lý với đầu vào Văn bản + Hình ảnh. Không nằm trong phạm vi của dự án: âm thanh hoặc video. ManyModalQA cũng không thuộc dự án này và không được dùng trong so sánh baseline.

## Thiết lập thí nghiệm

- **Dataset:** `data/processed/physics_dataset_candidate.json` (1.708 mẫu).
- **Chia tập:** `data/splits/split_assignments.json`, seed 42.
- **Đầu vào model:** `input.question` và `input.image.image_ref`, ảnh được phân giải từ thư mục `data/images/`.
- **Các trường bị cấm khi suy luận:** `reference.solution_cot`, `reference.answer`, `reference.unit`. Chỉ sử dụng các trường này sau khi model đã sinh đầu ra.
- **B0:** mỗi mẫu gọi một lần MLLM/VLM có khả năng xử lý ảnh; giải trực tiếp theo chế độ zero-shot; không sinh diễn giải mờ, không dùng agent và không có mô-đun xếp hạng bằng chứng.
- **Khả năng tái lập:** lưu tên model, đầu ra thô, đáp án/đơn vị đã parse, trạng thái chạy, độ trễ, số token sử dụng nếu API trả về, phiên bản prompt và toàn bộ dung sai trong đánh giá.

## Chính sách chia tập

Các thành phần liên thông được tạo khi các bản ghi có chung một trong hai điều kiện: (a) hash SHA-256 của nội dung byte ảnh giống hệt nhau, hoặc (b) câu hỏi sau khi chuẩn hóa khoảng trắng và chuyển chữ thường giống hệt nhau. Mỗi thành phần chỉ được gán vào một tập. Cách này ngăn ảnh trùng pixel chính xác và câu hỏi trùng chính xác xuất hiện giữa các tập. Các template gần trùng vẫn cần được kiểm tra thêm trong tương lai.

| Tập | Số mẫu |
|---|---:|
| Train | 1.195 |
| Validation | 257 |
| Test | 256 |

B0 không huấn luyện trên tập train; tập này được giữ lại cho các phương pháp sau này có thể cần dữ liệu phát triển hoặc huấn luyện. Dùng validation để lựa chọn prompt và sửa lỗi. Cố định prompt và cấu hình trước khi chạy test cuối cùng một lần.

## Các chỉ số đánh giá

1. **Độ chính xác đáp án có dung sai số:** với đáp án số, chấp nhận nếu `abs(pred-gold) <= max(0.001, 0.01 * abs(gold))`; với đáp án phi số, so sánh văn bản chính xác sau chuẩn hóa.
2. **Độ chính xác đáp án khớp chính xác:** giá trị số khớp trong sai số `1e-9`, hoặc văn bản sau chuẩn hóa giống hệt nhau.
3. **Độ chính xác đơn vị:** đơn vị dự đoán sau chuẩn hóa bằng đơn vị tham chiếu sau chuẩn hóa.
4. **Độ chính xác đồng thời đáp án + đơn vị:** cả đáp án và đơn vị đều đúng.
5. **Tỷ lệ parse:** tỷ lệ đầu ra có thể phân tích được thành đáp án.
6. **MAE số học / sai số tương đối:** chỉ tính trên các cặp giá trị vô hướng có thể parse thành số.
7. **Khoảng tin cậy Wilson 95%:** báo cáo cho độ chính xác đáp án và độ chính xác đồng thời đáp án + đơn vị.

Không so khớp chuỗi phần giải thích được sinh ra với `cot`. Một lời giải hợp lệ không cần có cách diễn đạt giống hệt. Nếu muốn đánh giá chất lượng suy luận, cần xây dựng rubric đánh giá thủ công riêng hoặc dùng một bộ chấm tự động đã được hiệu chuẩn và kiểm chứng với đánh giá của con người.

## Các lệnh chạy

Cài đặt bằng Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Chọn provider/model có khả năng xử lý hình ảnh trong `.env` hoặc truyền qua CLI. Ví dụ bắt đầu bằng pilot Ollama local:

```powershell
python src/baseline.py --provider ollama --mode local --model qwen3-vl:4b-instruct --split validation --limit 20 --output data/predictions/b0_ollama_qwen3vl4b_validation.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_ollama_qwen3vl4b_validation.jsonl --split validation
```

Khi model và prompt đã ổn định, chạy toàn bộ validation, xem xét các lỗi rồi cố định protocol. Chỉ sau đó mới chạy tập test:

```powershell
python src/baseline.py --split test --output data/predictions/b0_test_frozen.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_test_frozen.jsonl --split test
```

Nếu lượt chạy bị gián đoạn, dùng cùng đường dẫn đầu ra và `--resume`. Nếu file đã tồn tại và bạn chủ đích muốn thay thế, dùng `--overwrite`; không bao giờ trộn các model hoặc prompt khác nhau trong cùng một file.

## Lệnh smoke test (không phải kết quả nghiên cứu)

```powershell
python src/baseline.py --split validation --limit 3 --smoke-test-from-gold --output data/predictions/smoke_test_from_gold.jsonl
python -m unittest discover -s tests -v
```

Smoke test sao chép đáp án tham chiếu chỉ để xác minh luồng dữ liệu và evaluator. Các dòng được gắn `is_mock=true` và bị loại khỏi metric, trừ khi người dùng cố ý truyền tùy chọn không an toàn `--include-mock`. Không báo cáo điểm smoke test như hiệu năng của model.

## Giới hạn trước khi có báo cáo benchmark đạt chuẩn bài báo

Dataset ứng viên chưa được xác minh đầy đủ bằng đánh giá độc lập. Các lượt chạy Phase 2 hiện tại có thể xác nhận rằng pipeline thí nghiệm và mã metric hoạt động, đồng thời tạo kết quả baseline; tuy nhiên, các kết quả cần được ghi nhãn là sơ bộ cho đến khi một đợt rà soát thủ công theo mẫu phân tầng xác minh sự phù hợp giữa ảnh, câu hỏi, đáp án và đơn vị. Hiện cũng chưa có nhãn vàng dành riêng cho diễn giải mờ, kiểm chứng đa tác nhân hoặc lựa chọn theo ngữ cảnh.
