# Multimodal Fuzzy Agent — Phase 2 (Hỏi đáp Vật lý với Văn bản + Hình ảnh)

Gói dự án này tiếp nối Phase 1 của dự án hỏi đáp Vật lý. Mục tiêu hiện tại là **thiết lập một baseline MLLM/VLM trực tiếp (B0) có thể tái lập**, trước khi bổ sung các diễn giải mờ, kiểm chứng đa tác nhân và lựa chọn theo ngữ cảnh. Phạm vi chỉ gồm **Văn bản + Hình ảnh**; không xử lý âm thanh/video. ManyModalQA không được sử dụng trong dự án này.

## Quyết định sau Phase 1

- Dataset đã làm sạch ở Phase 1 có 1.731 mẫu, đủ các trường bắt buộc và ảnh tham chiếu đọc được từ máy cục bộ.
- Qua kiểm tra trực quan có mục tiêu, phát hiện bốn nhóm ảnh có dấu hiệu không nhất quán với câu hỏi/lời giải. 23 bản ghi liên quan vẫn được giữ trong `physics_dataset_clean_all.json` để truy vết, nhưng được cách ly khỏi dataset ứng viên dùng cho thí nghiệm.
- Dataset ứng viên có thể chạy ở Phase 2 gồm **1.708 mẫu**, lưu tại `data/processed/physics_dataset_candidate.json` và `data/processed/physics_dataset_candidate.jsonl`.
- Dataset sạch gốc gồm 1.731 bản ghi được bảo toàn tại `data/processed/physics_dataset_clean_all.json`; file Excel gốc được lưu trong `data/source/`.
- Xem `docs/phase1_readiness_report.md` để biết bằng chứng, giới hạn và kết luận về mức độ sẵn sàng của benchmark. Hai file `data/audit/human_review_blind_sample_100.csv` và `data/audit/human_review_gold_key_100.csv` tạo thành quy trình rà soát hai giai đoạn theo mẫu phân tầng: file rà soát mù không hiển thị đáp án tham chiếu và `cot` cho đến khi việc giải độc lập hoàn tất. Xem thêm `docs/human_review_instructions.md`.

## Cấu trúc dự án

```text
.
├── configs/baseline.yaml
├── data/
│   ├── source/Physics_dataset_with_image.xlsx
│   ├── raw/physics_dataset_unfiltered.json
│   ├── processed/physics_dataset_clean_all.json
│   ├── processed/physics_dataset_candidate.json
│   ├── processed/physics_dataset_candidate.jsonl
│   ├── images/                  # ảnh được dataset tham chiếu
│   ├── splits/split_assignments.json
│   ├── audit/                   # thống kê làm sạch, nhóm trùng lặp, hàng đợi rà soát
│   ├── predictions/             # đầu ra mô hình, được tạo khi chạy trên máy
│   └── results/                 # kết quả đánh giá, được tạo khi chạy trên máy
├── docs/phase1_readiness_report.md
├── docs/phase2_protocol.md
├── src/
│   ├── create_candidate_dataset.py
│   ├── audit_dataset.py
│   ├── create_splits.py
│   ├── baseline_openai.py
│   └── evaluate_predictions.py
└── tests/
```

## Cài đặt trên Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env`, đặt `OPENAI_API_KEY` và `OPENAI_MODEL` tương ứng với một model **có khả năng xử lý hình ảnh** mà bạn có quyền sử dụng. `OPENAI_BASE_URL` là tùy chọn nếu dùng endpoint tương thích với API OpenAI. Không commit file `.env` vào Git hoặc chia sẻ API key.

## Tạo lại dataset ứng viên và các tập chia sau khi thay đổi dữ liệu

Nếu danh sách cách ly hoặc dataset thay đổi, hãy tạo lại dataset ứng viên và split thay vì chỉnh thủ công file phân chia:

```powershell
python src/create_candidate_dataset.py
python src/create_splits.py --seed 42
python src/create_review_sample.py --sample-size 100 --seed 20261015
python src/audit_dataset.py --dataset data/processed/physics_dataset_candidate.json --image-dir data/images --split-file data/splits/split_assignments.json --out-dir data/audit/candidate_audit
```

Đọc lại báo cáo kiểm tra mới trước khi sử dụng tập test, sau đó tạo lại manifest bằng `python src/create_manifest.py`. Nếu thay đổi thành viên của dataset sau khi đã suy luận trên test, cần xem đó là một thay đổi đối với protocol test.

## Chạy pilot B0 nhỏ trên tập validation

```powershell
python src/baseline_openai.py --split validation --limit 20 --output data/predictions/b0_validation_pilot.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_validation_pilot.jsonl --split validation
```

Dùng validation để sửa lỗi phân tích đầu ra và cố định prompt. Sau khi đã cố định cấu hình, chỉ chạy tập test cuối cùng một lần:

```powershell
python src/baseline_openai.py --split test --output data/predictions/b0_test_frozen.jsonl
python src/evaluate_predictions.py --predictions data/predictions/b0_test_frozen.jsonl --split test
```

Để tiếp tục một lượt chạy bị gián đoạn, dùng cùng file đầu ra với `--resume`. Nếu muốn thay thế một lượt chạy trước đó, truyền `--overwrite`. Tránh trộn kết quả từ nhiều model hoặc nhiều phiên bản prompt trong cùng một file dự đoán.

## Manifest kiểm tra tính toàn vẹn dữ liệu

`data/audit/benchmark_manifest.json` lưu checksum SHA-256 của dataset ứng viên, split và cấu hình, số mẫu từng split cùng hash của các file ảnh được tham chiếu. File `data/audit/image_manifest.csv` ánh xạ mỗi ảnh tới SHA-256, kích thước ảnh và số mẫu sử dụng ảnh đó. Tạo lại các manifest bằng `python src/create_manifest.py` sau khi thay đổi dataset ứng viên hoặc split.

## Smoke test không cần API key

```powershell
python src/baseline_openai.py --split validation --limit 3 --smoke-test-from-gold --output data/predictions/smoke_test_from_gold.jsonl
python -m unittest discover -s tests -v
```

Chế độ này sao chép đáp án/đơn vị tham chiếu chỉ để kiểm tra luồng dữ liệu. Đầu ra được đánh dấu rõ `is_mock=true`; evaluator mặc định loại các dòng này khỏi metric. Đây **không phải thí nghiệm baseline**, và không được báo cáo điểm số smoke test như hiệu năng mô hình.

## Quy ước đánh giá quan trọng

- Khi suy luận, model chỉ được nhận `input.question` và ảnh được liên kết. Tuyệt đối không đưa `reference.solution_cot`, `reference.answer` hoặc `reference.unit` vào đầu vào model.
- Evaluator dùng `answer` làm nhãn vàng chính, dùng `unit` để kiểm tra đơn vị và không yêu cầu phần giải thích do model sinh ra phải trùng từng từ với `cot`.
- Các metric gồm độ chính xác đáp án có dung sai, độ chính xác đáp án khớp chính xác, độ chính xác đơn vị, độ chính xác đồng thời đáp án + đơn vị, tỷ lệ parse được đáp án, sai số số học và khoảng tin cậy Wilson.
- Đây là **dataset ứng viên cho benchmark thử nghiệm**, chưa phải benchmark công khai đã được xác thực đầy đủ. Cần rà soát thủ công thêm trước khi đưa ra các kết luận khoa học mạnh.
