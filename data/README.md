# Shared data storage / Dữ liệu dùng chung

Use one local dataset copy, preserve raw files and record provenance/checksums.
Dùng lại dataset local đã có, giữ nguyên raw files và ghi rõ nguồn/checksum.
Raw data is ignored by Git; a source-only clone contains this README and markers.

## Local paths / Đường dẫn local

Paths below are relative to `data/`; `configs/workspace.json` defines storage roots.

- Operator-provided dataset: `raw/Edge-IIoTset dataset/`.
- ML CSV: `raw/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`.
- DNN CSV: `raw/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv`.
- Canonical audited ML CSV: `raw/edge-iiotset/kaggle-v5/ML-EdgeIIoT-dataset.csv`.

The canonical ML CSV was recorded as 82,184,390 bytes, SHA-256
`53101fad091af20bee815860962a5016b802ee45d456a141042f6183094c3c1a`.
This identity does not establish provenance or verified labels for other files.
Thông tin operator bổ sung không thay kiểm revision/checksum và ground truth.

## Rules / Quy tắc

1. Reuse existing files; do not download, unpack or copy the dataset again.
   Dùng file đã có; không tự tải, giải nén hoặc nhân bản dataset giữa task folders.
2. Do not alter, move or delete raw files or automatically remove duplicate copies.
   Không sửa raw để vượt kiểm thử; derivatives phải dùng output path mới.
3. Verify each new asset against its own provenance, size/hash and label evidence.
   Không áp receipt của ML CSV cho toàn bộ PCAP/DNN archive.
4. Do not infer benign labels from unmatched packets or attack labels from filenames.
   CSV row labels chưa tự xác nhận PCAP/flow ground truth.
5. Do not redistribute the dataset while license differences remain unresolved.
   Audit trước ghi nhận metadata Kaggle CC BY-NC-SA 4.0 và IEEE CC BY 4.0 khác nhau;
   không đưa CSV/PCAP vào Git hoặc image trong phạm vi hiện tại.

| Directory | Purpose / Mục đích |
|---|---|
| `raw/` | Original local data / Dữ liệu gốc local |
| `interim/` | Reproducible intermediate outputs / Kết quả trung gian có thể tái tạo |
| `processed/` | Versioned transformed features / Features theo version preprocessing |
| `splits/` | Approved split indices/metadata / Metadata và indices chia tập được duyệt |

Runtime manifests remain under `artifacts/manifests/` and are not shipped in Git.
`dataset.manifest_path` defaults to null for source-only checks; set a local path
explicitly when needed. This does not bypass provenance or verified-label gates.
