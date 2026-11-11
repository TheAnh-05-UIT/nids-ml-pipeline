# Flow-Based CNN for DDoS Detection

Edge-IIoTset research pipeline: **PCAP → flow → HDF5 → CNN**, hướng tới demo Swarm Learning trên Raspberry Pi 5 và Orange Pi 5 Pro. This is a research demo, not a production intrusion-detection system.

## Phạm vi demo / Demo scope

- Hai capture: **Heart Rate Normal (0)** và **HTTP DDoS (1)**, chỉ IPv4 TCP.
- Flow hai chiều theo 5-tuple; cửa sổ 10 giây; tối đa 20 packet, 11 features, zero padding. CNN input `(N, 20, 11, 1)`, 2.241 parameters.
- Nhãn **DATASET_PROVIDED_MAPPED** lấy từ CSV cho packet có mapping nội dung + time witness duy nhất đã audit. Đây không phải ground truth packet xác minh độc lập; time witness không chứng nhận múi giờ.
- UNKNOWN/MIXED không thành Benign. Kiểm nhãn **mọi packet** trong flow-window, cả packet vượt n=20. TCP không giải mã chắc chắn làm cách ly cả cửa sổ; ngoài TCP được thống kê, không train.
- Chia theo **connection** từng class, không random các cửa sổ của cùng connection. Loại mọi bản tensor giống hệt thuộc connection khác nhau trước split. Không đủ nhóm/class thì dừng, không fallback gây leakage.
- Đây là **within-capture pilot**: thời gian giữa connection có thể chồng lấn. Không suy ra nhận diện capture hoặc attack family chưa gặp.

Hạ tầng OpenSwarm và smoke tests ARM64 có nghiệm thu riêng. CNN local không chứng minh CNN đã train bằng Swarm; source demo mới cần kiểm thử trên board. N10, Packet CNN, attack families khác và Continual Learning nằm ngoài demo này.

## Môi trường / Runtime

Data tools dùng Python 3.11+ và TShark có SHA-256 đã kiểm. CNN/HDF5 dùng **Python 3.11, TensorFlow 2.15.1, NumPy 1.26.4, h5py 3.16.0** trong môi trường đã chuẩn bị. Không tự cài lại Python host hoặc Docker daemon. Python 3.12 host dùng cho audit, không thay runtime CNN 3.11 trong User-TF image. Git pull không cập nhật code nằm trong image.

Clone source không chứa raw data, mapping SQLite hoặc Python environment. Xem [data guide](data/README.md), không tải lại dataset đã có. Không commit raw data, HDF5, checkpoints, caches và artifacts.

## Một điểm chạy chính / Main entry point

`scripts/run_demo.py`: `prepare`, `inspect`, `train`. Chạy từ repo; `python` bên dưới phải là runtime Python 3.11 đã chuẩn bị. Output là thư mục mới dưới `artifacts/runs/`; không ghi đè và không tự cleanup khi lỗi.

```bash
python -B scripts/run_demo.py --help

# Mỗi mapping directory phải có coverage.json và alignment.sqlite từ full-pair audit.
# TSHARK và TSHARK_SHA là đường dẫn/hash binary trên máy đang chạy.
python -B scripts/run_demo.py prepare \
  --normal-mapping "$NORMAL_MAPPING" \
  --attack-mapping "$HTTP_MAPPING" \
  --tshark "$TSHARK" --tshark-sha256 "$TSHARK_SHA" \
  --max-per-class 1000 --output artifacts/runs/demo-data-01

python -B scripts/run_demo.py inspect --dataset artifacts/runs/demo-data-01

python -B scripts/run_demo.py train \
  --dataset artifacts/runs/demo-data-01 \
  --epochs 3 --batch-size 64 --output artifacts/runs/demo-cnn-01
```

`prepare` kiểm SHA-256 PCAP/CSV và decode **theo source order** để giữ TCP state; sau đó mới sắp metadata theo `(timestamp, source ordinal)`. SQLite giữ lineage với RAM giới hạn. Quét full captures trước khi chọn tối đa số flow/class đã đặt: đây là bounded pilot, không phải toàn dataset. Trước chạy cần 1,5 GiB trống và luôn giữ dự phòng 1 GiB.

Output `dataset.h5`: `X`, `y`, `split`, `client`, `metadata`; `manifest.json` ghi nguồn/checksum, số bị loại và phân bố thật. Client 0/1 có train samples riêng và cả hai classes; validation/test dùng chung để đánh giá.

Đây là format demo riêng `nids-mapped-demo-hdf5/1.0.0`, không nới gate verified-only cũ. `metadata.core` vẫn giữ trạng thái diagnostic UNKNOWN; nhãn số nằm ở wrapper `metadata.y` với provenance DATASET_PROVIDED_MAPPED. Không biến core UNKNOWN thành VERIFIED.

`train` dùng class weights tính **chỉ từ train**, Adam/BCE, seed cố định, một CPU thread. Báo validation mỗi epoch; test đánh giá sau số epoch cố định. Output: `metrics.json`, `test-predictions.csv`, `weights.bin`, `checkpoint.json`, hình nếu có matplotlib. Checkpoint không pickle; kiểm SHA/shape/dtype/finite và prediction sau restore. Chỉ lưu weights, không lưu optimizer state.

`--client 0` hoặc `--client 1` train riêng một partition; **không tự trao đổi weights hoặc gọi Swarm**. Những bước native CNN và Swarm CNN chưa chạy không được ghi PASS.

## Kiểm thử / Tests

```bash
python -B scripts/validate_workspace.py
python -B -m unittest discover -s tests -p test_demo_pipeline.py -v
```

Suite demo dùng fixtures synthetic và CPU training nhỏ để kiểm HDF5, split/dedup guards, UNKNOWN overflow và checkpoint. Fixture metrics không phải Edge-IIoTset metrics. Regression khác xem [tests](tests/README.md). Local PASS không chứng minh native ARM64, CNN Swarm hoặc Colab PASS.

## Tìm source / Source map

| Công việc | Nơi đọc |
|---|---|
| Điểm chạy demo | `scripts/run_demo.py` |
| Mapped labels, connection split, HDF5 | `src/nids_ml/data/demo.py` |
| Train/evaluate/checkpoint demo | `src/nids_ml/training/demo.py` |
| CNN đã kiểm thử | `src/nids_ml/models/flow_cnn.py` |
| Decoder, 11 features, windows | `src/nids_ml/pcap/`, `src/nids_ml/flows/` |
| Công cụ audit và format cũ | [Source map](src/nids_ml/README.md), [scripts](scripts/README.md) |
| Cấu hình/contract | [configs](configs/README.md), [contracts](contracts/README.md) |
| Notebook NT120 | [Notebook](notebooks/NT120_DoAn_NhomXX_FlowCnnDdos.ipynb) |

Công cụ audit cũ giữ để tái lập bằng chứng, không phải tất cả đều cần chạy lại. Notebook hiện có chưa được xác minh với đường demo mới. Báo cáo/evidence task lưu ngoài repo source. Project license chưa chọn; không tự tái phân phối dataset hay third-party artifacts khi chưa rõ điều kiện.
