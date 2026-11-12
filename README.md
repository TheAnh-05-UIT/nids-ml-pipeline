# Flow-Based CNN for Network Intrusion Detection

Edge-IIoTset research pipeline: **PCAP → flow → HDF5 → CNN**, hướng tới demo Swarm Learning trên Raspberry Pi 5 và Orange Pi 5 Pro. This is a research demo, not a production intrusion-detection system.

Nhánh mới phân loại **Normal/Attack** và báo recall theo attack family. Pilot
Normal/HTTP DDoS và notebook cũ vẫn là nhánh riêng; không coi chúng đã tự nâng cấp
thành training nhiều loại tấn công.

## Phạm vi demo / Demo scope

- Hai capture: **Heart Rate Normal (0)** và **HTTP DDoS (1)**, chỉ IPv4 TCP.
- Flow hai chiều theo 5-tuple; cửa sổ 10 giây; tối đa 20 packet, 11 features, zero padding. CNN input `(N, 20, 11, 1)`, 2.241 parameters.
- Nhãn **DATASET_PROVIDED_MAPPED** lấy từ CSV cho packet có mapping nội dung + time witness duy nhất đã audit. Đây không phải ground truth packet xác minh độc lập; time witness không chứng nhận múi giờ.
- UNKNOWN/MIXED không thành Benign. Kiểm nhãn **mọi packet** trong flow-window, cả packet vượt n=20. TCP không giải mã chắc chắn làm cách ly cả cửa sổ; ngoài TCP được thống kê, không train.
- Chia theo **connection** từng class, không random các cửa sổ của cùng connection. Loại mọi bản tensor giống hệt thuộc connection khác nhau trước split. Không đủ nhóm/class thì dừng, không fallback gây leakage.
- Đây là **within-capture pilot**: thời gian giữa connection có thể chồng lấn. Không suy ra nhận diện capture hoặc attack family chưa gặp.

CNN Swarm pilot Normal/HTTP đã chạy native giữa RPi5 và Orange Pi: hai rounds, hai clients và replay FedAvg sai số 0.0 theo original archives đã review. Đây là pilot, không phải nghiệm thu training toàn bộ dataset. N10, Packet CNN và Continual Learning nằm ngoài demo này.

## Môi trường / Runtime

Data tools dùng Python 3.11+ và TShark có SHA-256 đã kiểm. CNN/HDF5 dùng **Python 3.11, TensorFlow 2.15.1, NumPy 1.26.4, h5py 3.16.0** trong môi trường đã chuẩn bị. Không tự cài lại Python host hoặc Docker daemon. Python 3.12 host dùng cho audit, không thay runtime CNN 3.11 trong User-TF image. Git pull không cập nhật code nằm trong image.

Clone source không chứa raw data, mapping SQLite hoặc Python environment. Xem [data guide](data/README.md), không tải lại dataset đã có. Không commit raw data, HDF5, checkpoints, caches và artifacts.

## Một điểm chạy chính / Main entry point

### Chuẩn bị dữ liệu lớn nhiều loại tấn công / Large-data mapping

Đường chính tiếp theo là **raw PCAP → mapping nhãn → LUCID flows → HDF5 shards → CNN**.
Target mới: Normal/Attack nhị phân, giữ `Attack_type` để đánh giá theo family. DNN CSV
chỉ dành cho benchmark/ablation riêng, không ghép vào packet sequences.

`configs/multi-attack-pcap-v1.json` ghim 24 cặp PCAP/CSV local, source SHA-256 và số
records đã audit. Case `enabled` là được phép thử mapping, chưa có nghĩa đã đủ nhãn
hoặc sẵn sàng training. UDP, MITM/ARP, Modbus CSV sai clock, các capture nhiều ngày
chưa có calendar mapping và Vulnerability Scanner PCAP lỗi header được chặn rõ;
không ép label hoặc sửa raw để bỏ qua blocker.

Trên board Linux ARM64 đã có source mới, đúng cặp raw data và TShark 4.2.2 đã xác minh:

```bash
# Chỉ kiểm input/hash/disk; không tạo output hoặc khởi chạy TShark.
bash scripts/node_mapping.sh --case backdoor \
  --output artifacts/runs/native-map-backdoor-01

# Chạy một lần sau preflight; tạo thư mục mới, không ghi đè hoặc tự cleanup.
bash scripts/node_mapping.sh --case backdoor \
  --output artifacts/runs/native-map-backdoor-01 --execute
```

Mapping dùng Python host **3.11+** và TShark offline; không cần TensorFlow, Docker
socket, sudo hoặc cài thêm package Python. Hash TShark trong launcher là bản Ubuntu
ARM64 đã kiểm; binary khác phải review, không tự đổi pin. Cần tối thiểu **2,5 GiB disk
free sau khi đã nhận raw input** theo declared budget (1,5 GiB scratch + 1 GiB reserve).
RAM abort thresholds được lấy mẫu: Python 256 MiB, TShark 512 MiB; SQLite cache 16 MiB,
main database cap 1536 MiB; không tuyên bố đây là OS hard limits. Mapper mới thu hồi
các bảng group/index trung gian trong output đang tạo và compact trước khi seal;
giữ nguyên p/c/witnessed, nhãn, coverage và histogram. Không sửa index của task cũ.

Output: `alignment.sqlite`, `coverage.json`, `source-manifest.json` và contract.
View `mapped_labels` giữ một row cho mỗi packet nguồn: `source_ordinal`, `timestamp_ns`,
`csv_ordinal`, `mapping_status`, `y` (0/1 hoặc NULL=UNKNOWN), raw/canonical attack type.
Nhãn là **DATASET_PROVIDED_MAPPED**, không independent ground truth; candidate selection
không sử dụng label. ICMP key thiếu type/code/identifier trong CSV nên chỉ là shared
header + exact time witness, không đầy đủ packet identity. Capture nhiều ngày hoặc
clock witness không đủ có thể hoàn tất audit nhưng toàn bộ nhãn vẫn UNKNOWN.

Profile lớn reset session TShark mỗi 50.000 packets **chỉ cho raw mapping keys**;
global source ordinal được kiểm theo toàn bộ input, frame.number của TShark là local
sau reset. Không dùng chế độ này để trích 11 LUCID features vốn có relative TCP state.
Legacy HTTP/TCP profiles và extraction preferences cũ giữ nguyên.

**Bước này chưa xuất HDF5 lớn, chưa chia train/test hoặc train Swarm mới.** Không chạy
`run_demo.py prepare` với SQLite profile mới; builder pilot còn giới hạn và scope cũ.
Đường sharded HDF5/streaming reader dưới đây nối annotations đã kiểm, không nới builder pilot.
Giữ nguyên job native pilot đang có; job train lớn cần bootstrap/model identity riêng.

### PCAP → flow HDF5 shards / Disk-backed flow pipeline

`scripts/run_flows.py` có ba lệnh, không tự Docker/SSH/install hoặc training:

```bash
# Python host 3.11+ + TShark đã ghim: decode liên tục, không reset TCP session.
python3 -B scripts/run_flows.py decode \
  --mapping "$MAPPING_DIR" --tshark "$TSHARK" --tshark-sha256 "$TSHARK_SHA" \
  --output artifacts/runs/flow-cache-01

# Runtime NumPy 1.26.4 / h5py 3.16.0 đã chuẩn bị; không cài vào host chỉ để chạy lệnh này.
python -B scripts/run_flows.py export \
  --cache artifacts/runs/flow-cache-01 \
  --output artifacts/runs/flow-shards-01

python -B scripts/run_flows.py inspect --dataset artifacts/runs/flow-shards-01 --batch-size 64
```

Trên board, bước `decode` dùng host có TShark; bước `export` dùng runtime User-TF đã
có h5py/NumPy với source được mount. Việc tạo container/mount cần được operator
duyệt trước; script không tự làm. Không cần decode hoặc tạo HDF5 toàn bộ trên Windows.

Cache giữ mọi ordinal trên đĩa, decode đúng source order rồi mới dùng processing
view ổn định `(timestamp_ns, source_ordinal)`. Không sửa PCAP hoặc timestamps. Window
10s có boundary strict `timestamp > anchor+10s`, bidirectional 5-tuple, n20/11features;
packet padding không có nhãn. Mọi packet thật, cả overflow, tham gia kiểm nhãn:
UNKNOWN/MIXED bị loại; packet IPv4 mơ hồ/không decode được cách ly cả cửa sổ.
Core vẫn UNKNOWN/không verified; numeric y ở wrapper `DATASET_PROVIDED_MAPPED`.

Output export: `manifest.json`, `flows-00000.h5`, `flows-00001.h5`, ... tối đa 4096
flow/shard. Mỗi shard có `X`, `y`, `group_id`, `attack_type`, `tensor_sha256`,
`metadata`. Reader `nids_ml.storage.shards.FlowShardReader` kiểm hash/schema và đọc
1–256 flow/batch; không load toàn HDF5 vào RAM. Shards giữ group/capture lineage,
không trộn label/identity vào 11 input features. UDP features được hỗ trợ nhưng
numeric UDP labels vẫn UNKNOWN vì chưa có mapping UDP hợp lệ.

Sau khi raw inputs/mapping đã có, mỗi pha cần tối thiểu **3 GiB disk free** (2 GiB
scratch + 1 GiB dự phòng); không cộng chung thành budget đủ cho toàn bộ dataset.
SQLite cap1536MiB/cache16MiB. Decode budget tối đa3,5 triệu packet/capture,2 giờ,
Python384MiB/TShark1536MiB sampled RSS; TShark nhiều TCP connections có thể chạm
ngưỡng và phải dừng, không reset state để vượt test. Không tự cleanup khi lỗi.

**Shards này chưa có split/client assignment, chưa dedup tensor và chưa train-ready.**
Cần bước group split/dedup toàn cục chống leakage trước training lớn. Không đưa
thẳng format mới vào `run_demo.py train` hoặc job CNN Swarm pilot cũ. `attack_type`
là metadata để đánh giá từng family, model mục tiêu vẫn nhị phân Normal/Attack.

### Dedup → connection split → batch training / NIDS catalog

Catalog là artifact riêng; không sửa shard nguồn để gán fold hoặc client.

```bash
# Runtime có NumPy/h5py; kiểm toàn bộ X/hash/metadata trước split.
python -B scripts/run_training.py catalog \
  --dataset artifacts/runs/flow-shards-01 --output artifacts/runs/catalog-01

python -B scripts/run_training.py inspect --catalog artifacts/runs/catalog-01

# Python 3.11 / TF CPU 2.15.1, source và datasets đã sẵn tại máy chạy.
python -B scripts/run_training.py train --catalog artifacts/runs/catalog-01 \
  --client 0 --epochs 1 --batch-size 64 --output artifacts/runs/nids-client0-01
```

Model mới vẫn 2241 parameters, sigmoid/BCE, nhưng có **NIDS model identity riêng**
với target0Normal/1Attack. Không dùng checkpoint này cho job/model ID pilot cũ.
Lệnh `train` chỉ local training; không tự gọi SN/SL/controller hoặc FedAvg.

Dedup xét toàn bộ inputs: cùng tensor khác connection hoặc khác nhãn thì loại
toàn pattern; cùng connection/nhãn giữ một mẫu. Group có cả0/1 bị loại. Các cửa sổ
của một connection luôn cùng fold, train connection chỉ giao một client. Chia nhóm
theo class, seed cố định, khoảng70/15/15 theo **số group**, không đảm bảo tỷ lệ sample
bằng hệt. Thiếu group/class ở client hoặc holdout thì dừng, không fallback random rows.
Đây là within-capture connection holdout, không chứng minh unseen-capture/attack
generalization hoặc temporal separation giữa các connection.

`--dataset` là sources shared phải có ở cả hai clients. Khi data nằm riêng từng node,
coordinator dùng `--local-0 ... --local-1 ...`; catalog vẫn phải được lập toàn cục từ
tất cả inputs một lần rồi chuyển **cùng catalog ID** đến hai node. Không tự xây hai
catalog từ hai tập input khác nhau để dùng chung một job. Tool hiện cần đọc HDF5 của
tất cả inputs tại coordinator khi lập catalog; chưa có metadata-only inventory
transfer. Có thể đặt coordinator trên ARM64, không bắt buộc Windows xử lý raw data.

Reader chỉ mở sources cần cho split/client. Relocate bằng `--source DATASET_SHA=DIR`,
không sửa index/catalog ID; missing local source dừng trước optimizer update. Eval
rows shared có thể dùng chung ở hai clients, **không cộng hai lượt test thành hai
tập test độc lập**. Source owner cố định; conflict owner trên cùng group bị reject.

Training đọc1–128 samples/batch, shuffle reproducible theo seed/epoch bằng disk-backed
SQL sort, tối đa8 HDF5 handles. Class weights chỉ từ train của client; validation
báo mỗi epoch, test sau số epoch đã chọn trước. Runtime budget2h, Python RSS768MiB,
scratch2GiB/reserve1GiB lấy mẫu; không là hard OS limit. Checkpoint weights-only
8964bytes, SHA/model/catalog identity, finite/shape và restore prediction checks;
không pickle, không lưu optimizer state, reset Adam khi load/global application.

Output `metrics.json` có overall/per-family counts, recall cho Attack và FPR cho
Normal; family không có mẫu ghi NOT_EVALUATED. Model chỉ trả Normal/Attack, không
trả tên loại tấn công. Đây là functional pipeline, chưa bảo đảm chất lượng detection.

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

## CNN Swarm / Trao đổi weights hai rounds

Xem [hướng dẫn CNN adapter](src/nids_ml/adapters/swarm/README.md) và chạy:

```bash
python -B scripts/run_swarm.py --help
```

Các role `sl`, `client`, `controller` mặc định chỉ dry-run; chỉ thao tác network
khi có `--execute-approved`. Adapter dùng SDK trong các image tương ứng, giữ
source SN/SL/User-TF/SWOP nguyên trạng. Chưa có thao tác deploy tự động hoặc Docker
trong script này. Không chạy controller của job TinyRegressor cũ cho CNN.

## Tìm source / Source map

| Công việc | Nơi đọc |
|---|---|
| Điểm chạy demo | `scripts/run_demo.py` |
| Mapped labels, connection split, HDF5 | `src/nids_ml/data/demo.py` |
| Train/evaluate/checkpoint demo | `src/nids_ml/training/demo.py` |
| CNN đã kiểm thử | `src/nids_ml/models/flow_cnn.py` |
| CNN exchange contract, launcher và replay | `src/nids_ml/adapters/swarm/` |
| Decoder, 11 features, windows | `src/nids_ml/pcap/`, `src/nids_ml/flows/` |
| Công cụ audit và format cũ | [Source map](src/nids_ml/README.md), [scripts](scripts/README.md) |
| Cấu hình/contract | [configs](configs/README.md), [contracts](contracts/README.md) |
| Notebook NT120 | [Notebook](notebooks/NT120_DoAn_NhomXX_FlowCnnDdos.ipynb) |

Công cụ audit cũ giữ để tái lập bằng chứng, không phải tất cả đều cần chạy lại. Notebook hiện có chưa được xác minh với đường demo mới. Báo cáo/evidence task lưu ngoài repo source. Project license chưa chọn; không tự tái phân phối dataset hay third-party artifacts khi chưa rõ điều kiện.
