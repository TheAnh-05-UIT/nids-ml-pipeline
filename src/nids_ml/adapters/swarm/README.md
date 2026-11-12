# CNN Swarm adapter

## Catalog-bound batch jobs / Job NIDS theo batch

`run_batch_swarm.py` là entrypoint riêng; các phần bên dưới về `run_swarm.py` giữ
cho pilot cũ. Không trộn bootstrap, config hoặc session của hai loại job.

Batch job `nids-batch-swarm-job/1.0.0` ghim catalog ID, wire model ID, pool counts,
sample counts thật mỗi round, subset policy, initial SHA và endpoints. Wire model
ID bọc architecture ID Normal/Attack cùng catalog ID; đổi catalog phải tạo job mới.
Raw numeric body vẫn 8.964 bytes, `IUTFW001` và SL API v1 giữ nguyên. SL không giữ data.

Service SDK chưa hỗ trợ hơn 4.096 samples/update. Theo lựa chọn operator, adapter
dùng `min(subset_cap,pool_size)` mẫu/client/round, cap tối đa 4.096, không scale giả
sample counts. SQL sắp theo tensor SHA/id rồi lấy cửa sổ xoay vòng, không replacement
trong round, có thể overlap giữa hai rounds. Class weights chỉ tính từ training
subset hiện tại. Subset chỉ có một lớp dừng; không chọn lại bằng nhãn. Với pool lớn
hơn hai lần cap, báo rõ chưa dùng hết pool. Không dùng validation/test để chọn subset.

```bash
# Trong đúng runtime Python/SDK đã có; các role này chưa execute network.
python -B scripts/run_batch_swarm.py validate --config "$BATCH_CONFIG"
python -B scripts/run_batch_swarm.py sl --config "$BATCH_CONFIG" \
  --peer-id sl-a --bind 0.0.0.0 --port 9000
python -B scripts/run_batch_swarm.py client --config "$BATCH_CONFIG" \
  --peer-id sl-a --catalog "$NIDS_CATALOG" --source "$DATASET_SHA=$LOCAL_SHARDS" \
  --output artifacts/runs/batch-client-a-01
python -B scripts/run_batch_swarm.py controller --config "$BATCH_CONFIG" \
  --bind 0.0.0.0 --port 9600 --output artifacts/runs/batch-controller-01
python -B scripts/run_batch_swarm.py verify --config "$BATCH_CONFIG" \
  --controller "$CONTROLLER_OUTPUT" --client-a "$CLIENT_A_OUTPUT" --client-b "$CLIENT_B_OUTPUT"
```

Network roles cần `--execute-approved` ở task runtime riêng. SL/controller không
đọc HDF5 và không cần TensorFlow. Client cần TensorFlow 2.15.1/NumPy/h5py và SDK
User-TF đã có. Không dependency mới. Địa chỉ advertised là LAN IP/DNS của host;
`--bind` là interface listener trong container, không đồng nghĩa advertised URL.
Hai Docker Engines không chia sẻ DNS. Không expose API thiếu authentication ra Internet.

Giữ completion receipts, fencing epoch/model/base, không retry ambiguous create/
upload, chỉ retry identical receipt. Controller chưa mở round tiếp nếu thiếu receipt.
Batch checkpoints weights-only reset Adam; không resume sau restart. Output ghi
subset IDs/hash, counts/coverage, local/global artifacts, checkpoint, validation,
final shared test và resources. `verify` dùng NumPy replay FedAvg độc lập.

Integration test local (không Docker hoặc thiết bị):

```bash
python -B tests/run_batch_swarm_integration.py --service-python "$SERVICE_PYTHON" \
  --sn-source "$SN_SOURCE" --sl-source "$SL_SOURCE" --swop-source "$SWOP_SOURCE" \
  --catalog "$NIDS_CATALOG" --output artifacts/runs/batch-integration-01
```

Service Python phải là absolute path có FastAPI/Uvicorn đã chuẩn bị; ML Python có
SDK User-TF/TF. Test chỉ tạo/đóng process của mình trên loopback. Native batch Swarm
và multi-host của adapter này phải được kiểm riêng; không suy từ local TCP PASS.

## Existing pilot / Pilot cũ

Adapt the existing SL exchange and SWOP controller to the fixed Flow CNN.
Kết nối CNN đã kiểm thử với data plane hiện có; không sửa hoặc monkey-patch SDK.

## Phạm vi và runtime

- CNN n20×11×1, Conv64 3×11/ReLU/GlobalMaxPool/Dense sigmoid; 2.241 parameters.
- Hai clients, hai rounds; aggregator lần lượt `sl-a`, `sl-b`, không tuyên bố aggregation phi tập trung.
- Cùng bootstrap weights ban đầu; mỗi client train partition riêng, nạp global weights trước round tiếp.
- Adam mới sau mỗi lần nạp global weights; không đồng bộ optimizer slots.
- Dataset pilot dùng DATASET_PROVIDED_MAPPED, không phải ground truth packet độc lập.
- Không Docker/SSH, không tải/cài SDK, không tự sửa source khác hoặc điều khiển job đang chạy.

| Role | Runtime / SDK đã có |
|---|---|
| bootstrap, client | User-TF: Python 3.11, TF 2.15.1, NumPy 1.26.4, Keras 2.15.0, h5py; `user_tf.exchange` |
| sl | SL: Python 3.11, `sl_runtime` và httpx; không cần TensorFlow/NumPy |
| controller | SWOP: Python 3.11, `swop.controller`, receipts và journal; không cần TensorFlow/NumPy |
| validate | Python stdlib và NIDS source, không SDK/ML/network |
| verify | Python/NumPy, chỉ đọc artifact, không network/training |

Source/config NIDS cần được mount read-only vào đúng runtime cho từng role.
Git pull không cập nhật image. Task source không chứng minh SDK trong image hoặc
deployment multi-host đã chạy với adapter mới; phải preflight và được duyệt riêng.

## Tạo một job mới / Bootstrap

Chạy bằng Python 3.11 trong runtime ML đã chuẩn bị; từ repo root. Các biến URL do
operator cấu hình cho **SN/SL/controller riêng của job CNN mới**. Không trỏ vào
job TinyRegressor đang chạy. Giữa các host phải dùng IP/DNS LAN, không Docker DNS.

```bash
python -B scripts/run_swarm.py bootstrap \
  --dataset artifacts/runs/demo-pilot-4accb0a \
  --sn-url "$CNN_SN_URL" \
  --sl-a-url "$CNN_SL_A_URL" --sl-b-url "$CNN_SL_B_URL" \
  --controller-url "$CNN_CONTROLLER_URL" \
  --epochs 1 --output artifacts/runs/cnn-job-01

python -B scripts/run_swarm.py validate \
  --config artifacts/runs/cnn-job-01/job.json
```

Output mới gồm `initial.iutfw`, `job.json`, `manifest.json`; không ghi đè.
Job ID có prefix `cnn-` và run ID riêng. Config version `nids-cnn-swarm-job/1.0.0`
ghim dataset SHA, bootstrap SHA, hai SL URLs, callback URL, sample counts và timing.
Bootstrap là cùng một file cho hai SL. Không tự generate hai bootstrap khác nhau
trên hai board rồi mặc định coi seed giống nhau là đủ.

## Role commands / Launcher

Các ví dụ này là CLI trong đúng runtime/image, **không phải lệnh deploy board**.
Chạy thiếu `--execute-approved` sẽ chỉ validate/dry-run và không mở listener/network.
Mọi output là đường dẫn mới dưới `artifacts/runs/` của source mount NIDS.

```bash
# Role SL, image SL; bind/port là listener nội bộ, peers URL là địa chỉ advertised.
python -B scripts/run_swarm.py sl --config "$CNN_CONFIG" \
  --peer-id sl-a --bind 0.0.0.0 --port 9000

# Role client, image User-TF; sl-a -> partition 0, sl-b -> partition 1.
python -B scripts/run_swarm.py client --config "$CNN_CONFIG" \
  --peer-id sl-a --dataset "$CNN_DATASET" --output artifacts/runs/cnn-client-a-01

# Role controller, image SWOP; callback port phải reachable từ hai ML clients.
python -B scripts/run_swarm.py controller --config "$CNN_CONFIG" \
  --bind 0.0.0.0 --port 9600 --output artifacts/runs/cnn-controller-01
```

Chỉ thêm `--execute-approved` khi kế hoạch deployment riêng được phê duyệt.
SN giữ nguyên API và không giữ model weights. Lõi SL nhận CNNTemplate qua API
Python của ExchangeState/ExchangeNode; launcher SL CLI cũ vẫn chọn TinyRegressor.
Controller kế thừa freshness/fencing/create/retry của SWOP và thay hai artifact
validation hooks bằng CNN; không sửa global functions của SWOP.
SWOP profile/bundle validator cũ vẫn chọn TinyRegressor; không đưa CNN config vào
generator đó. CNN Compose và kế hoạch chuyển sang hai host thuộc bước deployment riêng.

## Completion và failure semantics

- Controller chỉ mở round sau khi SN/SL cùng epoch/model/bootstrap, registry đúng hai peers.
- Round hoàn tất sau SN synchronized, hai SL commit cùng global weights và hai client receipts.
- Receipt chỉ gửi sau client nạp global model và kiểm checkpoint; round 2 kiểm carry-over digest.
- Controller retry read-only GET có giới hạn; ambiguous POST-create dừng, không tạo lại round.
- Client không tự retry training hoặc weights upload mơ hồ. Chỉ retry receipt giống hệt tối đa 3 lần;
  receipt store đã có idempotency và từ chối receipt cùng ID nhưng khác nội dung.
- Missing peer/update/receipt, stale epoch/round, mismatch hoặc timeout không giảm quorum.
- State SN/SL/controller có phần memory-only. Restart mất phiên; không resume/take over
  job cũ bằng cách chạy lại lệnh. Giữ failed outputs/logs và review trước job mới.
- Không production authentication/TLS hay Byzantine defense. Chỉ dùng trusted lab LAN,
  không expose API thiếu bảo vệ ra Internet.

## Weights và evidence contract

Wire format giữ `IUTFW001` + big-endian JSON-header length + canonical JSON + bốn
raw little-endian float32 tensors. Không pickle/NPY/ZIP/model code. CNN model identity
là SHA-256 của fixed architecture/feature/normalization description, khác TinyRegressor.

Tensor registry: `flow_conv/kernel [3,11,1,64]`, `flow_conv/bias [64]`,
`flow_output/kernel [64,1]`, `flow_output/bias [1]`. Numeric body 8.964 bytes;
whole artifact ≤64 KiB. Kiểm shape/dtype/finite/hash và epoch/round/producer/role;
global artifact training step phải bằng 0.

| API hiện có | Vai trò |
|---|---|
| SN `POST /v1/rounds` | Controller tạo round; chỉ metadata coordination |
| SL `GET /v1/ml/session` | Client/controller kiểm epoch, model, base version và phase |
| SL `POST /v1/ml/updates` | Client gửi local weights, sample count, round/base/hash |
| SL `POST /v1/ml/peer-updates` | SL chuyển local updates qua TCP tới SL peer |
| SL `POST /v1/ml/global`, `GET /v1/ml/global/{id}` | SL phát và client lấy global weights |
| SL `POST /v1/ml/applied` | Client xác nhận đã nạp global weights |
| Controller `POST /v1/controller/receipts` | Client xác nhận round hoàn tất với base/global tensor hashes |

Không đổi endpoint hoặc protocol_version=1. Chỉ bổ sung model template cố định
ở adapter; wire model identity, tensor names/shapes khác TinyRegressor nên không
trộn bootstrap hoặc gửi CNN updates vào job cũ.

Mỗi client lưu local/global `.iutfw`, global message có contribution counts/hashes,
round reports, checkpoint restore error và test metrics cuối. Controller lưu journal
append-only và result. Đây là evidence artifacts, không commit lên GitHub.

```bash
python -B scripts/run_swarm.py verify \
  --controller "$CONTROLLER_OUTPUT" --client-a "$CLIENT_A_OUTPUT" --client-b "$CLIENT_B_OUTPUT"
```

Replay độc lập dùng NumPy float64, chia theo sample counts rồi cast float32;
không gọi FedAvg của SL. So cả hai global models, contribution identities và
round-2 base với round-1 global. Không suy native/multi-host PASS từ replay local.

## Tests / Kiểm thử

```bash
python -B -m unittest discover -s tests -p test_cnn_swarm.py -v
```

Unit tests cần runtime ML. Các interop/controller tests cần SDK SL/SWOP; nếu chưa
có installed SDK, trỏ `NIDS_SL_SOURCE`/`NIDS_SWOP_SOURCE` tới source SDK đã có.
Thiếu SDK được ghi SKIP, không coi là interop PASS. Không tự cài packages.

Integration runner dùng SDK sources đã có và **absolute Python executable** cho
service fixture. Python ML phải là 3.11; service Python cần FastAPI/Uvicorn/httpx.
Chỉ bind loopback, tạo/đóng đúng các test processes do runner sở hữu.

```bash
python -B tests/run_cnn_swarm_integration.py \
  --service-python "$SERVICE_PYTHON" \
  --sn-source "$SN_SOURCE" --sl-source "$SL_SOURCE" --swop-source "$SWOP_SOURCE" \
  --dataset "$CNN_DATASET" --output artifacts/runs/cnn-integration-01
```

Runner không build/pull/start Docker hoặc deploy. Dùng HDF5 đã có, không chạy lại
PCAP preprocessing. Source-only/local PASS không thay thế native Docker/multi-host validation.
