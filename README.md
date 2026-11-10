# NIDS ML Pipeline

A research toolkit for traceable, flow-based network intrusion detection.
It includes bounded PCAP decoding, CSV/PCAP audits, LUCID-style flow tensors,
strict label contracts, integrity-checked HDF5 and a small Flow CNN.

Bộ công cụ nghiên cứu phát hiện xâm nhập mạng theo flow: giải mã PCAP,
audit dữ liệu, tạo flow tensor, kiểm nhãn, lưu HDF5 và thử nghiệm Flow CNN.

This is an **in-progress research project**, not a trained production detector.
Real-data supervised CNN training, final evaluation and CNN Swarm integration
are not complete. No detection score is claimed from synthetic tests.

## Start here / Bắt đầu từ đâu

| Goal / Công việc | Entry point / Nơi đọc |
|---|---|
| Understand the project / Hiểu cấu trúc | [Source map](src/nids_ml/README.md) |
| Read the coursework / Đọc notebook đồ án | [NT120 notebook](notebooks/NT120_DoAn_NhomXX_FlowCnnDdos.ipynb) |
| Run a small demo / Chạy demo nhỏ | Synthetic quickstart below / Ví dụ bên dưới |
| Review settings / Xem cấu hình | [configs](configs/README.md) |
| Understand data formats / Hiểu format dữ liệu | [contracts](contracts/README.md) |
| Run tests / Chạy kiểm thử | [tests](tests/README.md) |
| Check scope and gaps / Xem phần đã làm/chưa làm | Research scope below / Phạm vi nghiên cứu bên dưới |

## Project layout / Cấu trúc

```text
src/nids_ml/   Domain code / Thuật toán và contracts thực thi
scripts/       CLI launchers and audit utilities / Lệnh chạy và công cụ audit
configs/       Portable examples and labelled reference configs / Ví dụ và config tham khảo
contracts/     Versioned JSON Schemas / Schema có version
tests/         Small synthetic fixtures and regression tests / Fixtures nhỏ và tests
notebooks/     NT120 report and guided demo / Báo cáo đồ án và demo có giải thích
data/          Local dataset storage; ignored / Dataset local, không đưa lên Git
artifacts/     Local outputs and manifests; ignored / Outputs local, không đưa lên Git
```

Each Python file starts with a short English/Vietnamese purpose statement. Examples
call the package implementation instead of duplicating it inside notebooks.
Mỗi file Python có mô tả đầu file bằng Anh–Việt; xem file đó để biết đúng trách
nhiệm. Các package mở rộng còn trống được ghi rõ là **placeholder**.

## Environment / Môi trường

Use **Python 3.11** for the validated HDF5/CNN path. Core data/ordering tools
mostly use the standard library. Optional HDF5 pins are NumPy 1.26.4 and
h5py 3.16.0; CNN additionally expects TensorFlow 2.15.1 / Keras 2.15.0.
Direct pins are not a complete transitive lock. TShark is a separate requirement
for PCAP decoding; the synthetic quickstart does not need it.

Nếu cần môi trường mới, operator tạo venv riêng; không thay global Python hoặc
môi trường OpenSwarm đang chạy:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --editable '.[hdf5]'
.venv/bin/python -B scripts/validate_workspace.py
```

Windows: use `py -3.11 -m venv .venv`, then `.venv\Scripts\python.exe`.
Add `-X utf8` if the terminal encoding does not display Vietnamese correctly.
Installing dependencies may need Internet; no command runs automatically.
The HDF5 extra does not install TensorFlow, Jupyter or TShark.

## Synthetic quickstart / Demo synthetic

Run from this directory with the prepared Python 3.11/HDF5 environment. Pick a
**new** output directory each time. Đây là kiểm pipeline, không phải dataset
Edge-IIoTset thật hoặc kết quả training.

```bash
mkdir artifacts/runs/my-ordered-demo
python -B scripts/packet_ordering.py build-synthetic --fixture configs/packet-ordering-synthetic.example.json --output artifacts/runs/my-ordered-demo/view
python -B scripts/ordered_flow_hdf5.py export-synthetic --fixture configs/packet-ordering-synthetic.example.json --view artifacts/runs/my-ordered-demo/view --profile flow-cnn-n20-t10 --output artifacts/runs/my-ordered-demo/n20.hdf5
python -B scripts/ordered_flow_hdf5.py validate --input artifacts/runs/my-ordered-demo/n20.hdf5
```

For n10, select `paper-table3-n10` and a new output `n10.hdf5`.
X has shape `(N,n,11,1)`. Source ordinals, processing indices and full/overflow
lineage are separate. UNKNOWN/MIXED samples retain `y=-1`, mask=false and are
excluded from supervised training. Capture-order HDF5 and ordered-HDF5 have
different contracts; readers reject the wrong format.

## Tests / Kiểm thử

These suites use synthetic inputs and do not train a CNN:

```bash
python -B -m unittest discover -s tests -p test_packet_ordering.py -v
python -B -m unittest discover -s tests -p test_ordered_flow.py -v
python -B -m unittest discover -s tests -p test_ordered_hdf5.py -v
```

Do not assume full unittest discovery is a no-training check: some CNN tests
perform tiny synthetic optimizer updates. Xem [test guide](tests/README.md)
để chọn đúng phạm vi và dependencies. Synthetic test metrics are not dataset
metrics; local tests are not proof of native ARM64 or live Colab execution.

## Research scope / Phạm vi nghiên cứu

Target workflow: **PCAP + verified annotations → flow/HDF5 → leakage-safe split
→ centralized CNN baseline → evaluation → Swarm adapter on edge nodes**.

The local Edge-IIoTset dataset already exists and is not redistributed here.
Read the [data guide](data/README.md). CSV row labels do not by
themselves establish PCAP/flow ground truth. Không đổi unmatched traffic thành
benign, dùng folder name làm nhãn hoặc đưa metadata identity vào CNN inputs.

The Colab notebook is a draft: it requires this source checkout and compatible
dependencies; live Colab execution is not verified. Existing two-board tiny-model
FedAvg evidence does not establish CNN Swarm training. Pulling source does not
update the code already baked into a running container image.

## GitHub scope / Phần đưa lên Git

Keep source, tests, configs, schemas, notebooks and documentation. Keep raw data,
runtime manifests, HDF5/checkpoints, SQLite indexes, caches and secrets local.
Only `README.md` files are included as Markdown documentation; other `.md` files
remain local and are ignored. Chỉ các README được chọn commit; không xóa các
ghi chú lịch sử còn cần dùng ở local.
Only README/`.gitkeep` markers from data/artifacts belong in the source repo.
Historical host-specific audit configs are marked **reference-only**; their
TShark pins must be revalidated for a different host. `.gitignore` does not
replace a secrets review or prevent publishing files explicitly force-added.

The project license has not been selected in this workspace. Choose the intended
license and review applicable notices before granting public reuse rights;
this documentation does not assign a license or change dataset terms.
