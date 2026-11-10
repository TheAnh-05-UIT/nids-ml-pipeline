# Configuration guide / Hướng dẫn cấu hình

Keep versioned contracts and small examples; distinguish portable settings from historical host/data pins.
Giữ contracts và ví dụ nhỏ theo version; tách cấu hình dùng chung khỏi pins host/data lịch sử.

| File | English purpose | Mục đích tiếng Việt | Scope |
|---|---|---|---|
| `workspace.json` | Portable workspace paths; no active local manifest by default. | Paths dùng chung; mặc định không bind manifest local. | PORTABLE_DEFAULT |
| `eda.json` | Default bounded EDA budgets. | Budget EDA mặc định có giới hạn. | PORTABLE |
| `eda-ml-v5.json` | ML-CSV EDA budgets used in the local audit. | Budget EDA ML CSV của audit local. | DATA_REFERENCE |
| `eda-dnn-v5-audit.json` | DNN-CSV EDA budget reference. | Tham khảo budget EDA DNN CSV. | DATA_REFERENCE |
| `feature-policy-ml-v5-candidates.json` | Inactive feature-policy proposal bound to a dataset hash. | Feature policy đề xuất chưa áp dụng, bind dataset hash. | PROPOSAL_NOT_ACTIVE |
| `flow-cnn-n20-t10.json` | Primary n20/t10 Flow CNN contract. | Contract CNN chính n20/t10. | MODEL_CONTRACT |
| `paper-table3-n10.json` | n10 comparison profile for the paper table. | Profile n10 đối chiếu bảng của bài báo. | MODEL_CONTRACT |
| `flow-cnn-runtime.json` | Validated CNN runtime pins; not an installer/lock. | Pins runtime CNN; không là installer hoặc full lock. | RUNTIME_REFERENCE |
| `flow-hdf5-runtime.json` | Validated HDF5 runtime pins. | Pins runtime HDF5 đã kiểm. | RUNTIME_REFERENCE |
| `flow-cnn-training-reference.json` | Training reference and disclosed implementation assumptions. | Tham số train tham khảo và các giả định công bố rõ. | REFERENCE_NOT_A_RESULT |
| `flow-extraction-n10-t10.json` | Capture-order extraction n10. | Trích flow capture-order n10. | EXTRACTION_CONTRACT |
| `flow-extraction-n20-t10.json` | Capture-order extraction n20. | Trích flow capture-order n20. | EXTRACTION_CONTRACT |
| `ordered-flow-n10-t10.json` | Ordered synthetic extraction n10. | Trích ordered flow synthetic n10. | SYNTHETIC_CONTRACT |
| `ordered-flow-n20-t10.json` | Ordered synthetic extraction n20. | Trích ordered flow synthetic n20. | SYNTHETIC_CONTRACT |
| `pcap-extraction-n10-t10.json` | Versioned decoded-PCAP extraction n10. | Contract trích decoded-PCAP n10. | EXTRACTION_CONTRACT |
| `pcap-extraction-n20-t10.json` | Versioned decoded-PCAP extraction n20. | Contract trích decoded-PCAP n20. | EXTRACTION_CONTRACT |
| `flow-extraction-synthetic.example.json` | Small packet fixture for capture-order demos. | Fixture packet nhỏ để demo capture-order. | SYNTHETIC_EXAMPLE |
| `flow-labels-synthetic.example.json` | Synthetic rule/evidence/approval fixture. | Fixture rule/evidence/approval synthetic. | SYNTHETIC_EXAMPLE |
| `packet-ordering-synthetic.example.json` | Small fixture with deliberate timestamp regressions. | Fixture nhỏ cố ý có timestamp regression. | SYNTHETIC_EXAMPLE |
| `packet-alignment-heart-rate-v1.json` | Pinned local Heart-Rate prefix-audit reference. | Tham khảo audit prefix Heart-Rate với pins local. | HOST_AND_DATA_REFERENCE |
| `packet-alignment-http-v1.json` | Pinned local HTTP prefix-audit reference. | Tham khảo audit prefix HTTP với pins local. | HOST_AND_DATA_REFERENCE |
| `full-pair-http-v1.json` | Pinned full HTTP CSV/PCAP audit reference. | Tham khảo audit cả cặp HTTP đã ghim. | HOST_AND_DATA_REFERENCE |
| `full-pair-heart-rate-arm64-v2.json` | Full Heart-Rate TCP candidate audit with native-observed TShark pin. | Audit toàn cặp Heart Rate; pin TShark từ report board, chưa cấp nhãn verified. | HOST_AND_DATA_REFERENCE |
| `pcap-ordering-heart-rate-v1.json` | Raw-PCAP record-index plan; no flow/HDF5 export. | Kế hoạch index thứ tự record PCAP; chưa nối flow/HDF5. | AUDIT_PLAN_NOT_A_RESULT |
| `multicapture-audit-v1.json` | Pinned local multi-capture audit reference. | Tham khảo audit nhiều capture local đã ghim. | HOST_AND_DATA_REFERENCE |

JSON cannot contain comments. This index explains each file without adding keys
that would break exact schema/config comparisons. Do not merge n10/n20 or
capture-order/ordered configs: they represent distinct versioned behavior.

`workspace.json` defaults `dataset.manifest_path` to null so a source-only clone
can pass scaffold checks. A local dataset manifest remains in artifacts/manifests;
set its path explicitly when a workflow needs it. This does not verify labels.

Các config `HOST_AND_DATA_REFERENCE` chứa TShark path/hash và identities của
dataset local. Chúng được giữ để audit/tái lập; không dùng nguyên pins Windows
trên ARM64 hoặc coi clone source là đã có dataset. Không ghi secrets vào config.
Runtime configs mô tả scope tại thời điểm tạo; trạng thái hiện tại xem README ở root.

Heart-Rate exact counts (170,435 PCAP / 165,319 CSV) are inherited from the accepted
full metadata/quality audit. PCAP/CSV hashes match the transfer manifest. The ARM64
TShark SHA is observed from downloaded board reports, not an independent upstream
package checksum. Mismatch dừng audit; không tự sửa pin hoặc nới count/budget.
