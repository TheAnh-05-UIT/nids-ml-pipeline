# Contract guide / Hướng dẫn contracts

These JSON Schemas describe accepted data structures, not datasets or experimental results.
Các JSON Schemas mô tả cấu trúc được chấp nhận, không phải dataset hoặc kết quả thí nghiệm.

| Schema | English purpose | Mục đích tiếng Việt |
|---|---|---|
| `common.schema.json` | Shared IDs, versions, paths and SHA-256 constraints. | Ràng buộc dùng chung về ID, version, path và SHA-256. |
| `dataset.schema.json` | Dataset provenance, members and targets. | Nguồn dataset, các file thành viên và target. |
| `artifact.schema.json` | Generated artifact identity and lineage. | Định danh và lineage của artifact được tạo. |
| `feature-policy.schema.json` | Proposed feature selection/exclusion policy. | Policy đề xuất chọn/loại features. |
| `flow-cnn-checkpoint.schema.json` | Versioned CNN checkpoint metadata and tensor registry. | Metadata checkpoint CNN và registry tensor theo version. |
| `flow-label-rules.schema.json` | Synthetic rule, evidence and approval contract. | Contract rule, evidence và approval synthetic. |
| `flow-packet.schema.json` | Synthetic packet metadata accepted by the flow core. | Metadata packet synthetic được flow core chấp nhận. |
| `decoded-packet.schema.json` | Decoded-PCAP packet metadata contract. | Contract metadata packet decoded-PCAP. |
| `packet-ordering-source.schema.json` | Synthetic source identity and ordering policy. | Source identity và ordering policy synthetic. |
| `packet-processing-view.schema.json` | Completed synthetic processing-view metadata. | Metadata processing view synthetic đã hoàn tất. |
| `ordered-flow-configuration.schema.json` | Versioned ordered-flow extraction configuration. | Cấu hình trích ordered-flow theo version. |
| `ordered-flow-metadata.schema.json` | Original ordinals, processing indices and full-window lineage. | Ordinal gốc, processing indices và lineage full window. |
| `ordered-flow-hdf5.schema.json` | Synthetic ordered-HDF5 manifest; file invariants also checked in code. | Manifest ordered-HDF5 synthetic; code còn kiểm invariants của file. |

Versions differ by contract; ordered-flow metadata is not the same format as
capture-order HDF5. Python validators additionally enforce cross-field lineage,
hash, completion, dtype/shape and label-safety invariants. The local validator
supports a bounded keyword set; it is not a general JSON Schema engine and
does not resolve remote HTTP references.

Consumer phải kiểm containment/symlinks và checksum thực tế, không chỉ JSON
syntax. SHA thể hiện consistency, không tự chứng minh provenance hoặc nhãn đúng.
UNKNOWN/MIXED không được chuyển thành supervised numeric labels. Breaking changes
cần version mới; không sửa semantics của schema cũ để vượt kiểm thử.

Decoded-spool and decoded ordered-flow diagnostics use separate runtime contracts
in `ordering/decoded.py` and `flows/decoded_ordered.py` (metadata/events v3).
They reuse field/lineage checks without accepting real metadata as synthetic v2.
Whole-capture EOF/count, raw-record bindings, SQLite/blob limits, UNKNOWN labels
and diagnostic completion are additionally checked in code. HDF5 v2 stays closed
to the new event types; no schema file was relaxed to activate training.

Các schema files cần commit vì source/tests dùng chúng. Raw manifests, scores
và model weights thuộc local artifacts, không được thay bằng dữ liệu giả trong
contracts. File README này giải thích schema thay vì thêm comment không hợp lệ vào JSON.
