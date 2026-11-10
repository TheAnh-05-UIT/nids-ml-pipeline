# Artifact storage

TASK-010 khởi tạo không tạo model/checkpoint/metrics. DATA-001 hiện có dataset
manifest và hai EDA runs thật; vẫn không có model, checkpoint hoặc train scores.

- `manifests/`: metadata thực tế theo dataset/artifact contracts khi đã có dữ liệu.
- `runs/<run_id>/`: model, checkpoint, preprocessing artifacts, metrics và predictions
  của một lần chạy. Run ID mới cho lần chạy mới, không ghi đè thí nghiệm đã có.

Các EDA runs thực tế: `data001-kaggle-v5-ml-20261106` và
`data001-kaggle-v5-ml-quality-20261106`. COMPLETE.json của mỗi run liệt kê hashes
reports để phát hiện ghi thiếu/thay đổi. Không coi marker là PASS task nếu chưa
review provenance, scope thống kê và chất lượng dữ liệu. Dataset manifest:
`manifests/edge-iiotset-kaggle-v5-ml.json`.

Các paths lấy từ configs/workspace.json. Outputs và manifests runtime mặc định bị
Git ignore; source config/contracts vẫn được version hóa. Báo cáo task chỉ tham
chiếu paths/checksums ở đây, không chứa bản sao weights hoặc raw dataset.
