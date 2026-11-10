# Notebook đồ án NT120

Notebook chính: [NT120_DoAn_NhomXX_FlowCnnDdos.ipynb](NT120_DoAn_NhomXX_FlowCnnDdos.ipynb).
Đổi `NhomXX` và các ô `[TỰ ĐIỀN]` bằng thông tin thật trước khi nộp. Giữ nguyên
13 mục lớn theo mẫu của giảng viên. Mỗi phần code có Markdown giải thích mục
đích, input/output và cách đọc kết quả; thuật toán chính được gọi từ package
`src/nids_ml/`, không copy vào notebook.

## Cách chạy

1. Mở notebook trong Jupyter/VS Code hoặc Google Colab. Đưa source ML đầy đủ
   lên runtime trước; chỉ upload notebook sẽ thiếu package/configs.
2. Source root cần chứa `src/`, `configs/`, `contracts/`, `pyproject.toml`.
   Nếu không được tìm tự động, đặt `NIDS_WORKSPACE` trước Run All.
3. Kernel chạy orchestration/stdlib; HDF5/CNN chạy qua interpreter được chọn
   bởi `NIDS_PYTHON` (mặc định kernel). Nếu dùng interpreter riêng phải chuẩn
   bị trước, notebook không tự cài/downgrade Python hoặc packages.
4. Runtime HDF5: Python 3.11, NumPy 1.26.4, h5py 3.16.0. CNN forward cần thêm
   TensorFlow 2.15.1, Keras 2.15.0. Default Colab có thể không khớp các pins.
5. Chọn Restart & Run All và đọc **bảng gate cuối**, không chỉ xem cell chạy hết.

Thiếu source/runtime → BLOCKED rõ ràng, không giả kết quả. Khi runtime phù hợp
nhưng source/ABI thực sự lỗi, exception được giữ để xử lý. Chưa có xác minh
live Google Colab; kiểm tra local tuần tự không thay thế Jupyter/Colab runtime.

## Phạm vi thực thi

- Đọc synthetic fixture đã có; kiểm hash và thống kê fixture nhỏ.
- Tạo stable view, ordered flow tensors n20/n10; kiểm padding và lineage.
- HDF5 roundtrip bằng worker khi runtime phù hợp; UNKNOWN/MIXED giữ sentinel −1.
- Kiểm kiến trúc và forward **model chưa train**, không optimizer update.
- Tạo run mới dưới `artifacts/runs/nt120-*`, không ghi đè/xóa runs cũ.

Không scan CSV/PCAP thật, split/training/evaluation thật, download dataset,
Docker, SSH/deploy hoặc Git writes. Synthetic plots không là dataset EDA;
predictions chưa train không là kết quả detection. Ground truth/split/training
vẫn BLOCKED, nên notebook hiện là **bản nháp kỹ thuật**, chưa đủ bản nộp cuối.

Giữ dataset local theo [hướng dẫn dữ liệu](../data/README.md); không copy data vào notebook hoặc
GitHub. Lưu evidence thực thi riêng; bỏ outputs, secrets và data preview riêng
trước commit. Bản `.ipynb` được tạo với outputs rỗng và execution counts null.
