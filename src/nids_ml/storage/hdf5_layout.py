"""Share HDF5 layout creation, safety checks and failed-staging markers.
Dùng chung tạo layout HDF5, kiểm an toàn và đánh dấu staging lỗi.
"""
import json


def create_layout(handle, np, *, attributes, attribute_types, manifest_bytes,
                  manifest_size, row_types, tensor_shape, chunk_rows, compression):
    for name, dtype in attribute_types.items():
        value = attributes[name]
        handle.attrs.create(name, np.array(value.encode('ascii') if isinstance(value, str) else value, dtype=dtype))
    handle.create_dataset('manifest_json', data=np.array([manifest_bytes], dtype=f'S{manifest_size}'))
    for name, dtype in row_types.items():
        tail = tuple(tensor_shape) if name == 'X' else ()
        options = dict(chunks=(chunk_rows, *tail), fletcher32=True)
        if compression == 'gzip':
            options.update(compression='gzip', compression_opts=1)
        handle.create_dataset(name, shape=(0, *tail), maxshape=(None, *tail), dtype=dtype, **options)


def read_attributes(handle, types, np, require, names_error, type_error):
    require(set(handle.attrs) == set(types), names_error)
    result = {}
    for name, dtype in types.items():
        info = handle.attrs.get_id(name)
        require(info.shape == () and info.dtype == np.dtype(dtype), type_error)
        value = handle.attrs[name]
        result[name] = bytes(value).decode('ascii') if np.dtype(dtype).kind == 'S' else int(value)
    return result


def check_datasets(handle, h5py, require, errors):
    """Kiểm links/storage trước khi đọc; giữ nguyên error codes của từng format."""
    link_error, storage_error, attrs_error, alias_error, filter_error = errors
    addresses = set()
    for name in handle:
        require(isinstance(handle.get(name, getlink=True), h5py.HardLink), link_error)
        data = handle[name]
        require(isinstance(data, h5py.Dataset) and not data.is_virtual and not data.external, storage_error)
        require(not data.attrs, attrs_error)
        address = h5py.h5o.get_info(data.id).addr
        require(address not in addresses, alias_error)
        addresses.add(address)
        filters = data.id.get_create_plist()
        require(all(filters.get_filter(i)[0] in (1, 3) for i in range(filters.get_nfilters())), filter_error)


def check_row(data, shape, chunk_rows, dtype, np, require, errors):
    shape_error, chunk_error, filter_error = errors
    require(data.shape == shape and data.dtype == np.dtype(dtype), shape_error)
    require(data.chunks == (chunk_rows, *shape[1:]) and data.maxshape == (None, *shape[1:]), chunk_error)
    require(data.compression in (None, 'gzip') and data.compression_opts in (None, 1)
            and data.fletcher32 and not data.shuffle and data.scaleoffset is None, filter_error)


def record_failure(stage, h5py, error, **extra):
    """Chỉ ghi FAILED trong staging riêng, không sửa/xóa output đã có."""
    if stage.exists():
        try:
            with h5py.File(stage, 'r+') as handle:
                handle.attrs.modify('status', b'FAILED')
        except (OSError, RuntimeError, ValueError):
            pass
    if stage.parent.is_dir():
        with (stage.parent / 'FAILED.json').open('x', encoding='utf-8') as file:
            json.dump(dict(status='FAILED_NOT_PUBLISHED', error=str(error)[:512], **extra), file, sort_keys=True)
