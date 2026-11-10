"""Expose the separate synthetic ordered-HDF5 API.
Cung cấp API ordered-HDF5 synthetic riêng.
"""
from .contracts import OrderedHdf5Sample, make_manifest
from .reader import validate_hdf5, iter_batches
from .writer import write_events

__all__=['OrderedHdf5Sample','make_manifest','validate_hdf5','iter_batches','write_events']
