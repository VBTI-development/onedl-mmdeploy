# Copyright (c) OpenMMLab. All rights reserved.
import importlib

from mmdeploy.utils import get_root_logger
from .version import __version__, version_info  # noqa F401

if importlib.util.find_spec('torch'):
    # PyTorch 2.9+ defaults dynamo=True in torch.onnx.export, which bypasses
    # the TorchScript-based symbolic functions that mmdeploy relies on for
    # custom ops (e.g. mmdeploy::Mark).  Patch the default back to False so
    # that every call site — inside the library and in tests — uses the legacy
    # TorchScript exporter without needing individual dynamo=False arguments.
    import functools

    import torch

    _original_onnx_export = torch.onnx.export

    @functools.wraps(_original_onnx_export)
    def _onnx_export_legacy_default(*args, **kwargs):
        kwargs.setdefault('dynamo', False)
        return _original_onnx_export(*args, **kwargs)

    torch.onnx.export = _onnx_export_legacy_default

if importlib.util.find_spec('torch'):
    importlib.import_module('mmdeploy.pytorch')
else:
    logger = get_root_logger()
    logger.debug('torch is not installed.')

if importlib.util.find_spec('mmcv'):
    importlib.import_module('mmdeploy.mmcv')
else:
    logger = get_root_logger()
    logger.debug('mmcv is not installed.')
