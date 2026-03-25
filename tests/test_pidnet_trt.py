# Copyright (c) OpenMMLab. All rights reserved.
"""Integration test: export PIDNet-S to TensorRT and compare pixel-level
output against the original PyTorch model.

Requirements:
  - TensorRT must be installed (test auto-skips otherwise)
  - mmseg must be installed (test auto-skips otherwise)
  - work_dir_pidnet/ with pre-downloaded files, OR internet access so that
    the test can call ``mim download`` automatically.

Run:
    pytest tests/test_pidnet_trt.py -v -s
"""
import copy
import glob
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import pytest
import torch


def _sdk_available() -> bool:
    """Return True if the mmdeploy_runtime C++ SDK bindings are importable."""
    import importlib
    import sys

    # The built SDK .so may live in build/lib/ — add it to path if present.
    _build_lib = osp.join(_REPO_ROOT, 'build', 'lib')
    if _build_lib not in sys.path:
        sys.path.insert(0, _build_lib)
    return importlib.util.find_spec('mmdeploy_runtime') is not None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
_WORK_DIR_PIDNET = osp.join(_REPO_ROOT, 'work_dir_pidnet2')
_MODEL_NAME = 'pidnet-s_2xb6-120k_1024x1024-cityscapes'


def _find_pidnet_files(search_dir):
    """Return (config_path, checkpoint_path) from *search_dir*, or (None,
    None)."""
    config = osp.join(search_dir, f'{_MODEL_NAME}.py')
    if not osp.exists(config):
        config = None
    hits = glob.glob(osp.join(search_dir, 'pidnet-s*.pth'))
    checkpoint = hits[0] if hits else None
    return config, checkpoint


def _download_pidnet(dest_dir):
    """Run ``mim download onedl-mmsegmentation`` into *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable, '-m', 'mim', 'download', 'onedl-mmsegmentation',
            '--config', _MODEL_NAME, '--dest', dest_dir
        ],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


# Use a reduced spatial resolution so TRT build and inference stay fast.
# PIDNet-S has a stride-32 backbone; any multiple of 32 is valid.
INPUT_H = 256
INPUT_W = 256

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ort_deploy_cfg():
    """Build an OnnxRuntime dynamic-shape deploy config for ONNX export.

    This config is used for torch2onnx and export2SDK so that deploy.json
    records ``backend: onnxruntime`` as the C++ SDK inference backend.
    """
    from mmengine import Config
    return Config(
        dict(
            backend_config=dict(type='onnxruntime'),
            codebase_config=dict(
                type='mmseg', task='Segmentation', with_argmax=True),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file='end2end.onnx',
                input_names=['input'],
                output_names=['output'],
                input_shape=None,
                dynamic_axes=dict(
                    input={
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    },
                    output={
                        0: 'batch',
                        2: 'height',
                        3: 'width'
                    }),
                optimize=True),
        ))


def _make_sdk_deploy_cfg():
    """Build the SDK deploy config for PIDNet (mirrors
    segmentation_sdk_dynamic.py)."""
    from mmengine import Config
    return Config(
        dict(
            backend_config=dict(
                type='sdk',
                pipeline=[
                    dict(type='LoadImageFromFile'),
                    dict(type='LoadAnnotations'),
                    dict(
                        type='PackSegInputs',
                        meta_keys=['img_path', 'ori_shape', 'img_shape']),
                ]),
            codebase_config=dict(
                type='mmseg', task='Segmentation', model_type='sdk'),
            onnx_config=dict(
                type='onnx',
                export_params=True,
                keep_initializers_as_inputs=False,
                opset_version=11,
                save_file='end2end.onnx',
                input_names=['input'],
                output_names=['output'],
                input_shape=None,
                optimize=True),
        ))


def _remove_preprocessor_pad_size(model_cfg):
    """Drop fixed-size padding from SegDataPreprocessor.

    The trained PIDNet config pads inputs to 1024×1024.
    For this test we use
    a smaller input, so we remove ``size`` / ``size_divisor`` to let the
    preprocessor operate at whatever spatial size it receives.
    """
    cfg = copy.deepcopy(model_cfg)
    for path in ('model.data_preprocessor', 'data_preprocessor'):
        node = cfg
        try:
            for key in path.split('.'):
                node = node[key]
            node.pop('size', None)
            node.pop('size_divisor', None)
        except (KeyError, TypeError):
            pass
    return cfg


# ---------------------------------------------------------------------------
# Module-scoped fixture: do the heavy export + build ONCE for all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def pidnet_trt_context(tmp_path_factory):
    """Export PIDNet-S → ONNX and build reference + SDK models.

    Files are resolved in this order:
      1. ``work_dir_pidnet/`` (pre-downloaded, fastest)
      2. A module-scoped tmp dir populated by ``mim download`` (requires net)

    The fixture is skipped automatically when mmseg is unavailable or the
    download fails.
    """
    try:
        from mmdeploy.codebase import import_codebase
        from mmdeploy.utils import Codebase
        import_codebase(Codebase.MMSEG)
    except ImportError:
        pytest.skip('mmseg is not installed')

    # Try well-known location first; fall back to downloading.
    model_config, checkpoint = _find_pidnet_files(_WORK_DIR_PIDNET)
    if model_config is None or checkpoint is None:
        download_dir = str(
            tmp_path_factory.mktemp('pidnet_download', numbered=False))
        try:
            _download_pidnet(download_dir)
        except subprocess.CalledProcessError as exc:
            pytest.skip(f'mim download failed: {exc}')
        model_config, checkpoint = _find_pidnet_files(download_dir)

    if model_config is None or checkpoint is None:
        pytest.skip('Could not locate PIDNet config or checkpoint.')

    from mmdeploy.apis import build_task_processor, torch2onnx
    from mmdeploy.apis.sdk import export2SDK
    from mmdeploy.codebase.mmseg.deploy.segmentation_model import \
        build_segmentation_model
    from mmdeploy.utils import load_config

    # ORT config: used for ONNX export and export2SDK (
    # sets backend=onnxruntime in deploy.json so the C++ SDK runtime
    # knows which inference backend to use).
    ort_deploy_cfg = _make_ort_deploy_cfg()
    model_cfg, ort_deploy_cfg = load_config(model_config, ort_deploy_cfg)
    model_cfg = _remove_preprocessor_pad_size(model_cfg)

    work_dir = str(
        tmp_path_factory.mktemp('pidnet_trt_artifacts', numbered=False))

    # 1. PyTorch → ONNX -------------------------------------------------------
    img = np.random.randint(0, 255, (INPUT_H, INPUT_W, 3), dtype=np.uint8)
    torch2onnx(img, work_dir, 'end2end.onnx', ort_deploy_cfg,
               copy.deepcopy(model_cfg), checkpoint, 'cpu')
    onnx_path = osp.join(work_dir, 'end2end.onnx')
    assert osp.exists(onnx_path), 'torch2onnx produced no ONNX file'

    # 2. Reference PyTorch model ----------------------------------------------
    task_processor = build_task_processor(model_cfg, ort_deploy_cfg, 'cpu')
    pytorch_model = task_processor.build_pytorch_model(checkpoint)
    pytorch_model.eval()

    # Reference ORT model
    ort_model = build_segmentation_model([onnx_path],
                                         model_cfg,
                                         ort_deploy_cfg,
                                         device='cpu')
    ort_model.eval()

    # 3. SDK metadata (pipeline.json / deploy.json / detail.json) -------------
    #    export2SDK is called with the ORT deploy cfg so deploy.json records
    #    backend=onnxruntime, which the C++ SDK runtime can load.
    #    build_segmentation_model uses _make_sdk_deploy_cfg()
    #    (model_type='sdk') to select the Python SDKEnd2EndModel wrapper.
    sdk_model = None
    if _sdk_available():
        _, sdk_deploy_cfg = load_config(model_config, _make_sdk_deploy_cfg())
        sdk_deploy_cfg = _remove_preprocessor_pad_size(sdk_deploy_cfg)
        # export2SDK reads onnx_config.input_shape to determine the Resize
        # scale written into pipeline.json.  With input_shape=None the
        # original training scale (2048×1024) is used, causing the C++ SDK
        # to upscale every 256×256 test image before inference.  Set the
        # actual test dimensions so pipeline.json gets the correct Resize.
        sdk_export_deploy_cfg = copy.deepcopy(ort_deploy_cfg)
        sdk_export_deploy_cfg.onnx_config.input_shape = [INPUT_W, INPUT_H]
        export2SDK(
            sdk_export_deploy_cfg,
            copy.deepcopy(model_cfg),
            work_dir,
            pth=checkpoint,
            device='cpu')

        sdk_model = build_segmentation_model([work_dir],
                                             copy.deepcopy(model_cfg),
                                             sdk_deploy_cfg,
                                             device='cpu')

    yield dict(
        work_dir=work_dir,
        onnx_path=onnx_path,
        task_processor=task_processor,
        pytorch_model=pytorch_model,
        sdk_model=sdk_model,
        deploy_cfg=ort_deploy_cfg,
        model_cfg=model_cfg,
        ort_model=ort_model,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPIDNetSDK:
    """Integration tests: PIDNet-S SDK output vs PyTorch reference.

    All tests share one module-scoped fixture so the slow ONNX export and SDK
    metadata generation happen only once per test session.
    """

    def test_onnx_exported(self, pidnet_trt_context):
        """Torch2onnx must produce an ONNX file."""
        assert osp.exists(pidnet_trt_context['onnx_path'])

    @pytest.mark.skipif(
        not _sdk_available(), reason='mmdeploy_runtime SDK not available')
    def test_sdk_output_similarity(self, pidnet_trt_context):
        """Pixel-level class agreement between SDK and PyTorch must be ≥ 95%.

        The SDK C++ pipeline (``mmdeploy_runtime.Segmentor``) reads a raw
        BGR uint8 numpy image, runs its own preprocessing internally, and
        returns a per-pixel label map.  We compare it against the PyTorch
        reference to catch regressions in the SDK pre/post-processing.
        """
        ctx = pidnet_trt_context
        sdk_model = ctx['sdk_model']
        assert sdk_model is not None, \
            'SDK model was not built (mmdeploy_runtime ' \
            'unavailable at fixture time)'

        task_processor = ctx['task_processor']
        pytorch_model = ctx['pytorch_model']
        ort_model = ctx['ort_model']

        # Use an actual Cityscapes image from demo/resources/cityscapes.png
        import cv2
        img_path = os.path.join(_REPO_ROOT, 'demo', 'resources',
                                'cityscapes.png')
        assert os.path.exists(
            img_path), f'Cityscapes image not found: {img_path}'
        img = cv2.imread(img_path)
        assert img is not None, f'Failed to load image: {img_path}'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)

        # --- Preprocessors for each model type ---
        # ORT preprocessor (use the same as PyTorch for consistency)
        ort_preprocessor = pytorch_model.data_preprocessor

        def sdk_preprocessor(img_np):
            # SDK expects float32 CHW, not normalized
            return torch.from_numpy(img_np.copy()).permute(2, 0, 1).float()

        # --- PyTorch reference -----------------------------------------------
        raw_batch, _ = task_processor.create_input(img.copy(),
                                                   [INPUT_H, INPUT_W])
        with torch.no_grad():
            pt_results = pytorch_model.test_step(raw_batch)
        pt_seg = pt_results[0].pred_sem_seg.data.cpu()  # [1, H, W] int64

        # --- ORT inference ---------------------------------------------------
        preprocessed_batch = ort_preprocessor(
            {
                'inputs': copy.deepcopy(raw_batch['inputs']),
                'data_samples': copy.deepcopy(raw_batch['data_samples'])
            }, False)
        ort_input = preprocessed_batch[
            'inputs']  # [N, C, H, W] normalized float
        data_samples = preprocessed_batch['data_samples']
        print('ORT input (normalized) shape:', ort_input.shape)
        print('ORT input (normalized) last 20 values:',
              ort_input.flatten()[-20:].cpu().numpy())
        with torch.no_grad():
            ort_results = ort_model.forward(ort_input, data_samples)
        ort_instances = ort_results[0].pred_sem_seg.data.cpu(
        )  # [1, H, W] int64

        # --- SDK inference ---------------------------------------------------
        img_chw = sdk_preprocessor(img)
        print('SDK input (float, not normalized) shape:', img_chw.shape)
        print('SDK input (float, not normalized) last 20 values:',
              img_chw.flatten()[-20:].cpu().numpy())
        data_samples = copy.deepcopy(raw_batch['data_samples'])
        with torch.no_grad():
            sdk_results = sdk_model.forward(img_chw, data_samples)
        sdk_seg = sdk_results[0].pred_sem_seg.data.cpu()  # [1, H, W] int64

        # print the last 20 values of each output tensor for debugging
        print('ORT output shape:', ort_instances.shape)
        print('ORT output last 20 values:',
              ort_instances.flatten()[-20:].cpu().numpy())
        print('SDK output shape:', sdk_seg.shape)
        print('SDK output last 20 values:',
              sdk_seg.flatten()[-20:].cpu().numpy())

        # --- Compare ---------------------------------------------------------
        assert pt_seg.shape == sdk_seg.shape, (
            f'Output shape mismatch: PyTorch {pt_seg.shape} vs '
            f'SDK {sdk_seg.shape}')

        agreement1 = (sdk_seg == ort_instances).float().mean().item()
        assert not (sdk_seg == 0).all(
        ), 'SDK output is all zeros, likely a preprocessing failure.'
        assert not (ort_instances == 0).all(
        ), 'ORT output is all zeros, likely a preprocessing failure.'
        assert not (
            pt_seg
            == 0).all(), 'PyTorch output is all zeros, likely a model failure.'

        agreement2 = (pt_seg == ort_instances).float().mean().item()
        agreement3 = (pt_seg == sdk_seg).float().mean().item()

        print(f'Pixel agreement: SDK vs ORT {agreement1:.4f}, '
              f'PyTorch vs ORT {agreement2:.4f}, '
              f'PyTorch vs SDK {agreement3:.4f}')
        assert agreement1 >= 0.99, (
            f'Pixel-level agreement is {agreement1:.4f}, expected ≥ 0.99. '
            'The ORT pipeline may have diverged from the SDK reference.')
        assert agreement2 >= 0.99, (
            f'Pixel-level agreement is {agreement2:.4f}, expected ≥ 0.99. '
            'The ORT pipeline may have diverged from the PyTorch reference.')
        assert agreement3 >= 0.99, (
            f'Pixel-level agreement is {agreement3:.4f}, expected ≥ 0.99. '
            'The SDK pipeline may have diverged from the PyTorch reference.')
