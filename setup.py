import os

from packaging.version import parse as parse_version
from setuptools import setup

EXT_TYPE = ''
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension
    cmd_class = {'build_ext': BuildExtension}
    EXT_TYPE = 'torch'
except ModuleNotFoundError:
    cmd_class = {}
    print('Skip building ext ops due to the absence of torch.')
pwd = os.path.dirname(__file__)
version_file = 'mmdeploy/version.py'


def get_extensions():
    extensions = []
    ext_name = 'mmdeploy.backend.torchscript.ts_optimizer'

    if EXT_TYPE == 'torch':
        import glob
        import platform

        from torch.utils.cpp_extension import CppExtension

        try:
            import psutil
            num_cpu = len(psutil.Process().cpu_affinity())
            cpu_use = max(4, num_cpu - 1)
        except (ModuleNotFoundError, AttributeError):
            cpu_use = 4

        os.environ.setdefault('MAX_JOBS', str(cpu_use))
        define_macros = []

        # Before PyTorch1.8.0, when compiling CUDA code, `cxx` is a
        # required key passed to PyTorch. Even if there is no flag passed
        # to cxx, users also need to pass an empty list to PyTorch.
        # Since PyTorch1.8.0, it has a default value so users do not need
        # to pass an empty list anymore.
        # More details at https://github.com/pytorch/pytorch/pull/45956
        extra_compile_args = {'cxx': []}

        # c++14 is required.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['cxx'] = ['-std=c++14']
            else:
                extra_compile_args['cxx'] = ['-std=c++17']

        include_dirs = []

        op_files = glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/*.cpp'
        ) + glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/ir/*.cpp'
        ) + glob.glob(
            './csrc/mmdeploy/backend_ops/torchscript/optimizer/passes'
            '/onnx/*.cpp')
        extension = CppExtension

        # c++14 is required.
        # However, in the windows environment, some standard libraries
        # will depend on c++17 or higher. In fact, for the windows
        # environment, the compiler will choose the appropriate compiler
        # to compile those cpp files, so there is no need to add the
        # argument
        if 'nvcc' in extra_compile_args and platform.system() != 'Windows':
            if parse_version(torch.__version__) <= parse_version('1.12.1'):
                extra_compile_args['nvcc'] += ['-std=c++14']
            else:
                extra_compile_args['nvcc'] += ['-std=c++17']

        ext_ops = extension(
            name=ext_name,
            sources=op_files,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args)
        extensions.append(ext_ops)

    return extensions


if __name__ == '__main__':
    setup(ext_modules=get_extensions(), cmdclass=cmd_class, zip_safe=False)
