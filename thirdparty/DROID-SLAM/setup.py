from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
import os

ROOT = osp.dirname(osp.abspath(__file__))

# 针对 H200 的统一编译参数
h200_compile_args = [
    '-gencode=arch=compute_60,code=sm_60',
    '-gencode=arch=compute_61,code=sm_61',
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_75,code=sm_75',
    '-gencode=arch=compute_80,code=sm_80',
    '-gencode=arch=compute_86,code=sm_86',
    '-gencode=arch=compute_90,code=sm_90', # 核心：适配 H200 (Hopper)
]

setup(
    name='droid_and_lietorch', # 统一项目名称
    version='0.3',
    packages=find_packages(),
    ext_modules=[
        # 扩展模块 1: DROID-SLAM Backends
        CUDAExtension('droid_backends',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp', 
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3'] + h200_compile_args
            }),

        # 扩展模块 2: Lietorch Backends (从子目录引入)
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'), 
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp', 
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2'] + h200_compile_args
            }),
            
        # 扩展模块 3: Lietorch Extras (核心算子如 altcorr 在此)
        CUDAExtension('lietorch_extras', 
            sources=[
                'thirdparty/lietorch/lietorch/extras/altcorr_kernel.cu',
                'thirdparty/lietorch/lietorch/extras/corr_index_kernel.cu',
                'thirdparty/lietorch/lietorch/extras/se3_builder.cu',
                'thirdparty/lietorch/lietorch/extras/se3_inplace_builder.cu',
                'thirdparty/lietorch/lietorch/extras/se3_solver.cu',
                'thirdparty/lietorch/lietorch/extras/extras.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O2'], 
                'nvcc': ['-O2'] + h200_compile_args
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)