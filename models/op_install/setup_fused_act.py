from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='fused_act',
    ext_modules=[
        CUDAExtension(
            'fused_act',
            ['fused_bias_act.cpp', 'fused_bias_act_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)