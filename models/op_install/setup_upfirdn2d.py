# from setuptools import setup, Extension
# from torch.utils.cpp_extension import CUDA_HOME
# from torch.utils.cpp_extension import CppExtension
# from torch.utils.cpp_extension import CUDAExtension
# import torch
# import os 
# import glob

# this_dir = os.path.dirname(os.path.abspath(__file__))

# main_file = glob.glob(os.path.join(this_dir, "*.cpp"))
# source_cuda = glob.glob(os.path.join(this_dir, "*.cu"))

# sources = main_file

# extension = CppExtension
# extra_compile_args = {"cxx": []}
# define_macros = []

# if torch.cuda.is_available() and CUDA_HOME is not None:
#         extension = CUDAExtension
#         sources += source_cuda
#         define_macros += [("WITH_CUDA", None)]
#         extra_compile_args["nvcc"] = [
#             "-DCUDA_HAS_FP16=1",
#             "-D__CUDA_NO_HALF_OPERATORS__",
#             "-D__CUDA_NO_HALF_CONVERSIONS__",
#             "-D__CUDA_NO_HALF2_OPERATORS__",
#         ]
# else:
#     raise NotImplementedError('Cuda is not availabel')
# sources = [os.path.join(this_dir, s) for s in sources]
# include_dirs = [this_dir]

# setup(
#     name="fused", 
#     ext_modules = [
#         extension(
#             name="fused",
#             sources=sources,
#             include_dirs=include_dirs,
#             define_macros=define_macros,
#             extra_compile_args=extra_compile_args,
#         )
#     ],
#     cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
# )

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='upfirdn2d',
    ext_modules=[
        CUDAExtension(
            'upfirdn2d',
            ['upfirdn2d.cpp', 'upfirdn2d_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)