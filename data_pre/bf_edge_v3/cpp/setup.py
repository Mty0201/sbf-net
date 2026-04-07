"""Build bf_edge_cpp via cmake + pybind11.

Usage:
    cd data_pre/bf_edge_v3/cpp && pip install .
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        ext_dir = Path(self.get_ext_fullpath(ext.name)).parent.resolve()
        source_dir = Path(__file__).parent.resolve()

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={ext_dir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-Dpybind11_DIR={self._pybind11_cmake_dir()}",
        ]

        # Pass CONDA_PREFIX for Eigen3 lookup
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={conda_prefix}")

        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)

        subprocess.check_call(
            ["cmake", str(source_dir)] + cmake_args,
            cwd=build_dir,
        )
        subprocess.check_call(
            ["cmake", "--build", ".", "--config", "Release", "-j"],
            cwd=build_dir,
        )

    @staticmethod
    def _pybind11_cmake_dir() -> str:
        import pybind11
        return pybind11.get_cmake_dir()


setup(
    name="bf_edge_cpp",
    version="0.1.0",
    description="C++ accelerated bf_edge_v3 pipeline",
    ext_modules=[CMakeExtension("bf_edge_cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
