# Third-party notices

## Build and runtime dependencies

Eigen is used for numerical evaluation and Levenberg–Marquardt optimization. Its primary license is MPL-2.0. pybind11 supplies the Python bindings under BSD-3-Clause. Full dependency license texts are available in `third_party/licenses/`.

Eigen 3.4.0 source is available at:
https://gitlab.com/libeigen/eigen/-/tree/3.4.0

pybind11 2.13.6 source is available at:
https://github.com/pybind/pybind11/tree/v2.13.6

CMake fetches these dependencies when a system installation is not provided. Their source trees are not redistributed in this source package.

NumPy, PyTorch and SymPy are installed as runtime dependencies and retain their respective licenses.

No source code from third-party benchmark implementations or private datasets is bundled. External benchmark datasets remain subject to their respective access and license terms.
