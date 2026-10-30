# Release validation

Validated on Windows x64 with Python 3.12, MSVC 19.29, NumPy 1.26.4,
PyTorch 2.14 CPU, Eigen 3.4, and pybind11 2.13.6.

- 33 automated tests cover data splits, test trimming and normalization,
  non-finite predictions, FiLM initialization and gradients, legal action
  probabilities, elite learning, ancestry deduplication, bounded trust and
  conflict shutdown, native evaluation budgets, fixed fitted constants, and
  the round-based search loop.
- The native extension was compiled from the supplied C++ sources.
- A wheel was built and installed in a separate target directory. The CLI
  completed an offline two-round run from that installation using an explicit
  fixed prior; 240 candidate evaluations and actual inheritance, distillation,
  GP events and trust calibration were recorded.
- Release packaging excludes environment files, local paths, cached data,
  experimental logs, checkpoints, build products and unrelated projects.
- Full-budget benchmark runs with Qwen3-4B or Llama-3.1-8B are not part of these
  engineering checks. Historical result tables are not inferred from them.

To reproduce the component checks after installing the package:

```bash
python -m pip install -e '.[test]'
python -m pytest
```
