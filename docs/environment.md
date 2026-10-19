# Software environment and runtime

## Requirements and versions

ipldt requires Python 3.10 or later (Van Rossum and Drake 2009), NumPy (Harris et al. 2020), SciPy (Virtanen et al. 2020), and Numba (Lam et al. 2015), with CuPy (Okuta et al. 2017), ITK (McCormick et al. 2014), SimpleITK (Lowekamp et al. 2013), and ORMIR-XCT (Kuczynski et al. 2024) as optional extras. The validation ran in Python 3.11.15 with NumPy 2.3.5, SciPy 1.15.3, Numba 0.66.0, CuPy 14.2.0 (CUDA 12.9), ITK 5.4.6, SimpleITK 2.5.5, and ORMIR-XCT 1.1.0.

## Hardware and memory

The validation ran on a Windows 11 workstation with an AMD Ryzen Threadripper 9960X (24 cores), 96 GB of memory, and an NVIDIA GeForce RTX 4090 (24 GB); the distance transforms used the GPU back end in every validation record. Peak resident memory of the validation process, which holds IPL's products alongside ipldt's, was 10.35 GB.

## Runtime per scan

On the GPU back end of the workstation above, median times per radius or tibia scan (168 slices) were 0.7 s to read the AIM, 12.0 s for the compartment separation, 4.2 s for the two contour renderings, 1.9 s for the Laplace–Hamming filter, 2.9 s for the segmentation assembly, 4.9 s for the two pore cascades with their comparisons, and 28.0 s (range 4.2–69.8 s; ultradistal 39.1 s, diaphyseal 13.8 s) for the distance-transform stage, of which the 1/Tb.N map is the slowest (13.7 s); per patella, 18.9 s from AIM to segmentation and 36.2 s (21.2–101.9 s) for the maps.

## GPU and CPU back ends

The distance-transform sweep runs on the CPU (Numba); the containment test, the diameters, and the sphere drawing run on a GPU (CuPy) or on the CPU ([Distance-transform engine](implementation.md#distance-transform-engine)). The CPU back end gives the same result: 26 tests in [tests/test_gpu_cpu.py](../tests/test_gpu_cpu.py) and 17 in [tests/test_ridge_precision.py](../tests/test_ridge_precision.py), of the package's 434, assert that the two back ends give identical maps and sphere centers of dt_thickness, dt_spacing, and dt_number and equal reports, and that the ridge test decides near-ties as exact arithmetic does. The GPU back end was used in every validation record, so the identity of the two back ends rests on the test suite rather than on a second run of the validation set.

## References

- Harris CR, Millman KJ, van der Walt SJ, et al. (2020). Array programming with NumPy. Nature 585(7825): 357–362. doi:10.1038/s41586-020-2649-2.
- Kuczynski MT, Neeteson NJ, Stok KS, et al. (2024). ORMIR_XCT: a Python package for high resolution peripheral quantitative computed tomography image processing. J Open Source Softw 9(97): 6084. doi:10.21105/joss.06084.
- Lam SK, Pitrou A, Seibert S (2015). Numba: a LLVM-based Python JIT compiler. In: Proceedings of the Second Workshop on the LLVM Compiler Infrastructure in HPC (LLVM '15). New York, NY: ACM, 1–6. doi:10.1145/2833157.2833162.
- Lowekamp BC, Chen DT, Ibáñez L, Blezek D (2013). The design of SimpleITK. Front Neuroinform 7: 45. doi:10.3389/fninf.2013.00045.
- McCormick M, Liu X, Jomier J, Marion C, Ibáñez L (2014). ITK: enabling reproducible research and open science. Front Neuroinform 8: 13. doi:10.3389/fninf.2014.00013.
- Okuta R, Unno Y, Nishino D, Hido S, Loomis C (2017). CuPy: a NumPy-compatible library for NVIDIA GPU calculations. In: Proceedings of the Workshop on Machine Learning Systems (LearningSys) at the 31st Conference on Neural Information Processing Systems (NIPS 2017).
- Van Rossum G, Drake FL (2009). Python 3 Reference Manual. Scotts Valley, CA: CreateSpace.
- Virtanen P, Gommers R, Oliphant TE, et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Methods 17(3): 261–272. doi:10.1038/s41592-019-0686-2.
