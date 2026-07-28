# Scaling Limits of Multichannel Spectral Routers for Snapshot Imaging — Verification Package

Data and evaluation code accompanying:

> J. Han, S. Lee, D. Kim, and H. Chung, *“Scaling Limits of Multichannel
> Spectral Routers for Snapshot Imaging,”* Hanyang University.

This repository is a **result-verification package**. It provides the FDTD
evaluation scripts, configuration metadata, and environment specification
needed to reproduce the routing-efficiency values reported in the paper. The
companion dataset, distributed separately through Google Drive, contains the
final optimized designs, per-channel routing-efficiency spectra, focal-plane
maps, and structure renders.

The inverse-design optimization code is not included because the design method
is the subject of a pending patent application.

All devices are binary **TiO₂/SiO₂** dielectric routers evaluated by FDTD at a
resolution of **50 pixels/µm** (`dx = 0.02 µm`) under normal incidence, with
periodic boundaries in *x* and *y* and perfectly matched layers (PMLs) in *z*.
Unless otherwise noted, the device height is **h = 2.0 µm**. The incident field
contains equal-amplitude, in-phase \(E_x\) and \(E_y\) components.

## Repository layout

```text
.
├── 9CH_OE.py
├── 9CH_OE_compressed.py
├── 16CH_OE.py
├── 16CH_OE_compressed.py
├── 25CH_OE.py
├── 25CH_OE_compressed.py
├── 36CH_OE.py
├── 36CH_OE_arrangement.py
├── 36CH_OE_VIS_NIR.py
├── configs.csv
├── environment.yml
├── LICENSE
└── README.md
```

The baseline and condition-specific evaluation scripts are kept in the
repository root. The large design and simulation files are distributed
separately to keep the Git repository lightweight.

## Dataset download

- **Download:** [Google Drive](https://drive.google.com/file/d/1btFerqUMPD0_WF19Riu1CgCxhb3mPP0H/view?usp=sharing)
- **Archive:** `spectral_router_data_v1.0.zip`
- **Dataset version:** `v1.0`

Download the archive and extract it at the repository root. The resulting
layout should be:

```text
data/
├── DATA_LICENSE.txt
├── 9ch/
├── 16ch/
├── 25ch/
├── 36ch/
├── compressed/
├── arrangement/
└── nir/
```

Each configuration folder is named:

```text
{N}ch_p{P}_h{H}[_variant]
```

For example:

```text
36ch_p340_h2000
9ch_p340_h2000_compressed
```

Here, `N` is the channel count of a \(k\times k\) super-pixel
(\(k=3,4,5,6\)), `P` is the sub-pixel pitch in nanometers, and `H` is the
device height in nanometers.

## Dataset organization

Each configuration folder contains:

```text
{config}/
├── final_design.txt
├── E2_plot/
│   ├── Sz_XY_all_{k}x{k}.png
│   └── Sz_XY_band##.png
└── ucell/
    ├── Design_3D.png
    ├── Design_XY.png
    ├── Design_XZ.png
    ├── OE_abs.png
    └── OE_abs_data.csv
```

- `final_design.txt` contains the optimized TiO₂/SiO₂ structure.
- `E2_plot/` is a legacy directory name retained for compatibility; its files
  show the focal-plane \(S_z\) distributions.
- `OE_abs_data.csv` contains the per-channel routing-efficiency spectra used to
  obtain the values reported in the paper.

## `final_design.txt` format

Each file contains one `MaterialGrid` weight per line, flattened in NumPy C
order. The released designs are binarized, with `1` representing TiO₂ and `0`
representing SiO₂. The evaluation scripts load these weights directly without
additional thresholding.

Reshape the data to the FDTD design grid `(Nx, Ny, Nz)`, where `Nx` and `Ny`
span the \(k\times k\) super-pixel and `Nz` spans the device height:

```python
import numpy as np

structure_weight = np.loadtxt("final_design.txt")
structure_weight = structure_weight.reshape((Nx, Ny, Nz), order="C")
design_variables.update_weights(structure_weight)
```

The grid spacing is `dx = 0.02 µm`. The number of values therefore equals
`Nx * Ny * Nz`. For example, the 36-channel design with
\(P=340\ \mathrm{nm}\) and \(h=2.0\ \mathrm{µm}\) uses a
`103 × 103 × 101` grid, corresponding to 1,071,509 values. See `configs.csv`
for the dimensions and geometry of each configuration.

## Target-region geometry

Each sub-pixel has pitch `P`. Routing efficiency is integrated over a square
target aperture of side length `PD`. Adjacent target apertures are separated by
a gap \(g=P-\mathrm{PD}\), denoted `DTI_size` in the released scripts. The
detector and physical deep-trench-isolation structures are not explicitly
modeled; `PD_size` and `DTI_size` define the output-plane flux-monitor
geometry.

| P (nm) | Pitch (µm) | `DTI_size` (µm) | `PD_size` (µm) |
|:------:|:----------:|:---------------:|:--------------:|
| 120 | 0.12 | 0.04 | 0.08 |
| 160 | 0.16 | 0.04 | 0.12 |
| 200 | 0.20 | 0.04 | 0.16 |
| 240 | 0.24 | 0.04 | 0.20 |
| 280 | 0.28 | 0.04 | 0.24 |
| 340 | 0.34 | 0.04 | 0.30 |
| 500 | 0.50 | 0.10 | 0.40 |
| 600 | 0.60 | 0.12 | 0.48 |

## Target wavelengths

Target wavelengths are assigned in row-major order over the \(k\times k\)
sub-pixel array.

| N | Array | Range (nm) | Nominal spacing |
|:-:|:-----:|:----------:|:---------------:|
| 9  | 3×3 | 430–670 | 30 nm |
| 16 | 4×4 | 410–690 | 20 nm (mixed 15/20 nm) |
| 25 | 5×5 | 406–694 | 12 nm |
| 36 | 6×6 | 404–696 | 8 nm (mixed 8/9 nm) |

The variants are:

- `compressed`: uses a nominal 8 nm spacing. After rounding to integer
  nanometers, the realized wavelength steps are 8 or 9 nm.
- `arrangement`: applies the permutation
  \(p(j)=19j\bmod 36\), \(j=0,\ldots,35\), to the 36-channel baseline to
  maximize the mean wavelength separation between spatially adjacent
  sub-pixels.
- `nir`: uses `np.linspace(0.404, 0.996, 36)` µm.

The complete ordered wavelength list for every configuration is provided in
`configs.csv`.

## Installation

Create the tested Conda environment:

```bash
conda env create -f environment.yml
conda activate spectral-router
```

The package was tested with Python 3.13, Conda package `pymeep` 1.32.0
(`meep.__version__` reports `1.32.0`), MPICH 4.3.2, and `mpi4py` 4.1.1.

## Running the evaluation

First download and extract the companion dataset as described above. Then run
the matching evaluation script from inside a configuration folder. For
example:

```bash
cd data/36ch/36ch_p340_h2000
mpirun -np 8 python ../../../36CH_OE.py
```

The script reads `final_design.txt` from the current directory and writes the
calculated spectra and focal-plane maps to `ucell/` and `E2_plot/`,
respectively.

For a configuration whose parameters differ from those encoded in a baseline
script, set `design_region_x`, `design_region_y`, `PD_size`, `DTI_size`, device
height, and target wavelengths according to `configs.csv`. The corresponding
condition-specific scripts are provided in the repository root:

- `9CH_OE_compressed.py`, `16CH_OE_compressed.py`, and
  `25CH_OE_compressed.py` for the compressed-spacing controls
- `36CH_OE_arrangement.py` for the wavelength-arrangement variant
- `36CH_OE_VIS_NIR.py` for the visible–near-infrared extension

MPI is recommended for practical run times. The number of processes may be
adjusted to match the available hardware.

## `configs.csv`

`configs.csv` contains one row per configuration with the following fields:

```text
config, N, k, P_nm, H_nm, variant, design_region_um, DTI_um, PD_um,
n_wl, wl_min_um, wl_max_um, wavelengths_um, script_used, note
```

The `wavelengths_um` field gives the complete ordered wavelength list assigned
to the sub-pixels.

## Efficiency definitions

The per-channel routing efficiency \(T_{ii}(\lambda)\) is the \(z\)-directed
Poynting flux through the target sub-pixel divided by the incident power
obtained from the corresponding no-structure normalization simulation. The
paper's per-channel value is \(T_{ii}\) linearly interpolated at the target
wavelength \(\lambda_i\).

The average routing efficiency is

\[
\bar{T}=\frac{1}{N}\sum_{i=1}^{N}T_{ii}(\lambda_i),
\]

and the worst-channel efficiency is

\[
T_{\mathrm{worst}}=\min_i T_{ii}(\lambda_i).
\]

The per-channel focal-plane maps are independently normalized for
visualization and should not be used for quantitative comparisons between
channels. Quantitative efficiencies are provided in `OE_abs_data.csv`.

## Requirements

- Meep with MPI and adjoint support
- mpi4py
- NumPy
- SciPy
- Matplotlib
- h5py
- autograd
- pandas

Exact tested versions and package constraints are listed in `environment.yml`.

## Citation

If you use these designs or evaluation scripts, please cite:

> J. Han, S. Lee, D. Kim, and H. Chung, “Scaling Limits of Multichannel
> Spectral Routers for Snapshot Imaging,” arXiv preprint, 2026.
> [2607.23508]

## License

Copyright (c) 2026 Hanyang University and the authors.

The evaluation scripts and related code in this repository are licensed under
the MIT License. See `LICENSE` for details.

The companion optimized-device and simulation dataset is not covered by the
MIT License. It is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License
(CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). The complete
data-license notice is included as `data/DATA_LICENSE.txt`.

CC BY-NC 4.0 permits use, modification, and redistribution of the dataset for
non-commercial purposes with appropriate attribution. Commercial use requires
separate permission from the copyright holder.

No patent rights are granted. The inverse-design optimization method and its
implementation are not included and may be subject to pending patent
applications.
