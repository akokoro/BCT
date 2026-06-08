# Boundary-Condition Transformation for Electromagnetic Scattering

This repository provides a lightweight release of the code and selected numerical data for:

**Taming Complex Geometries in Electromagnetic Scattering: A Physics-Informed Surrogate Model via Boundary Condition Transformation**

The method uses a boundary-condition transformation (BCT) for physics-informed neural surrogate modeling of two-dimensional electromagnetic scattering. Instead of directly imposing magnetic-field boundary conditions involving normal derivatives, the surrogate learns the electric field under a Dirichlet-type condition and reconstructs the magnetic field through Maxwell curl relations when needed.

This release is intentionally compact. It keeps the main BCT training code and two representative electric-field comparison scripts. Pretrained model weights, the full paper figure suite, and extended post-processing scripts are not included.

## Contents

```text
.
|-- main_after_trans.py                    # Main BCT/RBF training and inference script
|-- rbf_net.py                             # RBF neural network modules
|-- shape.py                               # Geometry definitions and complexity metrics
|-- plot_3.2_figure3.py                    # Electric-field error comparison on six canonical geometries
|-- plot_engineering_electric_fields.py    # Electric-field comparison on tank and airplane geometries
|-- 3.1baseline/                           # Minimal precomputed predictions for the two plotting scripts
|-- comsol/                                # Minimal COMSOL reference fields used by the plotting scripts
|-- requirements.txt
`-- README.md
```

## Environment

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The code uses PyTorch, NumPy, SciPy, Matplotlib, and scikit-optimize. Install a PyTorch build matching your CUDA version if GPU acceleration is required.

## Training

The main training entry point is:

```bash
python main_after_trans.py
```

In `main_after_trans.py`, set:

```python
TRAIN_MODEL = True
```

to train a new BCT/RBF surrogate. New run outputs are written to `results/`. Model weights and generated outputs are ignored by Git by default.

## Plotting Examples

Run the scripts from the repository root.

To generate the canonical-geometry electric-field comparison:

```bash
python plot_3.2_figure3.py
```

This uses six COMSOL reference fields and minimal prediction files from PINN, GPINN, RoPINN, VS-PINN, and the proposed BCT/RBF model.

To generate the engineering-shape electric-field comparisons:

```bash
python plot_engineering_electric_fields.py
```

This uses the retained tank and airplane reference fields and the corresponding BCT/RBF prediction files.

Generated figures are written to the repository root and are ignored by Git.

## Data Scope

Only the data needed by the two retained plotting scripts are included:

- `comsol/`: eight COMSOL reference field files.
- `3.1baseline/`: thirty-three `prediction_data.txt` files for the canonical and engineering-shape electric-field comparisons.

The full training outputs, pretrained checkpoints, FEKO data, and all other analysis scripts were omitted to keep this release focused on the main implementation and representative visual validation.

## Citation

If you use this code, please cite the associated manuscript once available.
