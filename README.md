# IKFoM Designer (`ikfmd`)

## Introduction

This is a small Python package that could be helpful for designing an [IKFoM](https://github.com/hku-mars/IKFoM). What it does is simple: it leverages [CasADi](https://web.casadi.org/)'s symbolic framework to model the system and its AD (auto differentiation) engine to calculate the Jacobian matrices required by the IKFoM toolkit.

## Installation

This package can be installed with `pip`: clone this repo and run `pip install .` at the repo root or simply:

```
pip install git+https://github.com/ErcBunny/IKFoM-Designer.git
```

## Usage

`ikfmd` builds three kinds of kernels:

- one process kernel family through `SysDynamics`
- one function family for each measurement block
- one function family for each measurement combo

The user implements a child class derived from `BaseDesigner`, then calls
`generate_code()`.

### Designer Interface

The child class defines:

- parameters: `_define_parameters()`
- states: `_define_states()`
- state perturbations: `_define_states_perturbation()`
- inputs: `_define_inputs()`
- process noises: `_define_process_noises()`
- measurement blocks: `_define_measurement_blocks()`
- measurement combos: `_define_measurement_combos()`
- process model: `_dyn(...)`
- state perturbation rule: `_perturb_states(...)`

Measurement blocks are declared with `MeasBlockInfo`. Each block owns its local
measurement-noise symbols, measurement expression, and `boxminus` rule.

Measurement combos are declared as:

```python
def _define_measurement_combos(self) -> dict[str, tuple[str, ...]]:
    return {
        "combo_name_0": ("meas_name_0", "meas_name_1"),
        "combo_name_1": ("meas_name_0", "meas_name_2"),
        # ...
    }
```

A combo does not define new mathematics. It only reuses already compiled
measurement blocks in the specified order.

### Minimal Example

Two reference examples live in `tests/`:

- `tests/imu_filter_designer.py`: block-only usage
- `tests/meas_combo_demo.py`: block + combo usage

The normal workflow is:

```python
from ikfmd import BaseDesigner


class Designer(BaseDesigner):
    # implement abstract methods
    pass


designer = Designer(
    "gen",
    dict(cpp=True, with_header=True, main=False, verbose=True, with_mem=False),
    True,
)
designer.print_expr()
designer.print_func_io()
designer.generate_code()
```

### Exported Function Names

The generated CasADi function names are:

- process kernels:
  - `f`
  - `df_dx`
  - `df_dw`
- measurement block kernels:
  - `h_block_<block_name>`
  - `dh_dx_block_<block_name>`
  - `dh_dv_block_<block_name>`
- measurement combo kernels:
  - `h_combo_<combo_name>`
  - `dh_dx_combo_<combo_name>`
  - `dh_dv_combo_<combo_name>`

Block and combo names are sanitized before they are used in function names.

### Generated Files

`generate_code()` writes the generated source files using the code-generator
name passed to the designer constructor. By default, CasADi writes them into
the current working directory. When `with_header=True`, CasADi also generates a
header file with the same basename.

For example, if the designer is constructed as:

```python
designer = Designer("gen", ...)
```

and the script is run from the repository root, the generated files are:

- `gen.cpp`
- `gen.h`

### IKFoM Integration

The exported kernels map to IKFoM as follows:

- use `f / df_dx / df_dw` for prediction
- use one block family or one combo family for fixed-measurement updates
- use runtime concatenation of block/combo kernels for dynamic-vector updates
- use runtime selection of one combo for dynamic-manifold updates

In generated C++, each CasADi function is called with the standard pattern:

- fill `const casadi_real* arg[...]`
- fill `casadi_real* res[...]`
- allocate `casadi_int iw[...]` and `casadi_real w[...]`
- call the exported symbol from `gen.h`

The input order is exactly the order printed by `designer.print_func_io()`.
For a hand-written wrapper example around generated CasADi kernels,
please take a look at
https://github.com/ErcBunny/IMU-IKFoM/blob/casadi/include/imu_ikfom/ikfom_formulation.hpp.

Use these signatures:

- `f`: `[*x, *u, w (vec), *p]`
- `df_dx`: `[*x, dx (vec), *u, *p]`
- `df_dw`: `[*x, *u, w (vec), *p]`
- `h_block_*`: `[*x, v (vec), *p]`
- `dh_dx_block_*`: `[*x, dx (vec), *p]`
- `dh_dv_block_*`: `[*x, v (vec), *p]`
- `h_combo_*`: `[*x, combo_v (vec), *p]`
- `dh_dx_combo_*`: `[*x, dx (vec), *p]`
- `dh_dv_combo_*`: `[*x, combo_v (vec), *p]`

Practical notes:

- a block `h` has one output
- a combo `h` is a multi-output function, one output per block in combo order
- `dh_dx_combo_*` is stacked by combo order
- `dh_dv_combo_*` is expressed in combo-local noise coordinates
- for dynamic-vector IKFoM updates, flatten and concatenate runtime-selected
  block/combo outputs in one consistent order
- for dynamic-manifold IKFoM updates, select one combo and map its multi-output
  `h` to the target measurement manifold type in your wrapper

The generated code has no external runtime dependency beyond the generated
files themselves.
