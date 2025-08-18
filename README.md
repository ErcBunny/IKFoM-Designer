# IKFoM Designer (`ikfmd`)

## Introduction

This is a small Python package that could be helpful for designing an [IKFoM](https://github.com/hku-mars/IKFoM). What it does is simple: it leverages [CasADi](https://web.casadi.org/)'s symbolic framework to model the system and its AD (auto differentiation) engine to calculate the Jacobian matrices required by the IKFoM toolkit.

## Installation

This package can be installed with `pip`: clone this repo and run `pip install .` at the repo root or simply:

```
pip install git+https://github.com/ErcBunny/IKFoM-Designer.git
```

## Usage

An example can be found in the `tests` directory and the code is self-explanatory.

In short, the user should implement a child class derived from the base class `BaseDesigner` by implementing all abstract methods, then call the `generate_code`. The user should expect generated code at the scripts runtime directory.

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

The generated code needs no dependency to build. An example of using the code with IKFoM can be found in this repo: https://github.com/ErcBunny/IMU-IKFoM.
