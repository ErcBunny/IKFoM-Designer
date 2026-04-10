from dataclasses import dataclass
from typing import Callable

import casadi as ca

from .util import (
    as_column_vector,
    densify_mx,
    list_to_vec,
    print_measurement_expr,
    print_measurement_func,
    sanitize_symbol_name,
)


@dataclass(frozen=True)
class MeasBlockInfo:
    """Static schema for one measurement block."""

    name: str
    noises: tuple[ca.MX, ...]
    expr: Callable[[list[ca.MX], list[ca.MX], list[ca.MX]], ca.MX]
    boxminus: Callable[[ca.MX, ca.MX], ca.MX]


class MeasBlock:
    """Build one measurement-block expression family and CasADi functions.

    Attributes:
        info: Static schema used to build the block.
        v: Block-local measurement noise symbols.
        v_vec: Vertically concatenated block-local measurement noise vector.
        dx_vec: Vertically concatenated state perturbation vector.
        expr_h: Measurement manifold expression ``h(x, v, p)``.
        expr_dh_dx: Jacobian expression
            ``d boxminus(h(x ⊞ dx, 0, p), h(x, 0, p)) / d dx``.
        expr_dh_dv: Jacobian expression
            ``d boxminus(h(x, v, p), h(x, 0, p)) / d v``.
        h: CasADi function wrapper for ``expr_h``.
        dh_dx: CasADi function wrapper for ``expr_dh_dx``.
        dh_dv: CasADi function wrapper for ``expr_dh_dv``.
    """

    def __init__(
        self,
        blk_info: MeasBlockInfo,
        p: list[ca.MX],
        x: list[ca.MX],
        dx: list[ca.MX],
        perturb_states: Callable[[list[ca.MX], list[ca.MX]], list[ca.MX]],
        enforce_dense: bool,
    ):
        """Build symbolic expressions and CasADi functions for one block."""

        self.info = blk_info

        # build measurement expression and local input vectors
        self.v: list[ca.MX] = list(blk_info.noises)
        self.v_vec = list_to_vec(self.v)
        self.dx_vec = list_to_vec(dx)
        self.expr_h: ca.MX = blk_info.expr(x, self.v, p)

        # get dh_dx expression
        zero_v = [ca.MX.zeros(*it.shape) for it in self.v]
        self.expr_dh_dx: ca.MX = ca.jacobian(
            as_column_vector(
                blk_info.boxminus(
                    blk_info.expr(perturb_states(x, dx), zero_v, p),
                    blk_info.expr(x, zero_v, p),
                )
            ),
            self.dx_vec,
        )

        # get dh_dv expression
        self.expr_dh_dv: ca.MX = ca.jacobian(
            as_column_vector(
                blk_info.boxminus(
                    blk_info.expr(x, self.v, p),
                    blk_info.expr(x, zero_v, p),
                )
            ),
            self.v_vec,
        )

        # densify expressions if specified
        if enforce_dense:
            self.expr_h = densify_mx(self.expr_h)
            self.expr_dh_dx = densify_mx(self.expr_dh_dx)
            self.expr_dh_dv = densify_mx(self.expr_dh_dv)

        # define CasADi functions
        function_suffix = sanitize_symbol_name(
            blk_info.name,
            "measurement block name",
        )
        self.h = ca.Function(
            f"h_block_{function_suffix}",
            [*x, self.v_vec, *p],
            [self.expr_h],
        )
        self.dh_dx = ca.Function(
            f"dh_dx_block_{function_suffix}",
            [*x, self.dx_vec, *p],
            [self.expr_dh_dx],
        )
        self.dh_dv = ca.Function(
            f"dh_dv_block_{function_suffix}",
            [*x, self.v_vec, *p],
            [self.expr_dh_dv],
        )

    def print_expr(self):
        """Print the symbolic block measurement expressions."""
        print_measurement_expr(self.expr_h, self.expr_dh_dx, self.expr_dh_dv)

    def print_func(self):
        """Print the generated CasADi function signatures."""
        print_measurement_func(self.h, self.dh_dx, self.dh_dv, "v")
