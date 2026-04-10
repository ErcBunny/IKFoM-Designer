from typing import Callable

import casadi as ca

from .util import densify_mx, list_to_vec, print_dynamics_expr, print_dynamics_func


class SysDynamics:
    """Build the process-model expressions and CasADi function family.

    Attributes:
        expr_f: Process model expression ``f(x, u, w, p)``.
        expr_df_dx: Jacobian expression
            ``d f(x ⊞ dx, u, 0, p) / d dx`` evaluated in local state
            perturbation coordinates.
        expr_df_dw: Jacobian expression ``d f(x, u, w, p) / d w``.
        f: CasADi function wrapper for ``expr_f``.
        df_dx: CasADi function wrapper for ``expr_df_dx``.
        df_dw: CasADi function wrapper for ``expr_df_dw``.
    """

    def __init__(
        self,
        p: list[ca.MX],
        x: list[ca.MX],
        dx: list[ca.MX],
        u: list[ca.MX],
        w: list[ca.MX],
        dyn: Callable[
            [list[ca.MX], list[ca.MX], list[ca.MX], list[ca.MX]],
            ca.MX,
        ],
        perturb_states: Callable[[list[ca.MX], list[ca.MX]], list[ca.MX]],
        enforce_dense: bool,
    ):
        # build casadi expression for system dynamics
        self.expr_f: ca.MX = dyn(x, u, w, p)

        # get df_dx expression
        dx_vec = list_to_vec(dx)
        zero_w = [ca.MX.zeros(it.shape) for it in w]
        self.expr_df_dx: ca.MX = ca.jacobian(
            dyn(perturb_states(x, dx), u, zero_w, p),
            dx_vec,
        )

        # get df_dw expression
        w_vec = list_to_vec(w)
        self.expr_df_dw: ca.MX = ca.jacobian(
            self.expr_f,
            w_vec,
        )

        # densify expressions if sepcified
        if enforce_dense:
            self.expr_f = densify_mx(self.expr_f)
            self.expr_df_dx = densify_mx(self.expr_df_dx)
            self.expr_df_dw = densify_mx(self.expr_df_dw)

        # define casadi functions
        self.f = ca.Function(
            "f",
            [*x, *u, w_vec, *p],
            [self.expr_f],
        )
        self.df_dx = ca.Function(
            "df_dx",
            [*x, dx_vec, *u, *p],
            [self.expr_df_dx],
        )
        self.df_dw = ca.Function(
            "df_dw",
            [*x, *u, w_vec, *p],
            [self.expr_df_dw],
        )

    def print_expr(self):
        print_dynamics_expr(self.expr_f, self.expr_df_dx, self.expr_df_dw)

    def print_func(self):
        print_dynamics_func(self.f, self.df_dx, self.df_dw)
