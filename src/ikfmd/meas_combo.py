import casadi as ca

from .meas_block import MeasBlock
from .util import (
    densify_mx,
    list_to_vec,
    print_measurement_expr,
    print_measurement_func,
    sanitize_symbol_name,
)


class MeasCombo:
    """Compile one ordered composition of measurement blocks.

    A combo reuses already compiled ``MeasBlock`` objects. It preserves the
    measurement-manifold structure of ``h`` as one CasADi multi-output function
    while stacking block Jacobians in combo order.

    Attributes:
        name: User-facing combo name.
        block_names: Ordered block names referenced by the combo.
        blocks: Ordered compiled block instances used by the combo.
        dx_vec: Vertically concatenated state perturbation vector.
        v_vec: Vertically concatenated combo-local measurement noise vector.
        expr_h: Measurement manifold expressions in combo order.
        expr_dh_dx: Jacobian expression formed by vertically stacking each
            block's ``dh_dx`` contribution.
        expr_dh_dv: Jacobian expression formed by block-diagonal stacking of
            each block's ``dh_dv`` contribution in combo-local noise coordinates.
        h: CasADi function wrapper for ``expr_h``.
        dh_dx: CasADi function wrapper for ``expr_dh_dx``.
        dh_dv: CasADi function wrapper for ``expr_dh_dv``.
    """

    def __init__(
        self,
        name: str,
        block_names: tuple[str, ...],
        blocks: dict[str, MeasBlock],
        x: list[ca.MX],
        dx: list[ca.MX],
        p: list[ca.MX],
        enforce_dense: bool,
    ):
        if not name:
            raise ValueError("measurement combo name must not be empty")
        if not block_names:
            raise ValueError("measurement combo must contain at least one block")
        if len(set(block_names)) != len(block_names):
            raise ValueError(
                f"measurement combo contains duplicate block names: {name}"
            )

        missing = [block_name for block_name in block_names if block_name not in blocks]
        if missing:
            raise ValueError(
                f"measurement combo {name} references unknown block(s): {missing}"
            )

        # gather blocks and combo-local input vectors
        self.name = name
        self.block_names = tuple(block_names)
        self.blocks = [blocks[block_name] for block_name in self.block_names]
        self.dx_vec = list_to_vec(dx)
        self.v_vec = list_to_vec([blk.v_vec for blk in self.blocks])

        # build combo expressions in block order
        self.expr_h = [
            self._remap_block_noise_expr(blk.expr_h, blk) for blk in self.blocks
        ]
        self.expr_dh_dx = ca.vertcat(*[blk.expr_dh_dx for blk in self.blocks])
        self.expr_dh_dv = ca.diagcat(
            *[self._remap_block_noise_expr(blk.expr_dh_dv, blk) for blk in self.blocks]
        )

        # densify expressions if specified
        if enforce_dense:
            self.expr_h = [densify_mx(expr_h) for expr_h in self.expr_h]
            self.expr_dh_dx = densify_mx(self.expr_dh_dx)
            self.expr_dh_dv = densify_mx(self.expr_dh_dv)

        # define CasADi functions
        function_suffix = sanitize_symbol_name(
            name,
            "measurement combo name",
        )
        self.h = ca.Function(
            f"h_combo_{function_suffix}",
            [*x, self.v_vec, *p],
            self.expr_h,
        )
        self.dh_dx = ca.Function(
            f"dh_dx_combo_{function_suffix}",
            [*x, self.dx_vec, *p],
            [self.expr_dh_dx],
        )
        self.dh_dv = ca.Function(
            f"dh_dv_combo_{function_suffix}",
            [*x, self.v_vec, *p],
            [self.expr_dh_dv],
        )

    def print_expr(self):
        """Print the symbolic combo measurement expressions."""
        print_measurement_expr(self.expr_h, self.expr_dh_dx, self.expr_dh_dv)

    def print_func(self):
        """Print the generated CasADi function signatures."""
        print_measurement_func(self.h, self.dh_dx, self.dh_dv, "combo_v")

    def _remap_block_noise_expr(self, expr: ca.MX, blk: MeasBlock) -> ca.MX:
        """Replace one block's local noise symbols with combo-local slices."""
        combo_noise = []
        offset = 0
        for combo_blk in self.blocks:
            if combo_blk is blk:
                break
            offset += int(combo_blk.v_vec.numel())

        for noise_symbol in blk.v:
            noise_dim = int(noise_symbol.numel())
            noise_slice = self.v_vec[offset : offset + noise_dim]
            combo_noise.append(ca.reshape(noise_slice, *noise_symbol.shape))
            offset += noise_dim

        remapped = expr
        for noise_symbol, combo_noise_symbol in zip(blk.v, combo_noise):
            remapped = ca.substitute(remapped, noise_symbol, combo_noise_symbol)
        return remapped
