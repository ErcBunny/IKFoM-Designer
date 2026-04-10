import re

import casadi as ca


def list_to_vec(l: list[ca.MX]) -> ca.MX:
    """Vertically concatenate a list of CasADi expressions into one column vector."""
    if not l:
        return ca.MX.zeros(0, 1)
    return ca.vertcat(*l)


def as_column_vector(x: ca.MX) -> ca.MX:
    """Flatten a CasADi expression into one column vector."""
    return ca.reshape(x, x.numel(), 1)


def densify_mx(x: ca.MX) -> ca.MX:
    """Convert a CasADi expression to dense form when it is sparse."""
    if not x.is_dense():
        return ca.densify(x)
    return x


def sanitize_symbol_name(name: str, what: str) -> str:
    """Convert a user-facing name into a stable CasADi symbol suffix."""
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise ValueError(f"{what} must contain an alphanumeric character")
    if sanitized[0].isdigit():
        sanitized = f"m_{sanitized}"
    return sanitized


def print_dynamics_expr(expr_f: ca.MX, expr_df_dx: ca.MX, expr_df_dw: ca.MX):
    """Print the symbolic process-model expressions."""
    print("f =\n", expr_f, "\n")
    print("df/dx =\n", expr_df_dx, "\n")
    print("df/dw =\n", expr_df_dw, "\n")


def print_dynamics_func(f: ca.Function, df_dx: ca.Function, df_dw: ca.Function):
    """Print the generated process-model function signatures."""
    print(f, "with args:", "[*x, *u, w (vec), *p]")
    print(df_dx, "with args:", "[*x, dx (vec), *u, *p]")
    print(df_dw, "with args:", "[*x, *u, w (vec), *p]")


def print_measurement_expr(
    expr_h: ca.MX | list[ca.MX],
    expr_dh_dx: ca.MX,
    expr_dh_dv: ca.MX,
):
    """Print the symbolic measurement expressions for one block or combo."""
    if isinstance(expr_h, list):
        print("h outputs =")
        for index, output in enumerate(expr_h):
            print(f"  [{index}]")
            print(output)
    else:
        print("h =\n", expr_h, "\n")
    print("dh/dx =\n", expr_dh_dx, "\n")
    print("dh/dv =\n", expr_dh_dv, "\n")


def print_measurement_func(
    h: ca.Function,
    dh_dx: ca.Function,
    dh_dv: ca.Function,
    noise_label: str,
):
    """Print the generated measurement function signatures."""
    print(h, "with args:", f"[*x, {noise_label} (vec), *p]")
    print(dh_dx, "with args:", "[*x, dx (vec), *p]")
    print(dh_dv, "with args:", f"[*x, {noise_label} (vec), *p]")
