import casadi as ca


def list_to_vec(l: list[ca.MX]) -> ca.MX:
    return ca.vertcat(*l)
