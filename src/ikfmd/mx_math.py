import casadi as ca


def hat(v: ca.MX) -> ca.MX:
    return ca.skew(v)


def hat_square(v: ca.MX) -> ca.MX:
    return v @ v.T - (ca.norm_2(v) ** 2) * ca.MX.eye(3)


def boxplus_dcm_small_d(dcm: ca.MX, delta: ca.MX) -> ca.MX:
    return dcm @ (ca.MX.eye(3) + hat(delta))


def boxminus_dcm_small_d(dcm_perturbed: ca.MX, dcm: ca.MX) -> ca.MX:
    d = dcm.T @ dcm_perturbed
    return ca.inv_skew(0.5 * (d - d.T))


def dcm_from_axis_angle(axis: ca.MX, angle: ca.MX) -> ca.MX:
    return (
        ca.MX.eye(3)
        + ca.sin(angle) * hat(axis)
        + (1 - ca.cos(angle)) * hat_square(axis)
    )


def dcm_x(angle: ca.MX):
    return dcm_from_axis_angle(ca.MX([1, 0, 0]), angle)


def dcm_y(angle: ca.MX):
    return dcm_from_axis_angle(ca.MX([0, 1, 0]), angle)


def dcm_z(angle: ca.MX):
    return dcm_from_axis_angle(ca.MX([0, 0, 1]), angle)
