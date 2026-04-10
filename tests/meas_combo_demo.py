import casadi as ca

from ikfmd import (
    BaseDesigner,
    MeasBlockInfo,
    boxminus_dcm_small_d,
    boxplus_dcm_small_d,
    hat,
)


class MeasComboDemo(BaseDesigner):

    def _define_parameters(self) -> list[ca.MX]:
        return []

    def _define_states(self) -> list[ca.MX]:
        return [ca.MX.sym("R", 3, 3)]

    def _define_states_perturbation(self) -> list[ca.MX]:
        return [ca.MX.sym("dR", 3)]

    def _define_inputs(self) -> list[ca.MX]:
        return [ca.MX.sym("ang_vel", 3)]

    def _define_process_noises(self) -> list[ca.MX]:
        return [ca.MX.sym("n_ang_vel", 3)]

    def _define_measurement_blocks(self) -> list[MeasBlockInfo]:
        def gravity_body_expr(states, measurement_noises, parameters):
            # This unpacking order must stay aligned with `_define_states()`.
            R = states[0]
            return R.T @ ca.MX([0, 0, 1]) + measurement_noises[0]

        def attitude_identity_expr(states, measurement_noises, parameters):
            # This unpacking order must stay aligned with `_define_states()`.
            R = states[0]
            return R @ (ca.MX.eye(3) + hat(measurement_noises[0]))

        return [
            MeasBlockInfo(
                name="gravity_body",
                noises=(ca.MX.sym("n_lin_acc", 3),),
                expr=gravity_body_expr,
                boxminus=lambda meas_perturbed, meas: meas_perturbed - meas,
            ),
            MeasBlockInfo(
                name="attitude_identity",
                noises=(ca.MX.sym("n_rot", 3),),
                expr=attitude_identity_expr,
                boxminus=boxminus_dcm_small_d,
            ),
        ]

    def _define_measurement_combos(self) -> dict[str, tuple[str, ...]]:
        return {
            "gravity_only": ("gravity_body",),
            "attitude_only": ("attitude_identity",),
            "imu_measurement": ("gravity_body", "attitude_identity"),
        }

    def _dyn(
        self,
        states: list[ca.MX],
        inputs: list[ca.MX],
        process_noises: list[ca.MX],
        parameters: list[ca.MX],
    ) -> ca.MX:
        return inputs[0] + process_noises[0]

    def _perturb_states(
        self,
        states: list[ca.MX],
        perturbation: list[ca.MX],
    ) -> list[ca.MX]:
        return [boxplus_dcm_small_d(states[0], perturbation[0])]


def build_designer() -> MeasComboDemo:
    return MeasComboDemo(
        "combo_gen",
        dict(cpp=True, with_header=True, main=False, verbose=True, with_mem=False),
        True,
    )


def verify_designer(designer: MeasComboDemo) -> None:
    assert [blk.info.name for blk in designer.meas_blk] == [
        "gravity_body",
        "attitude_identity",
    ]
    assert tuple(designer.meas_cmb.keys()) == (
        "gravity_only",
        "attitude_only",
        "imu_measurement",
    )

    gravity_block = designer.meas_blk_map["gravity_body"]
    attitude_block = designer.meas_blk_map["attitude_identity"]
    imu_combo = designer.meas_cmb["imu_measurement"]

    assert gravity_block.h.name() == "h_block_gravity_body"
    assert gravity_block.dh_dx.name() == "dh_dx_block_gravity_body"
    assert gravity_block.dh_dv.name() == "dh_dv_block_gravity_body"

    assert attitude_block.h.name() == "h_block_attitude_identity"
    assert attitude_block.dh_dx.name() == "dh_dx_block_attitude_identity"
    assert attitude_block.dh_dv.name() == "dh_dv_block_attitude_identity"

    assert imu_combo.h.name() == "h_combo_imu_measurement"
    assert imu_combo.dh_dx.name() == "dh_dx_combo_imu_measurement"
    assert imu_combo.dh_dv.name() == "dh_dv_combo_imu_measurement"

    assert imu_combo.h.n_out() == 2
    assert imu_combo.h.size_out(0) == (3, 1)
    assert imu_combo.h.size_out(1) == (3, 3)
    assert imu_combo.dh_dx.size_out(0) == (6, 3)
    assert imu_combo.dh_dv.size_out(0) == (6, 6)


def main() -> None:
    designer = build_designer()
    verify_designer(designer)
    designer.print_expr()
    designer.print_func_io()
    designer.generate_code()


if __name__ == "__main__":
    main()
