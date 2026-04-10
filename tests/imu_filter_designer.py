import casadi as ca

from ikfmd import BaseDesigner, MeasBlockInfo, boxplus_dcm_small_d


class ImuFilterDesigner(BaseDesigner):

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

        return [
            MeasBlockInfo(
                name="gravity_body",
                noises=(ca.MX.sym("n_lin_acc", 3),),
                expr=gravity_body_expr,
                boxminus=lambda meas_perturbed, meas: meas_perturbed - meas,
            )
        ]

    def _define_measurement_combos(self) -> dict[str, tuple[str, ...]]:
        return {}

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


if __name__ == "__main__":
    designer = ImuFilterDesigner(
        "gen",
        dict(cpp=True, with_header=True, main=False, verbose=True, with_mem=False),
        True,
    )
    designer.print_expr()
    designer.print_func_io()
    designer.generate_code()
