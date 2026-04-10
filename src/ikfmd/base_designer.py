import abc
from abc import ABC

import casadi as ca

from .sys_dynamics import SysDynamics
from .meas_block import MeasBlockInfo, MeasBlock
from .meas_combo import MeasCombo


class BaseDesigner(ABC):

    def __init__(
        self,
        code_gen_name: str,
        code_gen_opts: dict = dict(cpp=True, with_header=True, main=False, verbose=True, with_mem=False),
        enforce_dense: bool = True,
    ):
        # define symbols
        self.p: list[ca.MX] = self._define_parameters()
        self.x: list[ca.MX] = self._define_states()
        self.dx: list[ca.MX] = self._define_states_perturbation()
        self.u: list[ca.MX] = self._define_inputs()
        self.w: list[ca.MX] = self._define_process_noises()
        self.meas_blk_info: list[MeasBlockInfo] = self._define_measurement_blocks()
        self.meas_combo: dict[str, tuple[str, ...]] = self._define_measurement_combos()
        assert len(self.x) != 0
        assert len(self.dx) != 0
        assert len(self.w) != 0
        assert len(self.x) == len(self.dx)

        # prepare f, df_dx, df_dw
        self.system_dynamics = SysDynamics(
            self.p,
            self.x,
            self.dx,
            self.u,
            self.w,
            self._dyn,
            self._perturb_states,
            enforce_dense,
        )

        # prepare measurement blocks
        self.meas_blk = []
        for info in self.meas_blk_info:
            self.meas_blk.append(
                MeasBlock(
                    blk_info=info,
                    p=self.p,
                    x=self.x,
                    dx=self.dx,
                    perturb_states=self._perturb_states,
                    enforce_dense=enforce_dense,
                )
            )

        # prepare measurement combos
        self.meas_blk_map: dict[str, MeasBlock] = {
            blk.info.name: blk for blk in self.meas_blk
        }
        self.meas_cmb: dict[str, MeasCombo] = {}
        for combo_name, block_names in self.meas_combo.items():
            self.meas_cmb[combo_name] = MeasCombo(
                name=combo_name,
                block_names=tuple(block_names),
                blocks=self.meas_blk_map,
                x=self.x,
                dx=self.dx,
                p=self.p,
                enforce_dense=enforce_dense,
            )

        # init code generator
        self.code_gen = ca.CodeGenerator(code_gen_name, code_gen_opts)
        self.code_gen.add(self.system_dynamics.f)
        self.code_gen.add(self.system_dynamics.df_dx)
        self.code_gen.add(self.system_dynamics.df_dw)
        for blk in self.meas_blk:
            self.code_gen.add(blk.h)
            self.code_gen.add(blk.dh_dx)
            self.code_gen.add(blk.dh_dv)
        for cmb in self.meas_cmb.values():
            self.code_gen.add(cmb.h)
            self.code_gen.add(cmb.dh_dx)
            self.code_gen.add(cmb.dh_dv)

    def generate_code(self):
        self.code_gen.generate()

    def print_symbols(self):
        print("p =", self.p)
        print("x =", self.x)
        print("dx =", self.dx)
        print("u =", self.u)
        print("w =", self.w)

    def print_expr(self):
        self.system_dynamics.print_expr()
        for blk in self.meas_blk:
            print(f"[block:{blk.info.name}]")
            blk.print_expr()
        for name, cmb in self.meas_cmb.items():
            print(f"[combo:{name}]")
            cmb.print_expr()

    def print_func_io(self):
        self.system_dynamics.print_func()
        for blk in self.meas_blk:
            print(f"[block:{blk.info.name}]")
            blk.print_func()
        for name, cmb in self.meas_cmb.items():
            print(f"[combo:{name}]")
            cmb.print_func()

    @abc.abstractmethod
    def _define_parameters(self) -> list[ca.MX]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_states(self) -> list[ca.MX]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_states_perturbation(self) -> list[ca.MX]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_inputs(self) -> list[ca.MX]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_process_noises(self) -> list[ca.MX]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_measurement_blocks(self) -> list[MeasBlockInfo]:
        raise NotImplementedError

    @abc.abstractmethod
    def _define_measurement_combos(self) -> dict[str, tuple[str, ...]]:
        raise NotImplementedError

    @abc.abstractmethod
    def _dyn(
        self,
        states: list[ca.MX],
        inputs: list[ca.MX],
        process_noises: list[ca.MX],
        parameters: list[ca.MX],
    ) -> ca.MX:
        raise NotImplementedError

    @abc.abstractmethod
    def _perturb_states(
        self,
        states: list[ca.MX],
        perturbation: list[ca.MX],
    ) -> list[ca.MX]:
        raise NotImplementedError
