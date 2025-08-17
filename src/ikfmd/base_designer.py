import abc
from abc import ABC

import casadi as ca

from .mx_util import list_to_vec


class BaseDesigner(ABC):

    def __init__(
        self,
        code_gen_name: str,
        code_gen_opts: dict,
        enforce_dense: bool = True,
    ):
        # define symbols
        self.p: list[ca.MX] = self._define_parameters()
        self.x: list[ca.MX] = self._define_states()
        self.dx: list[ca.MX] = self._define_states_perturbation()
        self.u: list[ca.MX] = self._define_inputs()
        self.w: list[ca.MX] = self._define_process_noises()
        self.v: list[ca.MX] = self._define_measurement_noises()
        assert len(self.x) != 0
        assert len(self.dx) != 0
        assert len(self.u) != 0
        assert len(self.w) != 0
        assert len(self.v) != 0

        # convert lists to vectors for jacobian calculation
        self.dx_vec = list_to_vec(self.dx)
        self.w_vec = list_to_vec(self.w)
        self.v_vec = list_to_vec(self.v)

        # calculate dynamics and measurement expressions
        self.expr_dyn = self._dyn(self.x, self.u, self.w, self.p)
        self.expr_meas = self._meas(self.x, self.v, self.p)

        # calculate jacobian
        self.jac_df_dx: ca.MX = ca.jacobian(
            self._dyn(
                self._perturb_states(self.x, self.dx),
                self.u,
                [ca.MX.zeros(w.shape) for w in self.w],
                self.p,
            ),
            self.dx_vec,
        )
        self.jac_df_dw: ca.MX = ca.jacobian(
            self.expr_dyn,
            self.w_vec,
        )
        self.jac_dh_dx: ca.MX = ca.jacobian(
            self._get_meas_perturbation(
                self._meas(
                    self._perturb_states(self.x, self.dx),
                    [ca.MX.zeros(v.shape) for v in self.v],
                    self.p,
                ),
                self._meas(self.x, [ca.MX.zeros(v.shape) for v in self.v], self.p),
            ),
            self.dx_vec,
        )
        self.jac_dh_dv: ca.MX = ca.jacobian(
            self._get_meas_perturbation(
                self._meas(self.x, self.v, self.p),
                self._meas(self.x, [ca.MX.zeros(v.shape) for v in self.v], self.p),
            ),
            self.v_vec,
        )

        # densify expressions if needed
        if enforce_dense:
            if not self.expr_dyn.is_dense():
                self.expr_dyn = ca.densify(self.expr_dyn)
            for i, h in enumerate(self.expr_meas):
                if not h.is_dense():
                    self.expr_meas[i] = ca.densify(h)
            if not self.jac_df_dx.is_dense():
                self.jac_df_dx = ca.densify(self.jac_df_dx)
            if not self.jac_df_dw.is_dense():
                self.jac_df_dw = ca.densify(self.jac_df_dw)
            if not self.jac_dh_dx.is_dense():
                self.jac_dh_dx = ca.densify(self.jac_dh_dx)
            if not self.jac_dh_dv.is_dense():
                self.jac_dh_dv = ca.densify(self.jac_dh_dv)

        # define functions
        self.f = ca.Function(
            "f",
            [*self.x, *self.u, self.w_vec, *self.p],
            [self.expr_dyn],
        )
        self.df_dx = ca.Function(
            "df_dx",
            [*self.x, self.dx_vec, *self.u, *self.p],
            [self.jac_df_dx],
        )
        self.df_dw = ca.Function(
            "df_dw",
            [*self.x, *self.u, self.w_vec, *self.p],
            [self.jac_df_dw],
        )
        self.h = ca.Function(
            "h",
            [*self.x, self.v_vec, *self.p],
            self.expr_meas,
        )
        self.dh_dx = ca.Function(
            "dh_dx",
            [*self.x, self.dx_vec, *self.p],
            [self.jac_dh_dx],
        )
        self.dh_dv = ca.Function(
            "dh_dv",
            [*self.x, self.v_vec, *self.p],
            [self.jac_dh_dv],
        )

        # init code generator
        self.code_gen = ca.CodeGenerator(code_gen_name, code_gen_opts)
        self.code_gen.add(self.f)
        self.code_gen.add(self.df_dx)
        self.code_gen.add(self.df_dw)
        self.code_gen.add(self.h)
        self.code_gen.add(self.dh_dx)
        self.code_gen.add(self.dh_dv)

    def generate_code(self):
        self.code_gen.generate()

    def print_expr(self):
        print("f =\n", self.expr_dyn, "\n")
        print("h =\n", self.expr_meas, "\n")
        print("df/dx =\n", self.jac_df_dx, "\n")
        print("df/dw =\n", self.jac_df_dw, "\n")
        print("dh/dx =\n", self.jac_dh_dx, "\n")
        print("dh/dv =\n", self.jac_dh_dv, "\n")

    def print_func_io(self):
        print(self.f, "with args:", "[*x, *u, w (vec), *p]")
        print(self.df_dx, "with args:", "[*x, dx (vec), *u, *p]")
        print(self.df_dw, "with args:", "[*x, *u, w (vec), *p]")
        print(self.h, "with args:", "[*x, v (vec), *p]")
        print(self.dh_dx, "with args:", "[*x, dx (vec), *p]")
        print(self.dh_dv, "with args:", "[*x, v (vec), *p]")

    @abc.abstractmethod
    def _define_parameters(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _define_states(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _define_states_perturbation(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _define_inputs(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _define_process_noises(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _define_measurement_noises(self) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _dyn(
        self,
        states: list[ca.MX],
        inputs: list[ca.MX],
        process_noises: list[ca.MX],
        parameters: list[ca.MX],
    ) -> ca.MX:
        pass

    @abc.abstractmethod
    def _meas(
        self,
        states: list[ca.MX],
        measurement_noises: list[ca.MX],
        parameters: list[ca.MX],
    ) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _perturb_states(
        self,
        states: list[ca.MX],
        perturbation: list[ca.MX],
    ) -> list[ca.MX]:
        pass

    @abc.abstractmethod
    def _get_meas_perturbation(
        self,
        meas_perturbed: list[ca.MX],
        meas: list[ca.MX],
    ) -> ca.MX:
        pass
