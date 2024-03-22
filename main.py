from typing import Any, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def cosspace(start: float, stop: float, n: int) -> np.ndarray[Any]:
    midpoint: float = (stop - start) / 2
    n_array: np.ndarray = np.arange(0, n, 1)

    return start + midpoint * (1 - np.cos(np.pi / (n - 1) * n_array))



class Airfoil:
    def __init__(self, polar_file: str):
        self.filename = polar_file
        self.polar_data = pd.read_csv(
            f"{polar_file}", header=0, names=["alpha", "cl", "cd", "cm"], sep="\t"
        )

    def get_polar_data(self):
        return self.polar_data

    def plot_polar_data(self) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()
        ax1.plot(
            self.polar_data["alpha"],
            self.polar_data["cl"],
            linestyle="-",
            color="k",
            label=r"$C_l$",
        )
        ax1.plot(
            self.polar_data["alpha"],
            self.polar_data["cm"],
            linestyle="--",
            color="r",
            label=r"$C_m$",
        )
        ax1.set(xlabel=r"$\alpha$", ylabel=r"$C_l$, $C_m$")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.polar_data["cd"], self.polar_data["cl"], linestyle="-", color="k")
        ax2.set(xlabel=r"$C_d$", ylabel=r"$C_l$")
        ax2.grid(True)
        plt.show()

    def calc_cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """
        Computes lift coefficient at a given angle of attack

        alpha: Airfoil angle of attack in degrees.
        """
        return np.interp(alpha, self.polar_data["alpha"], self.polar_data["cl"])

    def calc_cd(self, alpha: float | np.ndarray) -> float | np.ndarray:
        """
        Computes drag coefficient at a given angle of attack

        alpha: Airfoil angle of attack in degrees.
        """
        return np.interp(alpha, self.polar_data["alpha"], self.polar_data["cd"])


class Blade:
    def __init__(
        self,
        blade_airfoil: Airfoil,
        twist: Callable[[np.ndarray | float], np.ndarray | float],
        chord: Callable[[np.ndarray | float], np.ndarray | float],
    ):
        self.blade_airfoil = blade_airfoil
        self.twist = twist
        self.chord = chord


class Turbine:
    def __init__(
        self,
        turbine_blade: Blade,
        NBlades: int,
        pitch: float,
        radius: float,
        tipradius_R: float,
        rootradius_R: float,
        n_annuli: float,
        spacing="constant",
    ):
        self.turbine_blade = turbine_blade
        self.B = NBlades
        self.pitch = pitch
        self.radius = radius
        self.tipradius_R = tipradius_R
        self.rootradius_R = rootradius_R

        if spacing == "constant":
            self.r_Rs: np.ndarray = np.linspace(
                RootLocation_R, TipLocation_R, n_annuli + 1
            )
        elif spacing == "cosine":
            self.r_Rs: np.ndarray = cosspace(
                RootLocation_R, TipLocation_R, n_annuli + 1
            )

        self.dr = self.r_Rs[1:] - self.r_Rs[:-1]


class BemSimulation:
    def __init__(
        self,
        turbine: Turbine,
        uinf: float,
        tsr: float,
        rho: float,
        p_inf: float,
        tip_correction=True,
        glauert_correction=True,
        n_iterations=100,
        iteration_error=0.00001,
    ):
        self.turbine = turbine
        self.blade = self.turbine.turbine_blade
        self.airfoil = self.blade.blade_airfoil
        self.uinf = uinf
        self.tsr = tsr
        self.omega = uinf * tsr / self.turbine.radius
        self.rho = rho
        self.pinf = p_inf
        self.tip_corr = tip_correction
        self.glauert_corr = glauert_correction
        self.n_iter = n_iterations
        self.iter_error = iteration_error
        if tip_correction:
            self.results: np.ndarray = np.zeros([len(self.turbine.r_Rs) - 1, 19])
        else:
            self.results: np.ndarray = np.zeros([len(self.turbine.r_Rs) - 1, 16])

    def calc_axial_induction(self, CT: np.ndarray) -> np.ndarray:
        a: np.ndarray = np.zeros(shape=np.shape(CT))
        if self.glauert_corr:
            CT1: float = 1.816
            CT2: float = 2 * np.sqrt(CT1) - CT1
            a[CT >= CT2] = 1 + (CT[CT >= CT2] - CT1) / (4 * (np.sqrt(CT1) - 1))
            a[CT <= CT2] = 0.5 - 0.5 * np.sqrt(1 - CT[CT < CT2])
        else:
            a = 0.5 * (1 - np.sqrt(1 - CT))
        return a

    def calc_azimuthal_induction(self, f_tan: float, a_ax: float, rR: float) -> float:
        return (
            f_tan
            * self.turbine.B
            / (
                2
                * self.rho
                * np.pi
                * self.uinf
                * (1 - a_ax)
                * self.omega
                * 2
                * (rR * self.turbine.radius) ** 2
            )
        )

    def _load_blade_element(
        self,
        vnorm: float | np.ndarray,
        vtan: float | np.ndarray,
        r_R: float | np.ndarray,
    ) -> Tuple[float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chord: float = self.blade.chord(r_R)
        vmag2: float = vnorm**2 + vtan**2
        phi: float = np.arctan2(vnorm, vtan)
        alpha: float = self.blade.twist(r_R) + self.turbine.pitch + np.degrees(phi)
        cl: float = self.airfoil.calc_cl(alpha)
        cd: float = self.airfoil.calc_cd(alpha)
        lift: float = 0.5 * self.rho * vmag2 * cl * chord
        drag: float = 0.5 * self.rho * vmag2 * cd * chord
        fnorm: float = lift * np.cos(phi) + drag * np.sin(phi)
        ftan: float = lift * np.sin(phi) - drag * np.cos(phi)
        gamma: float = 0.5 * np.sqrt(vmag2) * cl * chord
        return fnorm, ftan, gamma, phi, alpha, cl

    def _prandtl_tip_root_correction(
        self, r_R: float, ax_ind: float
    ) -> Tuple[float, float, float]:
        temp1 = (
            -self.turbine.B
            / 2
            * (self.turbine.tipradius_R - r_R)
            / r_R
            * np.sqrt(1 + ((self.tsr * r_R) ** 2) / ((1 - ax_ind) ** 2))
        )
        Ftip = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
        Ftip[np.isnan(Ftip)] = 0
        temp1 = (
            self.turbine.B
            / 2
            * (self.turbine.rootradius_R - r_R)
            / r_R
            * np.sqrt(1 + ((self.tsr * r_R) ** 2) / ((1 - ax_ind) ** 2))
        )
        Froot = np.array(2 / np.pi * np.arccos(np.exp(temp1)))
        Froot[np.isnan(Froot)] = 0

        F_total: float = Froot * Ftip
        prandtl_min: float = 1e-4
        if F_total < prandtl_min:
            F_total = prandtl_min

        return F_total, Ftip, Froot

    def _solve_streamtube(
        self, r1_R: float, r2_R: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        area: float = np.pi * (
            (r2_R * self.turbine.radius) ** 2 - (r1_R * self.turbine.radius) ** 2
        )
        r_R: float = (r1_R + r2_R) / 2

        a: float = 0.0
        aprime: float = 0.0

        solving: bool = True

        iterations: int = 0
        while solving:
            urotor: float = self.uinf * (1 - a)
            utan: float = (1 + aprime) * self.omega * r_R * self.turbine.radius
            fnorm, ftan, gamma, phi, alpha, cl = self._load_blade_element(
                urotor, utan, r_R
            )
            load_axial: float = (
                fnorm * self.turbine.radius * (r2_R - r1_R) * self.turbine.B
            )
            load_tan: float = (
                ftan * self.turbine.radius * (r2_R - r1_R) * self.turbine.B
            )
            power: float = load_axial * urotor

            CT: float = load_axial / (0.5 * area * self.rho * self.uinf**2)
            Cn: float = load_axial / (
                0.5 * self.rho * self.uinf**2 * self.turbine.radius
            )
            Ct: float = load_tan / (
                0.5 * self.rho * self.uinf**2 * self.turbine.radius
            )
            Cp: float = power / (0.5 * self.rho * self.uinf**3 * area)
            CQ: float = Cp / self.tsr

            anew: float = self.calc_axial_induction(CT)

            if self.tip_corr:
                prandtl, prandtltip, prandtlroot = self._prandtl_tip_root_correction(
                    r_R, anew
                )
                anew: float = anew / prandtl

            a: float = 0.75 * a + 0.25 * anew

            aprime: float = self.calc_azimuthal_induction(ftan, a, r_R)

            if self.tip_corr:
                aprime: float = aprime / prandtl

            iterations += 1

            if (np.abs(a - anew) < self.iter_error) or (iterations == self.n_iter):
                if self.tip_corr:
                    return [
                        a,
                        aprime,
                        r_R,
                        fnorm,
                        ftan,
                        gamma,
                        alpha,
                        phi,
                        cl,
                        CT,
                        Cn,
                        Ct,
                        CQ,
                        Cp,
                        load_axial,
                        area,
                        prandtl,
                        prandtltip,
                        prandtlroot,
                    ]
                    
                else:
                    return [
                        a,
                        aprime,
                        r_R,
                        fnorm,
                        ftan,
                        gamma,
                        alpha,
                        phi,
                        cl,
                        CT,
                        Cn,
                        Ct,
                        CQ,
                        Cp,
                        load_axial,
                        area,
                    ]

    def simulate(self) -> np.ndarray:
        for i in range(len(self.turbine.r_Rs) - 1):
            self.results[i, :] = self._solve_streamtube(
                self.turbine.r_Rs[i], self.turbine.r_Rs[i + 1]
            )
        return self.results

    def calc_perf(self) -> Tuple[float, float]:
        dr: np.ndarray = (
            self.turbine.r_Rs[1:] - self.turbine.r_Rs[:-1]
        ) * self.turbine.radius
        CT: float = np.sum(
            dr
            * self.results[:, 3]
            * self.turbine.B
            / (0.5 * self.uinf**2 * np.pi * self.turbine.radius**2)
        )
        CP: float = np.sum(
            dr
            * self.results[:, 4]
            * self.results[:, 2]
            * self.turbine.B
            * self.turbine.radius
            * self.omega
            / (0.5 * self.uinf**3 * np.pi * self.turbine.radius**2)
        )
        return CT, CP

    def calc_thrust_torque(self) -> Tuple[float, float]:
        thrust: float = np.sum(
            self.results[:, 3] * self.turbine.dr * self.turbine.radius * self.turbine.B
        )
        torque: float = np.sum(
            self.results[:, 4]
            * self.results[:, 2]
            * self.turbine.dr
            * self.turbine.radius**2
            * self.turbine.B
        )

        return thrust, torque

    def plot_induction_factors(self, name="", save=False, size=8) -> None:
        fig = plt.figure(figsize=(size, size))
        plt.plot(self.results[:, 2], self.results[:, 0], "r-", label="a")
        plt.plot(self.results[:, 2], self.results[:, 1], "g--", label="a'")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel("a, a'")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_flow_angles(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 6], "r-", label=r"$\alpha$")
        plt.plot(self.results[:, 2], self.results[:, 7], "g--", label=r"$\phi$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$\alpha$, $\phi$ [deg]")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_normal_azimuthal_loading_coef(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 10], "r-", label=r"$C_n$")
        plt.plot(self.results[:, 2], self.results[:, 11], "g--", label=r"$C_t$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$C_{n}$, $C_{t}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_thrust_torque_loading_coef(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 10], "r-", label=r"$C_n$")
        plt.plot(self.results[:, 2], self.results[:, 11], "g--", label=r"$C_t$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$C_{n}$, $C_{t}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_normal_azimuthal_loading_coef(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 10], "r-", label=r"$C_n$")
        plt.plot(self.results[:, 2], self.results[:, 11], "g--", label=r"$C_t$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$C_{n}$, $C_{t}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_normal_azimuthal_loading_coef(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 10], "r-", label=r"$C_n$")
        plt.plot(self.results[:, 2], self.results[:, 11], "g--", label=r"$C_t$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$C_{n}$, $C_{t}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_normal_azimuthal_loading_coef(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 9], "r-", label=r"$C_T$")
        plt.plot(self.results[:, 2], self.results[:, 12], "g--", label=r"$C_Q$")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$C_{T}$, $C_{Q}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_norm_tan_loading(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 3], "r-", label=r"Axial")
        plt.plot(self.results[:, 2], self.results[:, 4], "g--", label=r"Azimuthal")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$f_{axial}$, $f_{azim}$")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_prandtl_corr(self, name="", save=False) -> None:
        fig = plt.figure(figsize=(8, 8))
        plt.plot(self.results[:, 2], self.results[:, 16], "r-", label="Tip corection")
        plt.plot(self.results[:, 2], self.results[:, 18], "g--", label="Root corection")
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel("Correction factor [-]")
        plt.legend()
        if save:
            plt.savefig(name)
        plt.show()

    def plot_stag_press(self, name="", save=False) -> None:
        uaxial_rotor: np.ndarray = self.uinf * (1 - self.results[:, 0])
        utan_rotor: np.ndarray = (
            (1 + self.results[:, 1])
            * self.omega
            * self.results[:, 2]
            * self.turbine.radius
        )
        wrotor2: np.ndarray = uaxial_rotor**2 + utan_rotor**2

        uaxial_downwind: np.ndarray = self.uinf * (1 - 2 * self.results[:, 0])
        utan_downwind: np.ndarray = (
            (1 + 2 * self.results[:, 1])
            * self.omega
            * self.results[:, 2]
            * self.turbine.radius
        )
        wdownwind2: np.ndarray = uaxial_downwind**2 + utan_downwind**2

        p2: np.ndarray = self.pinf + 0.5 * self.rho * (
            self.uinf**2 - uaxial_rotor**2
        )
        delta_p: np.ndarray = 0.5 * self.rho * (self.uinf**2 - uaxial_downwind**2)

        p_stag_upwind: np.ndarray = self.pinf + 0.5 * self.rho * self.uinf**2 * np.ones(
            shape=self.results[:, 2].shape
        )
        p_stag_rotor_upwind: np.ndarray = p2 + 0.5 * self.rho * uaxial_rotor**2
        # p_stag_rotor_downwind: np.ndarray = (self.pinf - 0.5*self.results[:, 13]/self.results[:, 14]) + 0.5*self.rho*u_rotor_downwind**2
        p_stag_rotor_downwind: np.ndarray = (
            p2 - delta_p
        ) + 0.5 * self.rho * uaxial_rotor**2
        p_stag_downwind: np.ndarray = self.pinf + 0.5 * self.rho * uaxial_downwind**2

        plt.plot(
            self.results[:, 2],
            p_stag_upwind / p_stag_upwind,
            "r-",
            label="Infinity Upwind",
        )
        plt.plot(
            self.results[:, 2],
            p_stag_rotor_upwind / p_stag_upwind,
            "g--",
            label="Rotor Upwind",
        )
        plt.plot(
            self.results[:, 2],
            p_stag_rotor_downwind / p_stag_upwind,
            "b-.",
            label="Rotor Downwind",
        )
        plt.plot(
            self.results[:, 2],
            p_stag_downwind / p_stag_upwind,
            "k.",
            label="Infinity Downwind",
        )
        plt.ticklabel_format(useOffset=False)
        plt.grid(True)
        plt.xlabel("r/R")
        plt.ylabel(r"$p_{0}$ [Pa]")
        plt.legend()
        if save:
            plt.savefig()
        plt.show()


if __name__ == "__main__":
    # n_ann: int = 80
    n_arr: np.ndarray = np.arange(10, 101, 1) 
    TipLocation_R = 1
    RootLocation_R = 0.2
    
    pitch = 2

    def chord_dist(x):
        return 3 * (1 - x) + 1

    def twist_dist(x):
        return -14 * (1 - x)

    tsr_list: np.ndarray = np.array([8])

    # y_dir: dict = {"tsr6": [], "tsr8": [], "tsr10": []}
    # r_R = 0
    
    thrust_const: np.ndarray = np.zeros(shape=n_arr.shape)
    thrust_cos: np.ndarray = np.zeros(shape=n_arr.shape)
    
    spacing_array: list = ["constant", "cosine"] 
    
    airfoil = Airfoil("DU95W180.dat")
    blade = Blade(airfoil, twist_dist, chord_dist)

    for i in tqdm(range(len(tsr_list))):
        for j in tqdm(range(len(spacing_array)), desc="LOOPING OVER SPACING METHODS"):
            for k in tqdm(range(len(n_arr)), desc="LOOPING OVER ANNULI"):
                Uinf = 10
                TSR = 8
                Radius = 50
                Nblades = 3
                
                spacing: str = spacing_array[j]
                N: int = n_arr[k]

                turbine = Turbine(
                    blade,
                    Nblades,
                    pitch,
                    Radius,
                    TipLocation_R,
                    RootLocation_R,
                    N,
                    spacing,
                )
                sim = BemSimulation(turbine, Uinf, TSR, 1, 101325, True, True)
                
                results: np.ndarray = sim.simulate()
                
                if spacing == "constant":
                    thrust_const[k] = sim.calc_perf()[0]
                elif spacing == "cosine":
                    thrust_cos[k] = sim.calc_perf()[0]
                
                # sim_uncorr = BemSimulation(turbine, Uinf, TSR, 1, 101325, False, True)

                # results: np.ndarray = sim.simulate()
                # r_R = results[:, 2]
                # y_dir[f"tsr{tsr_list[i]}"] = results[:, 13]
                # results_uncorr: np.ndarray = sim_uncorr.simulate()

                # sim.plot_flow_angles()
                # sim.plot_induction_factors()
                # sim.plot_prandtl_corr()
                # sim.plot_normal_azimuthal_loading_coef()
                # sim.plot_thrust_torque_loading_coef()
                # sim.plot_norm_tan_loading()
                # sim.plot_stag_press()
                
        plt.plot(n_arr, thrust_const, "r-", label = "Constant")       
        plt.plot(n_arr, thrust_cos, "g--", label = "Cosine")
        plt.grid(visible=True, which="both")
        plt.legend()
        plt.show()

        # plt.figure(figsize=(8, 8))
        # plt.plot(r_R, y_dir["tsr6"], "r-", label="TSR = 6")
        # plt.plot(r_R, y_dir["tsr8"], "g--", label="TSR = 8")
        # plt.plot(r_R, y_dir["tsr10"], "b-.", label="TSR = 10")
        # plt.xlabel("r/R")
        # plt.ylabel(r"$C_{p}$")
        # plt.grid(True)
        # plt.legend()
        # plt.show()
