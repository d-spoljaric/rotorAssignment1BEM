from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable


class Airfoil:
    def __init__(self, polar_file: str):
        self.filename = polar_file
        self.polar_data = pd.read_csv(f"{polar_file}", header=0, names=["alpha", "cl", "cd", "cm"], sep="\t")

    def get_polar_data(self):
        return self.polar_data

    def plot_polar_data(self) -> None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        fig.tight_layout()
        ax1.plot(self.polar_data["alpha"], self.polar_data["cl"], linestyle="-", color="k", label=r"$C_l$")
        ax1.plot(self.polar_data["alpha"], self.polar_data["cm"], linestyle="--", color="r", label=r"$C_m$")
        ax1.set(xlabel=r"$\alpha$", ylabel=r"$C_l$, $C_m$")
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.polar_data["cd"], self.polar_data["cl"], linestyle="-", color="k")
        ax2.set(xlabel=r"$C_d$", ylabel=r"$C_l$")
        ax2.grid(True)
        plt.show()

    def calc_cl(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return np.interp(alpha, self.polar_data["alpha"], self.polar_data["cl"])

    def calc_cd(self, alpha: float | np.ndarray) -> float | np.ndarray:
        return np.interp(alpha, self.polar_data["alpha"], self.polar_data["cd"])


class Blade:
    def __init__(self, blade_airfoil: Airfoil, twist: Callable, chord: Callable):
        self.blade_airfoil = blade_airfoil
        self.twist = twist
        self.chord = chord


# class turbine():
#     def __init__(self, ):

# class bem_simulation():
#     def __init__(self):


if __name__ == "__main__":
    airfoil = Airfoil("DU95W180.dat")
    airfoilpolar = airfoil.plot_polar_data()
