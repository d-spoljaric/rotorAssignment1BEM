from typing import Any, Tuple

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
    def __init__(self, blade_airfoil: Airfoil, twist: np.ndarray, chord: np.ndarray):
        self.blade_airfoil = blade_airfoil
        self.twist = twist
        self.chord = chord
    
    def get_chord(self, r_R_range: np.ndarray, r_R: float | np.ndarray) -> float | np.ndarray:
       return np.interp(r_R, r_R_range, self.chord)
   
    def get_twist(self, r_R_range: np.ndarray, r_R: float | np.ndarray) -> float | np.ndarray:
       return np.interp(r_R, r_R_range, self.chord)
   


class Turbine:
    def __init__(self, turbine_blade: Blade, NBlades: int, radius: float, tipradius_R: float, rootradius_R: float):
        self.turbine_blade = turbine_blade
        self.B = NBlades
        self.radius = radius
        self.tipradius_R = tipradius_R
        self.rootradius_R = rootradius_R
        
class BemSimulation: 
    def __init__(self, turbine: Turbine, uinf: float, tsr: float, r_Rs: np.ndarray, n_iterations = 100, iteration_error = 0.00001, glauert_correction = True):
        self.turbine = turbine
        self.blade = self.turbine.turbine_blade
        self.airfoil = self.blade.blade_airfoil
        self.uinf = uinf
        self.tsr = tsr 
        self.omega = uinf*tsr/self.turbine.radius
        self.r_Rs = r_Rs 
        self.glauert = glauert_correction 
        self.n_iter = n_iterations
        self.iter_error = iteration_error
        
    def calc_axial_induction(self, CT: np.ndarray) -> np.ndarray:
        a: np.ndarray = np.zeros(shape = np.shape(CT))
        CT1: float = 1.816
        CT2: float = 2*np.sqrt(CT1) - CT1
        a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1)) 
        a[CT<=CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2]) 

        return a

    def _load_blade_element(self, vnorm: float, vtan: float, r_R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chord: float = self.blade.get_chord(self.r_Rs, r_R)
        vmag2: float = vnorm**2 + vtan**2
        inflowangle: float = np.arctan2(vnorm, vtan)
        alpha: float = self.blade.twist + np.degrees(inflowangle) 
        cl: float = self.airfoil.calc_cl(alpha)
        cd: float = self.airfoil.calc_cd(alpha) 
        lift: float = 0.5*vmag2*cl*chord
        drag: float = 0.5*vmag2*cd*chord
        fnorm: float = lift*np.cos(inflowangle) + drag*np.sin(inflowangle)
        ftan: float = lift*np.sin(inflowangle) - drag*np.cos(inflowangle)
        gamma: float = 0.5*np.sqrt(vmag2)*cl*chord
        
        return fnorm, ftan, gamma
    
    def _prandtl_tip_root_correction(self, r_R: float, ax_ind: float) -> Tuple[float, float, float]:
        temp1 = -self.turbine.B/2*(self.turbine.tipradius_R-r_R)/r_R*np.sqrt( 1+ ((self.tsr*r_R)**2)/((1-ax_ind)**2))
        Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
        Ftip[np.isnan(Ftip)] = 0
        temp1 = self.turbine.B/2*(self.turbine.rootradius_R-r_R)/r_R*np.sqrt( 1+ ((self.tsr*r_R)**2)/((1-ax_ind)**2))
        Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
        Froot[np.isnan(Froot)] = 0
        return Froot*Ftip, Ftip, Froot
    
    def solve_streamtube(self, r1_R: float, r2_R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        area: float = np.pi*((r2_R*self.turbine.radius)**2-(r1_R*self.turbine.radius)**2)
        r_R: float = (r1_R + r2_R)/2
        
        a: float = 0.0
        aline: float = 0.0
        
        prandtl_min = 0.0001
        
        for i in range(self.n_iter):
            urotor: float = self.uinf*(1-a)
            utan: float = (1+aline)*self.omega*r_R*self.turbine.radius
            
            fnorm, ftan, gamma = self._load_blade_element(urotor, utan, r_R) 
            load_3d_axial: float = fnorm*self.turbine.radius*(r2_R-r1_R)*self.turbine.B
            
            CT = load_3d_axial/(0.5*area*self.uinf**2)
            anew = self.calc_axial_induction(CT)
            prandtl, prandtltip, prandtlroot = self._prandtl_tip_root_correction(r_R, anew)
            if prandtl < prandtl_min:
                prandtl = prandtl_min
                
            anew = anew/prandtl
            a = 0.75*a+0.25*anew
            
            aline = ftan*self.turbine.B/(2*np.pi*self.uinf*(1-a)*self.omega*2*(r_R*self.turbine.radius)**2)
            aline = aline/prandtl
            
            if np.abs(a-anew) < self.iter_error:
                break
            
            return [a, aline, r_R, fnorm, ftan, gamma]
            
            
if __name__ == "__main__":
    delta_r_R = 0.01
    TipLocation_R =  1
    RootLocation_R =  0.2
    r_R = np.arange(RootLocation_R, TipLocation_R+delta_r_R/2, delta_r_R)
    
    pitch = 2
    chord_dist = 3*(1-r_R)+1
    twist_dist = -14*(1-r_R)+pitch
    
    Uinf = 1
    TSR = 8
    Radius = 50
    Nblades = 3
     
    airfoil = Airfoil("DU95W180.dat")
    blade = Blade(airfoil, twist_dist, chord_dist)
    turbine = Turbine(blade, Nblades, Radius, TipLocation_R, RootLocation_R)
    sim = BemSimulation(turbine, Uinf, TSR, r_R)
    
    
    results = np.zeros([len(r_R)-1, 6])
    
    for i in range(len(r_R) - 1):
        results[i:, ] = sim.solve_streamtube(r_R[i], r_R[i+1])