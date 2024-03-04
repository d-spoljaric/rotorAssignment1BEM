from typing import Any, Tuple, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    def __init__(self, blade_airfoil: Airfoil, twist: Callable[[np.ndarray | float], np.ndarray | float], chord: Callable[[np.ndarray | float], np.ndarray | float]):
        self.blade_airfoil = blade_airfoil
        self.twist = twist
        self.chord = chord


class Turbine:
    def __init__(self, turbine_blade: Blade, NBlades: int, pitch: float, radius: float, tipradius_R: float, rootradius_R: float, delta_r_R: float):
        self.turbine_blade = turbine_blade
        self.B = NBlades
        self.pitch = pitch
        self.radius = radius
        self.tipradius_R = tipradius_R
        self.rootradius_R = rootradius_R
        self.dr = delta_r_R
        self.r_Rs = np.arange(RootLocation_R, TipLocation_R+delta_r_R/2, delta_r_R)
            
class BemSimulation: 
    def __init__(self, turbine: Turbine, uinf: float, tsr: float, rho: float, tip_correction = True, glauert_correction = True, n_iterations = 100, iteration_error = 0.00001):
        self.turbine = turbine
        self.blade = self.turbine.turbine_blade
        self.airfoil = self.blade.blade_airfoil
        self.uinf = uinf
        self.tsr = tsr 
        self.omega = uinf*tsr/self.turbine.radius
        self.rho = rho
        self.tip_corr = tip_correction
        self.glauert_corr = glauert_correction
        self.n_iter = n_iterations
        self.iter_error = iteration_error
        self.results: np.ndarray = np.zeros([len(self.turbine.r_Rs)-1, 6])
        
    def calc_axial_induction(self, CT: np.ndarray) -> np.ndarray:
        a: np.ndarray = np.zeros(shape = np.shape(CT))
        if self.glauert_corr:
            CT1: float = 1.816
            CT2: float = 2*np.sqrt(CT1) - CT1
            a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1)) 
            a[CT<=CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
        else:
            a = 0.5*(1-np.sqrt(1-CT))
        return a

    def _load_blade_element(self, vnorm: float | np.ndarray, vtan: float | np.ndarray, r_R: float | np.ndarray) -> Tuple[float, float, float] | Tuple[np.ndarray, np.ndarray, np.ndarray]:
        chord: float = self.blade.chord(r_R)
        vmag2: float = vnorm**2 + vtan**2
        phi : float = np.arctan2(vnorm, vtan)
        alpha: float = self.blade.twist(r_R) + self.turbine.pitch + np.degrees(phi)
        cl: float = self.airfoil.calc_cl(alpha)
        cd: float = self.airfoil.calc_cd(alpha) 
        lift: float = 0.5*self.rho*vmag2*cl*chord
        drag: float = 0.5*self.rho*vmag2*cd*chord
        fnorm: float = lift*np.cos(phi) + drag*np.sin(phi)
        ftan: float = lift*np.sin(phi) - drag*np.cos(phi)
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
    
    def _solve_streamtube(self, r1_R: float, r2_R: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        area: float = np.pi*((r2_R*self.turbine.radius)**2-(r1_R*self.turbine.radius)**2)
        r_R: float = (r1_R + r2_R)/2
        
        a: float = 0.0
        aprime: float = 0.0
        
        prandtl_min: float = 0.0001
        
        solving: bool = True
        
        iterations: int = 0
        while solving:
            urotor: float = self.uinf*(1-a)
            utan: float = (1+aprime)*self.omega*r_R*self.turbine.radius
            fnorm, ftan, gamma = self._load_blade_element(urotor, utan, r_R)
            load_axial: float = fnorm*self.turbine.radius*(r2_R-r1_R)*self.turbine.B
            
            CT: float = load_axial/(0.5*area*self.uinf**2)
            anew: float = self.calc_axial_induction(CT)
            
            if self.tip_corr:
                prandtl, prandtltip, prandtlroot = self._prandtl_tip_root_correction(r_R, anew)
                if prandtl < prandtl_min:
                    prandtl = prandtl_min
                anew: float = anew/prandtl
                
            a: float = 0.75*a+0.25*anew
            
            aprime: float = ftan*self.turbine.B/(2*np.pi*self.uinf*(1-a)*self.omega*2*(r_R*self.turbine.radius)**2)
            
            if self.tip_corr:
                aprime: float = aprime/prandtl
                
            iterations += 1
            
            if (np.abs(a-anew) < self.iter_error) or (iterations==self.n_iter):
                return [a, aprime, r_R, fnorm, ftan, gamma]

    
    def simulate(self) -> np.ndarray:
        for i in range(len(self.turbine.r_Rs)-1):
            self.results[i, :] = self._solve_streamtube(self.turbine.r_Rs[i], self.turbine.r_Rs[i+1])
            
        return self.results
    
    

if __name__ == "__main__":
    delta_r_R = 0.01
    TipLocation_R =  1
    RootLocation_R =  0.2
    
    pitch = 2
    
    def chord_dist(x):
        return 3*(1-x)+1
    
    def twist_dist(x):
        return -14*(1-x)
    
    Uinf = 1
    TSR = 8
    Radius = 50
    Nblades = 3
    
    omega = Uinf*TSR/Radius
     
    airfoil = Airfoil("DU95W180.dat")
    blade = Blade(airfoil, twist_dist, chord_dist)
    turbine = Turbine(blade, Nblades, pitch, Radius, TipLocation_R, RootLocation_R, 0.01)
    sim = BemSimulation(turbine, Uinf, TSR, 1, True, True)
    
    results: np.ndarray = sim.simulate()
   
    # areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2

    # dr = (r_R[1:]-r_R[:-1])*Radius
    # CT = np.sum(dr*results[:,3]*Nblades/(0.5*Uinf**2*np.pi*Radius**2))
    # CP = np.sum(dr*results[:,4]*results[:,2]*Nblades*Radius*omega/(0.5*Uinf**3*np.pi*Radius**2))
    
    # print("CT is ", CT)
    # print("CP is ", CP)
    
    # fig1 = plt.figure(figsize=(12, 6))
    # plt.title('Axial and tangential induction')
    # plt.plot(results[:,2], results[:,0], 'r-', label=r'$a$')
    # plt.plot(results[:,2], results[:,1], 'g--', label=r'$a^,$')
    # plt.grid()
    # plt.xlabel('r/R')
    # plt.legend()
    # plt.show()