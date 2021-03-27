from scipy import integrate
import numpy as np
# Constants
Ha_energy = 4.39 * 10**(-18)
Ha_mass = 9.1 * 10**(-31)
bohr_radii = 5.291772 * 10 ** (-11)
radii_nanoparticle = 10 * 10 ** (-9)
R = radii_nanoparticle / bohr_radii  # radii for estimation
n_aluminum = 26.9 * 10 ** (-3)
a = 0.0269
ckin = 3 / 10 *(3 * np.pi**2)**(2/3)
def integral(R,lamda,beta):
    p = 1 / (2 - beta)
    def Iin(r):
        return ((1 - p * np.exp(-lamda * (R - r)) + p
   * (np.sin(beta * (R - r))) / (R - r) )) * r**2
    def Iex(r):
        return (p * np.exp(-lamda * (r - R))) * r**2
    return integrate.quad(Iin,0,R)[0] + integrate.quad(Iex,R,float("inf"))[0]
# Kinetic energy
def functionalT(R,lamda,beta):
    p = 1 / (2 - beta)
    def Tin(r):
        return ckin * 4 * np.pi * r**2 * ((1 - p * np.exp(-lamda * (R - r)) + p
   * (np.sin(beta * (R - r))) / (R - r) ))**5/3
    def Tex(r):
        return (p * np.exp(-lamda * (r - R)))**(5/3)*r**2
    return 2.87 * (((0.33 * R**3 *n_aluminum)/(integral(R,lamda,beta)))**(5/3) * \
 integrate.quad(Tin,0,R)[0] + ((0.33 * R**3 * n_aluminum)/(integral(R,lamda,beta)))**(5/3) * \
 integrate.quad(Tex,R,float("inf"))[0])

# Coulomb interactions
def functionalU(R,lamda,beta):
    p = 1 / (2 - beta)
    def Uin(r):
        i1 = ((1 - p * np.exp(-lamda * (R - r)) + \
    p * (np.sin(beta * (R - r)))))

        i2 = r**3 / 3 - p * np.exp(-lamda * R) * ((np.exp(lamda * r) * \
    (lamda * r * (lamda * r - 2) + 2) - 2) / lamda**3)

        i5 = p * (-(np.sin(beta * (R - r)) + beta * (R + r) * np.cos(beta * (R - r)))/ beta**2 + \
    (np.sin(beta * R) + beta * R * np.cos(beta * R)) / beta**2  + \
    R**2 * (beta * (R - r) - (beta * (R - r))**3 / 18 + (beta * (R - r))**5 / 600 - (beta * (R - r))**7 / 35280) + \
    R**2 * (beta * (R) - (beta * (R))**3 / 18 + (beta * (R))**5 / 600 - (beta * (R))**7 / 35280))
        return 4 * np.pi * r * (0.5*(0.3333*R**3*n_aluminum)/integral(R,lamda,beta))**2 * i1 * (i2 + i5)
    def Uex(r):
        return 4 * np.pi * r * (0.5*(0.3333*R**3*n_aluminum)/integral(R,lamda,beta)) * \
    p * np.exp(-lamda * (r - R)) * a * p * (-(lamda * r * (lamda * r + 2) + \
        2) * np.exp(- lamda * (r - R)) + lamda * R * (lamda * R + 2) + 2) / lamda**3
    return integrate.quad(Uin,0,R)[0] + integrate.quad(Uex,R,float("inf"))[0]

def functionalV(R,lamda,beta):
    def Vin(r):
        return -8/3 * np.pi**2 * r**4 * ((0.3333*R**3*n_aluminum)/integral(R,lamda,beta))**2
    return integrate.quad(Vin,0,R)[0]
# Exchange-correlation functional
def functionalC(R,lamda,beta):
    p = 1 / (2 - beta)
    def vxc_in_1(r):
        rs2_in = (3 / (4 * np.pi * ((0.33333*R**3*n_aluminum)/integral(R,lamda,beta) * (1 - p * np.exp(-lamda * (R - r)) + p * (np.sin(beta * (R - r)) / (R - r)))))) ** 1/3
        return 4 * np.pi * r**2 * (-1.222 / rs2_in - 0.066 * np.log(1 + 11.4 / rs2_in))
    def vxc_ex_1(r):
        rs2_ex =  (3 / (4 * np.pi * (0.33333*R**3*n_aluminum)/integral(R,lamda,beta) * p * (np.exp(-lamda * (r - R))))) ** 1/3
        return 4 * np.pi * r ** 2 * (-1.222 / rs2_ex - 0.066 * np.log(1 + 11.4 / rs2_ex))
    return integrate.quad(vxc_in_1,0,R-0.1)[0] + integrate.quad(vxc_ex_1,R+0.1,float("inf"))[0]
from scipy.optimize import minimize
lkl1 = lambda pars: (functionalT(R,*pars)+functionalU(R, *pars) - functionalV(R, *pars) + functionalC(R, *pars))
tol = 1e-6
bnds = ((0.001,6.001),(0.0001,0.0099901))
init_guess = (1.187,0.016)
lil_model_init = minimize(lkl1,init_guess,method='TNC',tol=tol,bounds=bnds)
lil_model_init1 = minimize(lkl1,init_guess,method='L-BFGS-B',tol=tol,bounds=bnds)
print(lil_model_init)
print(lil_model_init1)
