Расчет удельной поверхностной энергии.
# -*- coding: utf-8 -*-

from scipy import integrate
import numpy as np
import warnings

warnings.filterwarnings('ignore')
#----------------------------VALUES-----------------------------------------------
HaEnergy = 4.39 * 10**(-18)
HaMass = 9.1 * 10**(-31)
bohrRadii = 5.291772 * 10 ** (-11)
radiiNanoparticle = 10 * 10 ** (-9)
R = radiiNanoparticle/ bohrRadii #radii for estimation
dR = 10 ** (-4) #step for derivation
Rplus = R + dR #upper layer
Rminus = R - dR #lower layer
nAluminum = 26.9 * 10 ** (-3)
beta = 0.01
lamda = 0.0067
kf = (2 * np.pi **2 * nAluminum) ** (1/3)
N = round(4 / 3 * np.pi * R**3 * nAluminum) #electron numbers
S = 4 * np.pi * R**2 #surface
ckin = 3 / 10 *(3 * np.pi**2)**(2/3)
V = 4 / 3 * np.pi * R**3
ro = 2700 * 9.31 * 10**(-31) * (bohrRadii) ** (-3)
m = V * ro
p = 1 / (2 - beta)


#--------density distribution function for electrons in jellium model-----------

def nInUnnormalized(r):
    #the function includes r**2 for integration in spherical coordinates
    return r ** 2 * (1 - p * np.exp(-lamda * (R - r)) +
   p * (np.sin(beta * (R - r)) / (R - r) ))
def nOutUnnormalized(r):
    return r ** 2 * p * np.exp(-lamda * (r - R))
a = R ** 3 * nAluminum * (3 * (integrate.quad(nInUnnormalized, 0, R)[0]
+ integrate.quad(nOutUnnormalized, R, np.inf)[0])) ** (-1.0)


#-------------------energy for non-homogenius-----------------------------------
print('-------------------ENERGY_VALUES-----------------------')
#integrand for kinetic energy
def kineticEnergyIn(r):
    return ckin * 4 * np.pi * r**2 * (a * (1 - p * np.exp(-lamda * (R - r)) + p
   * (np.sin(beta * (R - r))) / (R - r) ))**(5.0/3)
def kineticEnergyOut(r):
    return ckin * 4 * np.pi * r**2 * (a * p * np.exp(-lamda * (r - R)))**(5.0/3)

KineticEnergy = integrate.quad(kineticEnergyIn, 0, R)[0] + \
integrate.quad(kineticEnergyOut, R, np.inf)[0]

epsKineticEnergy = integrate.quad(kineticEnergyIn, 0, R)[1] + \
integrate.quad(kineticEnergyOut, R, np.inf)[1]
print ('Kinetic energy = ', KineticEnergy,'eps = ', epsKineticEnergy, '\n')

#Energy for Coulomb interaction between electrons

def coulombEnergyIn(r):
    i1 = 1 - p * np.exp(-lamda * (R - r)) + p * (np.sin(beta * (R - r)))

    i2 = r**3 / 3 - p * np.exp(-lamda * R) * ((np.exp(lamda * r) * \
    (lamda * r * (lamda * r - 2) + 2) - 2) / lamda**3)

    i3 = p * (-(np.sin(beta * (R - r)) + beta * (R + r) * np.cos(beta * (R - r)))/ beta**2 + \
    (np.sin(beta * R) + beta * R * np.cos(beta * R)) / beta**2  + \
    R**2 * (np.pi/2 - np.cos(beta*(r - R)) / (r-R) - np.sin(beta*(r - R))/(r - R)**2) + \
    R**2 * (np.pi/2 - np.cos(beta*(R)) / (R) - np.sin(beta*(R))/(R)**2))

    i4 = p * (-(np.sin(2 * kf * (R - r)) + 2 * kf * (R + r) * np.cos(2 * kf * (R - r)))/ (2 * kf)**2 + \
    (np.sin(2 * kf * R) + 2 * kf * R * np.cos(2 * kf * R)) / (2 * kf)**2  + \
    R**2 * (np.pi/2 - np.cos(2 * kf * (r - R)) / (r - R) - np.sin(2 * kf * (r - R)) / (r - R)**2) + \
    R**2 * (np.pi/2 - np.cos(2 * kf * (R)) / (R) - np.sin(2 * kf * (R)) / (R)**2))

    i5 = p * (-(np.sin(beta * (R - r)) + beta * (R + r) * np.cos(beta * (R - r)))/ beta**2 + \
    (np.sin(beta * R) + beta * R * np.cos(beta * R)) / beta**2  + \
    R**2 * (beta * (R - r) - (beta * (R - r))**3 / 18 + (beta * (R - r))**5 / 600 - (beta * (R - r))**7 / 35280) + \
    R**2 * (beta * (R) - (beta * (R))**3 / 18 + (beta * (R))**5 / 600 - (beta * (R))**7 / 35280))

    i6 = p * (-(np.sin(2 * kf * (R - r)) + 2 * kf * (R + r) * np.cos(2 * kf * (R - r)))/ (2 * kf)**2 + \
    (np.sin(2 * kf * R) + 2 * kf * R * np.cos(2 * kf * R)) / (2 * kf)**2  + \
    R**2 * (2 * kf * (R - r) - (2 * kf * (R - r))**3 / 18 + (2 * kf * (R - r))**5 / 600 - (2 * kf * (R - r))**7 / 35280) + \
    R**2 * (2 * kf * (R) - (2 * kf * (R))**3 / 18 + (2 * kf * (R))**5 / 600 - (2 * kf * (R))**7 / 35280))

    return 4 * np.pi * r * a**2 * i1 * (i2 + i5)
CoulombEnergyIn = integrate.quad(coulombEnergyIn, 0 , R)
print ('CoulombIn = ',CoulombEnergyIn[0], 'eps = ', CoulombEnergyIn[1])

def coulombEnergyOut(r):
    return 4 * np.pi * r *  a * p * np.exp(-lamda * (r - R)) * a * p * (-(lamda * r * (lamda * r + 2) + \
        2) * np.exp(- lamda * (r - R)) + lamda * R * (lamda * R + 2) + 2) / lamda**3
CoulombEnergyOut = integrate.quad(coulombEnergyOut, R , float("inf"))
print ('CoulombOut = ',CoulombEnergyOut[0], 'eps = ', CoulombEnergyOut[1], '\n')


# ------------------------------GGA--------------------------------------------
def ggaIn(r):
    dn_in = a * (-0.5 * lamda * np.exp(-lamda * (R-r)) - p * np.cos(beta * (R-r)) * beta / (R-r) + p/(R-r)**2 * np.sin(beta * (R-r)))
    n_in = a * (1 - p * np.exp(-lamda * (R - r)) +
   p * (np.sin(beta * (R - r)) / (R - r) ))
    return 4*np.pi * r**2 * 1/(72*n_in) * (dn_in)**2
def ggaOut(r):
    dn_out = -p * lamda * np.exp(-lamda * (r - R))
    n_out = p * np.exp(-lamda * (r - R))
    return 4*np.pi * r**2 * 1/(72*n_out) * (dn_out)**2

GGAIn = integrate.quad(ggaIn, 0, R)
print('GGAIn = ', GGAIn[0], 'eps = ', GGAIn[1], '\n')
GGAOut = integrate.quad(ggaOut, R, float("inf"))


# --------electrostatic energy for ion-electron interaction-------------------
def coulombIonIn(r):
    return -8/3 * np.pi**2 * r**4 * (a)**2 * (1 - p * np.exp(-lamda * (R - r)) +
                                     p * (np.sin(beta * (R - r)) / (R - r)))
CoulombIonIn = integrate.quad(coulombIonIn, 0, R)
print('Ion = ', CoulombIonIn[0], 'eps = ', CoulombIonIn[1], '\n')


# -------------------Exchange-correlation functionals--------------------------
# Gunnarson correlation function
def gunnarsonIn(r):
    rs2_in = (3 / (4 * np.pi * (a * (1 - p * np.exp(-lamda * (R - r)) + \
                        p * (np.sin(beta * (R - r)) / (R - r)))))) ** (1.0/3)
    return 4 * np.pi * r**2 * (-1.222 / rs2_in - 0.066 * np.log(1 + 11.4 / rs2_in))
def gunnarsonOut(r):
    rs2_ex =  (3 / (4 * np.pi * a * p * (np.exp(-lamda * (r - R))))) ** (1.0/3)
    return 4 * np.pi * r ** 2 * (-1.222 / rs2_ex - 0.066 * np.log(1 + 11.4 / rs2_ex))

# ---------------------------------LDA-----------------------------------------
def ldaIn(r):
    return 4 * np.pi * r**2 * (-3/4 * (3/np.pi)**(1/3) * (a * (1-p*np.exp(-lamda*(R-r))+\
    p*(np.sin(beta*(R-r))/(R-r))))**(4/3) - 0.056 * (a * (1-p*np.exp(-lamda*(R-r))+\
    p*(np.sin(beta*(R-r))/(R-r))))**(4/3) / (0.079 + (a * (1-p*np.exp(-lamda*(R-r))+\
    p*(np.sin(beta*(R-r))/(R-r))))**(1/3)))
def ldaOut(r):
    return 4 * np.pi * r**2 * (-3/4 * (3/np.pi)**(1/3) * (a*p*np.exp(-lamda*(r-R)))**(4/3) - 0.056 * (a*p*np.exp(-lamda*(r-R)))**(4/3)\
                               / (0.079 + (a*p*np.exp(-lamda*(r-R)))**(1/3)))



#Beckke 88
def becke(r):
    Klda = 3 * (3/(4*np.pi))**(1/3)
    e = 0.0042
    modn = abs(a * (-0.5 * lamda * np.exp(-lamda * (R-r)) -
 p * np.cos(beta * (R-r)) * beta / (R-r) + p/(R-r)**2 * np.sin(beta * (R-r))))
    n_in = a * (1 - p * np.exp(-lamda * (R - r)) +
                p * (np.sin(beta * (R - r)) / (R - r) ))
    x = modn / n_in**(4.0/3)
    elda = (2 * e * x**2) / (1 + 6 * e *x * np.sinh(x)**(-1))
    return -0.5 * n_in**(4.0/3) * (Klda + elda)


def lda_in(r):
    return -3/4 * (3/np.pi)**(1/3) * (a * (1- p*np.exp(-lamda*(R-r))+\
    p*(np.sin(beta*(R-r))/(R-r))))**(4/3) * 4 * np.pi * r**2
def lda_ex(r):
    return -3 / 4 * (3 / np.pi) ** (1 / 3) * 4 * np.pi * r**2 * p * (np.exp(-lamda*(r-R))) ** (4 / 3)
LDApin = integrate.quad(lda_in, 0, Rplus)[0]
LDAmin = integrate.quad(lda_in, 0, Rminus)[0]
LDApex = integrate.quad(lda_ex, Rplus, np.inf)[0]
LDAmex = integrate.quad(lda_ex, Rminus, np.inf)[0]


#energy for homogenius
def t_kinetic_homos(r):
    return ckin * 4 * np.pi * r**2 * (a * (1 + np.exp(- 2 * (R - r)))**(-1.0)) ** (5.0/3.0)
def w_ee_homos(r):
    return 4 * np.pi * r * a * a * r**3 / 3
def v_ie_homos(r):
    return -8/3 * np.pi**2 * r**4 * (a * (1 + np.exp(- 2 * (R - r)))**(-1.0)) ** (2.0)
Tkin_homos = integrate.quad(t_kinetic_homos, 0, R)[0]
Wee_homos = integrate.quad(w_ee_homos, 0, R)[0]
Vee_homos = integrate.quad(v_ie_homos, 0, R)[0]
E_homos = Tkin_homos + Wee_homos + Vee_homos
e_homos = t_kinetic_homos(R) + w_ee_homos(R) + v_ie_homos(R)


def t_kinetic_homosr(r):
    return ckin * 4 * np.pi  * (a * (1 + np.exp(- 2 * (R - r)))**(-1.0)) ** (5.0/3.0)
def w_ee_homosr(r):
    return 4/3 * np.pi**2 * r**2 * (a * (1 + np.exp(- 2 * (R - r)))**(-1.0)) ** (2.0)
def v_ie_homosr(r):
    return -8/3 * np.pi**2 * r**2 * (a * (1 + np.exp(- 2 * (R - r)))**(-1.0)) ** (2.0)
e_homosr = R/2*(t_kinetic_homosr(R) + w_ee_homosr(R) + v_ie_homosr(R))


# ---------------------------calculating energies------------------------------
# Kinetic energy
Tkin = integrate.quad(kineticEnergyIn, 0, R)[0] + \
        integrate.quad(kineticEnergyOut, R, float('inf'))[0]
Tkinp = integrate.quad(kineticEnergyIn, 0, Rplus)[0] + \
        integrate.quad(kineticEnergyOut, Rplus, np.inf)[0]
Tkinm = integrate.quad(kineticEnergyIn, 0, Rminus)[0] + \
        integrate.quad(kineticEnergyOut, Rminus, np.inf)[0]

Wee = integrate.quad(coulombEnergyIn, 0, R )[0] + integrate.quad(coulombEnergyOut, R, np.inf )[0]
Weep = integrate.quad(coulombEnergyIn, 0, Rplus )[0] + integrate.quad(coulombEnergyOut, Rplus, np.inf )[0]
Weem = integrate.quad(coulombEnergyIn, 0, Rminus )[0] + integrate.quad(coulombEnergyOut, Rminus, np.inf )[0]

Vie = integrate.quad(coulombIonIn, 0, R )[0]
Viep = integrate.quad(coulombIonIn, 0, Rplus )[0]
Viem = integrate.quad(coulombIonIn, 0, Rminus )[0]

gunnarson = integrate.quad(gunnarsonIn, 0, R )[0] +  integrate.quad(gunnarsonOut, R , np.inf )[0]
gunnarsonp = integrate.quad(gunnarsonIn, 0, Rplus )[0] +  integrate.quad(gunnarsonOut, Rplus, np.inf )[0]
gunnarsonm = integrate.quad(gunnarsonIn, 0, Rminus )[0] +  integrate.quad(gunnarsonOut, Rminus, np.inf )[0]

LDA = integrate.quad(ldaIn, 0, R )[0] +  integrate.quad(ldaOut, R, np.inf )[0]
LDAp = integrate.quad(ldaIn, 0, Rplus )[0] +  integrate.quad(ldaOut, Rplus, np.inf )[0]
LDAm = integrate.quad(ldaIn, 0, Rminus )[0] +  integrate.quad(ldaOut, Rminus, np.inf )[0]

GGA = integrate.quad(ggaIn, 0, R)[0]
GGAp = integrate.quad(ggaIn, 0, Rplus)[0]
GGAm = integrate.quad(ggaIn, 0, Rminus)[0]

Becke = integrate.quad(becke, 0, R)[0]
Beckep = integrate.quad(becke, 0, Rplus)[0]
Beckem = integrate.quad(becke, 0, Rminus)[0]



# Full energies
E1 = Tkin + 0.5*Wee + Vie + gunnarson

E2 = Tkin + 0.5*Wee + Vie + LDA

E3 = Tkin + 0.5*Wee + Vie + GGA

E4 = Tkin + 0.5*Wee + Vie + Becke


# ------formule 2 -----------------------------
Ep1 = Tkinp + 0.5*Weep + Viep + gunnarsonp
Em1 = Tkinm + 0.5*Weem + Viem + gunnarsonm

Ep2 = Tkinp + 0.5*Weep + Viep + LDAp
Em2 = Tkinm + 0.5*Weem + Viem + LDAm

Epgga = Tkinp + 0.5*Weep + Viep + GGAp
Emgga = Tkinm + 0.5*Weem + Viem + GGAm

Epbecke = Tkinp + 0.5*Weep + Viep + Beckep
Embecke = Tkinm + 0.5*Weem + Viem + Beckem

#Finding surface energy
surfaceEnergyGunnarson1 = (-E_homos + E1) / S
surfaceEnergyLDA1 = (-E_homos + E2) / S
surfaceEnergyGGA1 = (-E_homos + E3) / S
surfaceEnergyBecke1 = (-E_homos + E4) / S

surfaceEnergyGunnarson2 = (Ep1 - Em1) / (8 * np.pi * R * dR) - R/2 * e_homosr
surfaceEnergyLDA2 = (Ep2 - Em2) / (8 * np.pi * R * dR) - R/2 * e_homosr
surfaceEnergyGGA2 = (Epgga - Emgga) / (8 * np.pi * R * dR) - R/2 * e_homosr
surfaceEnergyBecke2 = (Epbecke - Embecke) / (8 * np.pi * R * dR) - R/2 * e_homosr

dE1 = (E1 - E_homos) / m * HaEnergy/HaMass
dE2 = ((Ep1 - Em1)) / m * HaEnergy/HaMass
dE3 = (Ep2 - Em2) / m * HaEnergy/HaMass
print('---------------------RESULTS-----------------------')
print ('Energy for uniform gas = ',E_homos)
print ('Normalization coefficient = ',a, '\n')
print ('Energy per mass = ', (Ep2 - Em2), dE1, dE2, dE3, '\n')
print('---------------------SURFACE ENERGY------------------')
print('Surface energy Gunnarson formula 1 = ', surfaceEnergyGunnarson1)
print('Surface energy LDA formula 1 = ', surfaceEnergyLDA1)
print('Surface energy GGA formula 1 = ', surfaceEnergyGGA1)
print('Surface energy Becke formula 1 = ', surfaceEnergyBecke1, '\n')
print('Surface energy Gunnarson formula 2 = ', surfaceEnergyGunnarson2)
print('Surface energy LDA formula 2 = ', surfaceEnergyLDA2)
print('Surface energy GGA formula 2 = ', surfaceEnergyGGA2)
print('Surface energy Becke formula 2 = ', surfaceEnergyBecke2)
