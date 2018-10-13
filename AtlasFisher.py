'''
Compute fisher matrix (bispectrum) of Atlas probe.
'''
import numpy as np

import BFisher_utils as BF

# Total number of Atlas galaxies
Ntotal = 183000000.0
# Maximum wavenumber to go to
kmax = 0.3
# power and amplitude
fPk = BF.init_Pk('test_matterpower.dat')
sigma8 = 0.8
# Cosmological parameters
parcosmo = (-1, 0, 0.3, 0, 0.7)

def volume(zmin,zmax):
    """Volume of a redshift shell in (Mpc/h)^3"""
    dist = lambda z: 1/sqrt(Om*(1+z)**3 + Ol)*clight
    rmax = quad(dist, 0, zmax)[0]
    rmin = quad(dist, 0, zmin)[0]
    return Area*(rmax**3 - rmin**3)*4*pi/3

def H(z,par):
    """Hubble parameter in w0waCDMGR in km/Mpc/s"""
    w0,wa,Om,Ok,h = par
    return 100*h*sqrt(Om*(1+z)**3 + Ok*(1+z)**2 +
          (1-Om-Ok)*(1+z)**(3*(1+w0))*exp(3*wa*z/(1+z)))

def DA(z,par):
    """The angular distance in Mpc"""
    da = quad(lambda x: 1/H(x,par),0,z)[0] da *= clight/(1+z)
    return da

def G(z,par):
    """The growth factor (LCDMGR). Normalized so that G(z=0) = 1"""
    Gz = H(z,par)*quad(lambda x: (1+x)/H(x,par)**3,z,10000)[0]
    G0 = H(0,par)*quad(lambda x: (1+x)/H(x,par)**3,0,10000)[0]
    return Gz/G0

def f(z,par):
    """The growth rate = dlnG/dlna (LCDMGR)"""
    dz = 0.01
    lnG2 = log(G(z+dz,par))
    lnG1 = log(G(z,par))
    return -(lnG2-lnG1)/dz*(1+z)

# Load number densities
zmin, zmax, fraction = np.loadtxt('navg.txt', unpack = True)
Ngal = fraction*Ntotal
zmid = (zmin + zmax)/2.

for i in np.size(zmid):
    Vs = volume(zmin[i], zmax[i])
    parc = (b1[i], b2[i], f(zmid[i], parcosmo), 1, 1, 0)
    FM = BF.FisherB(parc, Vs, navg[i], kmax, fPk)
