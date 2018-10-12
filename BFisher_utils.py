'''
This code will compute Bispectrum Fisher matrix predictions for ATLAS project
https://arxiv.org/pdf/1802.01539.pdf
'''

#Load and interpolate Pk Data
def init_Pk(Pk, sigma8):
    PP = np.loadtxt(Pk_file)
    K = PP[:,0]
    P = PP[:,1]*sigma8**2
    fPk = interpolate.interp1d(K, P, kind='cubic', bounds_error = False,
                               fill_value = 0)
    return fPk

#Power Spectrum Interpolation 
def Piso(fPk, k):
    pow = fPk(k)
    return pow

def Pow(fPk, k,mu):
    pow = (b1+f*mu**2)**2*Piso(fPk, k)
    return pow

#Bispectrum Function
# park - shape parameters
# parc - cosmological parameters
def Bisp(park, parc, fPk):

    k1, k2, k3, mu, phi12 = park
    b1, b2, f, apar, aper, sigmav = parc

    if (k1 < k2): 
        return 0
    if (k3 < k1 - k2 or k3 > k1 + k2): 
        return 0

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - sqrt(1 - mu1**2)*sqrt(1 - mu12**2)*cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3

    # rescale for AP
    mu1 = mu1*apar/np.sqrt(aper**2+mu1**2*(apar**2-aper**2))
    mu2 = mu2*apar/np.sqrt(aper**2+mu2**2*(apar**2-aper**2))
    mu3 = mu3*apar/np.sqrt(aper**2+mu3**2*(apar**2-aper**2))
    k1 = k1*np.sqrt(aper**2+mu1**2*(apar**2-aper**2))
    k2 = k2*np.sqrt(aper**2+mu2**2*(apar**2-aper**2))
    k3 = k3*np.sqrt(aper**2+mu3**2*(apar**2-aper**2))

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu31 = -(k1 + k2*mu12)/k3
    mu23 = -(k1*mu12 + k2)/k3

    k12 = sqrt(k1**2 + k2**2 + 2*k1*k2*mu12)
    k23 = sqrt(k2**2 + k3**2 + 2*k2*k3*mu23)
    k31 = sqrt(k3**2 + k1**2 + 2*k3*k1*mu31)

    Z1k1 = b1 + f*mu1**2
    Z1k2 = b1 + f*mu2**2
    Z1k3 = b1 + f*mu3**2

    F12 = 5./7. + mu12/2*(k1/k2 + k2/k1) + 2./7.*mu12**2
    F23 = 5./7. + mu23/2*(k2/k3 + k3/k2) + 2./7.*mu23**2
    F31 = 5./7. + mu31/2*(k3/k1 + k1/k3) + 2./7.*mu31**2

    G12 = 3./7. + mu12/2*(k1/k2 + k2/k1) + 4./7.*mu12**2
    G23 = 3./7. + mu23/2*(k2/k3 + k3/k2) + 4./7.*mu23**2
    G31 = 3./7. + mu31/2*(k3/k1 + k1/k3) + 4./7.*mu31**2

    mu1p2 = (mu1*k1 + mu2*k2)/k12
    mu2p3 = (mu2*k2 + mu3*k3)/k23
    mu3p1 = (mu3*k3 + mu1*k1)/k31

    Z2k12 = b2/2. + b1*F12 + f*mu1p2**2*G12
    Z2k12 += f*mu1p2*k12/2.*(mu1/k1*Z1k2 + mu2/k2*Z1k1)
    Z2k23 = b2/2. + b1*F23 + f*mu2p3**2*G23
    Z2k23 += f*mu2p3*k23/2.*(mu2/k2*Z1k3 + mu3/k3*Z1k2)
    Z2k31 = b2/2. + b1*F31 + f*mu3p1**2*G31
    Z2k31 += f*mu3p1*k31/2.*(mu3/k3*Z1k1 + mu1/k1*Z1k3)

    Bi = 2*Z2k12*Z1k1*Z1k2*Piso(fPk, k1)*Piso(fPk, k2)
    Bi += 2*Z2k23*Z1k2*Z1k3*Piso(fPk, k2)*Piso(fPk, k3)
    Bi += 2*Z2k31*Z1k3*Z1k1*Piso(fPk, k3)*Piso(fPk, k1)

    # Fingers of God
    Bi *= np.exp(- k1**2*mu1**2*sigmav**2/2) 
    Bi *= np.exp(- k2**2*mu2**2*sigmav**2/2) 
    Bi *= np.exp(- k3**2*mu3**2*sigmav**2/2) 

    return Bi

#Bispectrum derivatives
# Return a vector of derivatives for all cosmological parameters
def dBisp(park, parc, fPk):
    Npar = np.size(parc)
    dB = np.zeros(Npar)
    eps = 0.001
    for i in range(Npar):
       parcfin = np.copy(parc)
       parcfin[i] += parcfin[i]*eps
       Bini = Bisp(park, parc, fPk)
       Bfin = Bisp(park, parcfin, fPk)
       dB[i] = (Bfin - Bini)/eps
    return dB

#BiSpectrum Covariance for fixed triangular configuration
def CovB(park, navg, fPk):
    
    bignumber = 10000000000

    k1, k2, k3, mu1, phi12 = park
    if (k1 < k2):
        return bignumber
    if (k3 < k1 - k2 or k3 > k1 + k2):
        return bignumber

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - sqrt(1 - mu1**2)*sqrt(1 - mu12**2)*cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    
    C1 = (Pow(fPk, k1, mu1) + 1/navg)
    C2 = (Pow(fPk, k2, mu2) + 1/navg)
    C3 = (Pow(fPk, k3, mu3) + 1/navg)
    C = C1*C2*C3

    return C

# Compute Fisher Matrix of Bkk
# in a redshift slice of given Vs and constant navg
def FisherB(parc, Vs, navg, kmax, fPk):

    #Number of Monte Carlo points
    NMC = 10000

    Npar = np.size(parc)
    #Fisher Matrix
    FM = np.zeros((Npar,Npar))

    Vol = Vs/(2*np.pi)**5 
    #Monte Carlo integration in 5D
    etamax = kmax**2/2
    MCvol = etamax**3*2*np.pi
    RR = random.rand(NMC,5)
    for i in range(NMC):
        k1 = np.sqrt(2*etamax*RR[i,0])
        k2 = np.sqrt(2*etamax*RR[i,1])
        k3 = np.sqrt(2*etamax*RR[i,2])
        mu1 = 2*RR[i,3] - 1
        phi12 = 2*np.pi*RR[i,4]
        park = (k1, k2, k3, mu1, phi12)
        CB = CovB(park, navg, fPk)
        db = dBisp(park, parc, fPk)
        FM += np.outer(dB,dB)/CB

    FM *= Vol*MCvol/NMC/2

    return FM
