'''
This code will compute Bispectrum Fisher matrix predictions for ATLAS project
https://arxiv.org/pdf/1802.01539.pdf
'''

#Load and interpolate Pk Data
PP = np.loadtxt("test_matterpower.dat")
K = PP[:,0]
P = PP[:,1]*0.8**2
fPk = interpolate.interp1d(K, P, kind='cubic')

#k, mu, and theta ranges
kmax = 0.2
mumax = 1.0
thmax = 2*np.pi

#Power Spectrum Interpolation 
def Piso(k):
    if k < K[0] or k > K[-1]:
        return 0
    pow = fPk(k)
    return pow

def Pow(k,mu):
    if k < K[0]:
        return 0
    pow = (b1+f*mu**2)**2*Piso(k)
    return pow

#Bispectrum Function
#LS. Let's make it a function of eta_i = k_i**2/2,mu1,phi12
def Bisp(eta1,eta2,eta3,mu1,phi12):
    # Back to k
    k1 = sqrt(2*eta1)
    k2 = sqrt(2*eta2)
    k3 = sqrt(2*eta3)
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

    Bi = 2*Z2k12*Z1k1*Z1k2*Piso(k1)*Piso(k2)
    Bi += 2*Z2k23*Z1k2*Z1k3*Piso(k2)*Piso(k3)
    Bi += 2*Z2k31*Z1k3*Z1k1*Piso(k3)*Piso(k1)

    return Bi

#Bispectrum derivatives
def dBdb1(k1, mu1, k2, mu2, theta):
    global b1
    eps = 1e-6
    B0 = Bisp(k1, mu1, k2, mu2, theta)
    b1 += eps
    B1 = Bisp(k1, mu1, k2, mu2, theta)
    b1 -= eps
    return (B1 - B0)/eps

def dBdb2(k1, mu1, k2, mu2, theta):
    global b2
    eps = 1e-6
    B0 = Bisp(k1, mu1, k2, mu2, theta)
    b2 += eps
    B1 = Bisp(k1, mu1, k2, mu2, theta)
    b2 -= eps
    return (B1 - B0)/eps

def dBdf(k1, mu1, k2, mu2, theta):
    global f
    eps = 1e-6
    B0 = Bisp(k1, mu1, k2, mu2, theta)
    f += eps
    B1 = Bisp(k1, mu1, k2, mu2, theta)
    f -= eps
    return (B1 - B0)/eps

def dBdapar(k1, mu1, k2, mu2, theta):
    global apar
    eps = 1e-6
    B0 = Bisp(k1, mu1, k2, mu2, theta)
    apar += eps
    B1 = Bisp(k1, mu1, k2, mu2, theta)
    apar -= eps
    return (B1 - B0)/eps

#BiSpectrum Covariance
def CovB(eta1,eta2,eta3,mu1,phi12):
    k1 = sqrt(2*eta1)
    k2 = sqrt(2*eta2)
    k3 = sqrt(2*eta3)
    if (k1 < k2):
        return 10000000000
    if (k3 < k1 - k2 or k3 > k1 + k2):
        return 10000000000

    mu12 = (k3**2 - k1**2 - k2**2)/2/k1/k2
    mu2 = mu1*mu12 - sqrt(1 - mu1**2)*sqrt(1 - mu12**2)*cos(phi12)
    mu3 = -(mu1*k1 + mu2*k2)/k3
    
    C = (Pow(k1,mu1)+1/navg)*(Pow(k2,mu2)+1/navg)*(Pow(k3,mu3)+1/navg)
    return C

# Compute Fisher Matrix of Bkk
def FisherB(b1,b2,f,apar,aper):
#Fisher Matrix
    FM = np.array(np.zeros((5,5)))

    Vol = Vs/(2*np.pi)**5 
#Monte Carlo integration in 5D
    etamax = kmax**2/2
    MCvol = etamax**3*mumax*thmax
#Number of Monte Carlo points
    NMC = 10000
    RR = random.rand(NMC,5)
    for i in range(NMC):
        eta1 = etamax*RR[i,0]
        eta2 = etamax*RR[i,1]
        eta3 = etamax*RR[i,2]
        mu1 = 2*mumax*RR[i,3] - 1
        phi12 = thmax*RR[i,4]
        CB = CovB(eta1,eta2,eta3,mu1,phi12)
        db1 = dBdb1(eta1,eta2,eta3,mu1,phi12)
        db2 = dBdb2(eta1,eta2,eta3,mu1,phi12)
        df = dBdf(eta1,eta2,eta3,mu1,phi12)
        dapar = dBdapar(eta1,eta2,eta3,mu1,phi12)
        daper = dBdaper(eta1,eta2,eta3,mu1,phi12)
        FM[0,0] += db1**2/CB
        FM[0,1] += db1*db2/CB
        FM[0,2] += db1*df/CB
        FM[0,3] += db1*dapar/CB
        FM[0,4] += db1*daper/CB
        FM[1,1] += db2**2/CB
        FM[1,2] += db2*df/CB
        FM[1,3] += db2*dapar/CB
        FM[1,4] += db2*daper/CB
        FM[2,2] += df**2/CB
        FM[2,3] += df*dapar/CB
        FM[2,4] += df*daper/CB
        FM[3,3] += dapar**2/CB
        FM[3,4] += dapar*daper/CB
        FM[4,4] += daper**2/CB

    FM[1,0] = FM[0,1]
    FM[2,0] = FM[0,2]
    FM[3,0] = FM[0,3]
    FM[4,0] = FM[0,4]
    FM[2,1] = FM[1,2]
    FM[3,1] = FM[1,3]
    FM[4,1] = FM[1,4]
    FM[3,2] = FM[2,3]
    FM[4,2] = FM[2,4]
    FM[4,3] = FM[3,4]

    FM *= Vol*MCvol/NMC/2

    return FM
