'''
Compute fisher matrix (bispectrum) of Atlas probe.
'''

import Bfisher_utils as BF

for z in redshifts:
    FM = BF.FisherB(parc, Vs, navg, kmax, fPk)
    
