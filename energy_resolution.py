import numpy as np
import matplotlib.pyplot as plt
'''
bfile1 = 'first_beginnings/obtained_values/energy_resolution_blue.txt'
bfile2 = 'first_beginnings/obtained_values/mean_fwhm_blue.txt'

gfile1 = 'first_beginnings/obtained_values/energy_resolution_green.txt'
gfile2 = 'first_beginnings/obtained_values/mean_fwhm_green.txt'

rfile1 = 'first_beginnings/obtained_values/energy_resolution_red.txt'
rfile2 = 'first_beginnings/obtained_values/mean_fwhm_red.txt'
'''

bfile1 = 'Energy_Resolution/energy_resolution_520nm.txt'
bfile2 = 'Energy_Resolution/mean_fwhm_520nm.txt'

gfile1 = 'Energy_Resolution/energy_resolution_635nm.txt'
gfile2 = 'Energy_Resolution/mean_fwhm_635nm.txt'

rfile1 = 'Energy_Resolution/energy_resolution_780nm.txt'
rfile2 = 'Energy_Resolution/mean_fwhm_780nm.txt'

bres_id, bEres = np.loadtxt(bfile1, usecols=(0,1), unpack=True)
print('Blue median energy resolution =', np.nanmedian(bEres))

gres_id, gEres = np.loadtxt(gfile1, usecols=(0,1), unpack=True)
print('Green median energy resolution =', np.nanmedian(gEres))

rres_id, rEres = np.loadtxt(rfile1, usecols=(0,1), unpack=True)
print('Red median energy resolution =', np.nanmedian(rEres))