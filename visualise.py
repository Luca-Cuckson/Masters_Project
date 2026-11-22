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
bmean, bfwhm = np.loadtxt(bfile2, usecols=(1,2), unpack=True)
bmed = np.nanmedian(bmean)
print('Blue median =', bmed)
print('Blue mean =', bmean)

gres_id, gEres = np.loadtxt(gfile1, usecols=(0,1), unpack=True)
gmean, gfwhm = np.loadtxt(gfile2, usecols=(1,2), unpack=True)
gmed = np.nanmedian(gmean)
print('Green median =', gmed)
print('Green mean =', gmean)

rres_id, rEres = np.loadtxt(rfile1, usecols=(0,1), unpack=True)
rmean, rfwhm = np.loadtxt(rfile2, usecols=(1,2), unpack=True)
rmed = np.nanmedian(rmean)
print('Red median =', rmed)
print('Red mean =', rmean)

#print(len(rEres), len(rmean))
#print(len(rres_id), len(rfwhm))

fig0 = plt.figure()
plt.xlim([0,125])
plt.ylabel('Energy resolution')
plt.xlabel('Detector mean pulse height (phase angle)')
plt.scatter(bmean, bEres, s=3, alpha=0.4, color='b')
plt.scatter(gmean, gEres, s=3, alpha=0.4, color='g')
plt.scatter(rmean, rEres, s=3, alpha=0.4, color='r')


fig1 = plt.figure()
plt.ylabel('Number of resonators')
plt.xlabel('Mean pulse height')
plt.hist(bmean, bins=50, color='b')
plt.axvline(bmed, color='k')

fig2 = plt.figure()
plt.ylabel('Number of resonators')
plt.xlabel('Mean pulse height')
plt.hist(gmean, bins=50, color='g')
plt.axvline(gmed, color='k')

fig3 = plt.figure()
plt.ylabel('Number of resonators')
plt.xlabel('Mean pulse height')
plt.hist(rmean, bins=50, color='r')
plt.axvline(rmed, color='k')

plt.show()