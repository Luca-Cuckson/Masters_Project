import numpy as np
import matplotlib.pyplot as plt

bfile1 = 'energy_resolution_blue.txt'
bfile2 = 'mean_fwhm_blue.txt'

gfile1 = 'energy_resolution_green.txt'
gfile2 = 'mean_fwhm_green.txt'

rfile1 = 'energy_resolution_red.txt'
rfile2 = 'mean_fwhm_red.txt'

bres_id, bEres = np.loadtxt(bfile1, usecols=(0,1), unpack=True)
bmean, bfwhm = np.loadtxt(bfile2, usecols=(1,2), unpack=True)
bmed = np.nanmedian(bmean)
print(bmed)

gres_id, gEres = np.loadtxt(gfile1, usecols=(0,1), unpack=True)
gmean, gfwhm = np.loadtxt(gfile2, usecols=(1,2), unpack=True)
gmed = np.nanmedian(gmean)
print(gmed)

rres_id, rEres = np.loadtxt(rfile1, usecols=(0,1), unpack=True)
rmean, rfwhm = np.loadtxt(rfile2, usecols=(1,2), unpack=True)
rmed = np.nanmedian(rmean)
print(rmed)

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