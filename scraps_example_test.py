import numpy as np
import scraps as scr
import os
import pprint as pp
import matplotlib.pyplot as plt
import pygtc


# ======================================================================================================= 
# LOAD IN DATA
# ======================================================================================================= 

#Load in a file
dataFile = 'D:/LucaC_L4_Project/Scraps_Example_Data/ExampleData/RES-1_-20_DBM_TEMP_1.226.S2P'

#Use the process_file routine to read the file and create a dict of resonator data
fileDataDict = scr.process_file(dataFile, skiprows=1)

#Look at the contents of the dict:
pp.pprint(fileDataDict)

# ======================================================================================================= 
# MAKE AND FIT RESONATOR OBJECT
# ======================================================================================================= 

#Create a resonator object using the helper tool
resObj1 = scr.makeResFromData(fileDataDict)

#Create a resonator object using the helper tool and also fit the data
#To do this, we pass a function that initializes the parameters for the fit, and also the fit function
resObj2 = scr.makeResFromData(fileDataDict, paramsFn = scr.cmplxIQ_params, fitFn = scr.cmplxIQ_fit)

#Check the temperature and power
print('Temperature = ', resObj1.temp)
print('Power = ', resObj1.pwr)

#Check to see whether a results object exists
print('Do fit results exist for the first object? ', resObj1.hasFit)
print('Do fit results exist for the second object? ', resObj2.hasFit)


#Explicitly call the fitter on the first object.
#Here we will call it, and also override the guess for coupling Q with our own quess
resObj1.load_params(scr.cmplxIQ_params)
resObj1.do_lmfit(scr.cmplxIQ_fit, qc=5000)

#Check to see whether a results object exists again, now they are both True
print('Do fit results exist for the first object? ', resObj1.hasFit)
print('Do fit results exist for the second object? ', resObj2.hasFit)

#Compare the best guess for the resonant frequency (minimum of the curve) to the actual fit
#Because we didn't specify a label for our fit, the results are stored in the lmfit_result
#dict under the 'default' key. If we passed the optional label argument to the do_lmfit
#method, it would store the results under whatever string is assigned to label.
print('Guess = ', resObj2.fmin, ' Hz')
print('Best fit = ', resObj2.lmfit_result['default']['result'].params['f0'].value, ' Hz')
print('Best fit with different qc guess = ',
      resObj1.lmfit_result['default']['result'].params['f0'].value, ' Hz')

#You can see the fit is not terribly sensitive to the guess for qc.

# ======================================================================================================= 
# PLOTTING
# ======================================================================================================= 

#This function takes a list of resonators. It can handle a single one, just need to pass it as a list:
figA = scr.plotResListData([resObj1],
                          plot_types = ['LogMag', 'Phase', 'IQ'], #Make two plots
                          num_cols = 3, #Number of columns
                          fig_size = 4, #Size in inches of each subplot
                          show_colorbar = False, #Don't need a colorbar with just one trace
                          force_square = True, #If you love square plots, this is for you!
                          plot_fits = [True]*3) #Overlay the best fit, need to specify for each of the plot_types
print(resObj1)
plt.show()



# ======================================================================================================= 
# EMCEE
# ======================================================================================================= 
"""
#Call the emcee hook and pass it the fit function that calculates your residual.
#Since we already ran a fit, emcee will use that fit for its starting guesses.
resObj2.do_emcee(scr.cmplxIQ_fit, nwalkers = 30, steps = 1000, burn=200)

#Check to see that a emcee result exists
print('Does an emcee chain exist? ', resObj2.hasChain)

#Look at the first few rows of the output chain:
chains = resObj2.emcee_result['default']['result'].flatchain

print('\nHead of chains:')
pp.pprint(chains.head())

#Compare withe the mle values (percent difference):
#Maximum liklihood estimates (MLE) are stored in Resonator.mle_vals
#lmfit best fit values for varied parameters are in Resonator.lmfit_vals
diffs = list(zip(resObj2.mle_labels, (resObj2.mle_vals - resObj2.lmfit_vals)*100/resObj2.lmfit_vals))

print('\nPerecent difference:')
pp.pprint(diffs)


#Plot the triangle plot, and overlay the best fit values with dashed black lines (default)
#You can see that the least-squares fitter did a very nice job of getting the values right

#You can also see that there is some strange non-gaussian parameter space that the MCMC
#analysis maps out! This is kind of wierd, but not too worrisome. It is probably suggestive
#that more care is needed in choosing good options for the MCMC engine.

figGTC = pygtc.plotGTC(chains, truths = [resObj2.lmfit_vals])

plt.show()
"""