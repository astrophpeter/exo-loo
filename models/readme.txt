Files for simulated data of HD209458b. 
The data points are modelled after the binned spectrum of Sing et al 2016 as used in Welbanks et al 2020.
The input values to the model correspond to the median retrieved values using the 'standard' 18 parameter model.
Three folders are included here:
-retrieved_HD209458b: Full model with all parameters set to the retrieved median values.
-retrieved_HD209458b_noK: Full model but the K abundance is set to 1e-25 (i.e. no K in the spectrum)
-retrieved_HD209458b_noNa: Full model but the Na abundance is set to 1e-25 (i.e. no Na in the spectrum)

The files have four columns: Wavelength (center of the bin), Bin width (half of the bin), Transit depth (%), Error (%)
The error has been modelled by superimposing the error bars of Sing 2016 on the binned model. No shift has been included.
