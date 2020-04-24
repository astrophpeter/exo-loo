import numpy as np
import run_params_ret as run_params
import celerite
from celerite import terms


'''
class: 	    Retrieval
purpose: 	Performs retrieval
            Plots the corner plot, and best fitting spectrum & P-T profile
            Compares evidences of 2 models
inputs: 	Instance of Profile
            Instance of Absorption
            Instance of RT
            Instance of Instrument
            Parameters from run_params_ret
'''

class Retrieval:

    def __init__(self):
        self.ret_params_list = run_params.ret_params
        self.default_params = run_params.default_params

        self.n_params = len(self.ret_params_list)


    '''
    GP
    '''


    def get_model(self, ret_params=None):
        print(ret_params)   #Just making sure that different ret_params are being passed
        result=np.ones(29) #Here, in reality we would call the other functions that would evaluate the model given the ret_params

        return result

    def loglikelihood_gp(self,cube,ndim,nparams): #This is a skeleton of what I would like to code for multines to receive
        # convert cube to a numpy array
        arr = np.zeros(ndim)
        for i in range(ndim):
            arr[i] = 1.0*cube[i]

        ret_params = {}

        for i,key in enumerate(self.ret_params_list):
            # print(key, i)
            ret_params[key] = arr[i]

        loglike = self.generate_model(ret_params)
        # print(loglike)

        return loglike
