""""

loo.py

This modeule contains a function to compute approximate leave one out (LOO) cross validation
using posterior samples from the full data set, and save and plot the results.

"""

from psis import psisloo
import matplotlib.pyplot as plt
import numpy as np

def plot_loo(loo_i, data, output_dir):
    """
    Plot the leave one out cross validation score over the
    data points
    """

    plt.clf()
    plt.errorbar(data[:,0], data[:,2],yerr=data[:,3],fmt='o',alpha=0.25)
    plt.scatter(data[:,0], data[:,2] ,c=loo_i)
    plt.xscale('log')
    plt.ylabel('Transit depth',fontsize=20)
    plt.xlabel('$\lambda$',fontsize=20)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('loo_i',fontsize=20)
    plt.savefig(f"{output_dir}loo_i.png")


def save_ouput(loo_i,pareto_k, output_dir):
    """
    Saves the output into numpy arrays
    """
    np.save(f"{output_dir}loo_i", loo_i)
    np.save(f"{output_dir}pareto_k",pareto_k)


def approximate_loo_cv(likelihood_samples, data, output_dir):
    """
    Makes plots and saves the loo cv results

    """

    loo, loo_i , pareto_k = psisloo(likelihood_samples)
    plot_pareto_k(pareto_k,data,output_dir)
    plot_loo(loo_i,data,output_dir)
    save_ouput(loo_i,pareto_k,output_dir)


def plot_pareto_k(pareto_k, data, output_dir, threshold=0.7):
    """
    Plots the pareto k values
    """	

    plt.clf()
    plt.scatter(data[:,0],pareto_k)
    plt.xscale('log')
    plt.axhline(0.7,linestyle='--',color='red',label="threshold")
    
    plt.ylabel('Pareto K',fontsize=20)
    plt.xlabel('$\lambda$',fontsize=20)
    plt.savefig(f"{output_dir}pareto_k.png")
