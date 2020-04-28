"""
This is a minimum working example showing recovering posterior of gp parameters
with a constant mean model.
"""
import emcee
import numpy as np
import george
from george import kernels
from george.modeling import Model
import matplotlib.pyplot as plt
import corner
from scipy.optimize import minimize
class ConstantMeanModel(Model):
    parameter_names = ("constant",)

    def get_value(self, wavelength):
        return np.full_like(wavelength, self.constant)


gp_lsc_true = 1.3
gp_var_true = 1.5
constant_mean_true = 3.0

mean_model = ConstantMeanModel(constant=constant_mean_true)
true_params = mean_model.get_parameter_vector()


def generate_data(params, model, N, rng=(-5, 5)):
    gp = george.GP(gp_var_true * kernels.Matern32Kernel(gp_lsc_true))
    t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
    y = gp.sample(t)
    y += model(**params).get_value(t)
    yerr = 0.05
    y += yerr * np.random.randn(N)
    return t, y, yerr

t, y, yerr = generate_data({'constant': constant_mean_true}, ConstantMeanModel, 50)


# Plot the data generated
plt.errorbar(t, y, yerr=yerr, fmt='o')
plt.title('Data generated from the GP')
plt.show()

# now lets model the correlated noise, intitialising the gp, and mean mdoel
# to sensible values but not true values.
mean_model = ConstantMeanModel(constant=2.0)
gp = george.GP(np.var(y) * kernels.Matern32Kernel(2.0), mean=mean_model, fit_mean=True)
gp.compute(t, yerr)
print(f'GP parameters are: {gp.get_parameter_dict()}')


# log_likelihood + log_prior
def lnprob(params):
    gp.set_parameter_vector(params)
    return gp.log_likelihood(y, quiet=True) + gp.log_prior()

#find map to start mcmc chain
neg_lnprob = lambda params : -1.0 * lnprob(params)
initial = gp.get_parameter_vector()
map = minimize(neg_lnprob, initial).x
print(f'MAP :{map}')

# run mcmc
initial = map
ndim, nwalkers = len(initial), 32
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

print("Running mcmc...")
p0 = initial + 1e-8 * np.random.randn(nwalkers, ndim)
sampler.run_mcmc(p0, 10000, progress=True)


# now plot the fit over the data
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)

# The positions where the prediction should be computed.
x = np.linspace(-5, 5, 500)

# Plot 24 posterior samples.
samples = sampler.flatchain
for s in samples[np.random.randint(len(samples), size=24)]:
    gp.set_parameter_vector(s)
    mu = gp.sample_conditional(y, x)
    plt.plot(x, mu, color="#4682b4", alpha=0.3)

plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.xlim(-5, 5)
plt.title("fit with GP model")
plt.show()

#plot posteriors
samples = sampler.get_chain(discard=5000, flat=True)
corner.corner(samples, truths=[constant_mean_true, np.log(gp_var_true), np.log(gp_lsc_true)],
              labels=['constant', 'log(gp_var)', 'log(gp_lsc)'])
plt.show()













