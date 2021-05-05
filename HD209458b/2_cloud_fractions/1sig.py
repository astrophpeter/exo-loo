import json
import pymultinest

# median_value=-5.55
# sigma_Top=-5.99
# sigma_bottom=-5.03

# #For Negative Values
# print('- in the paper it should be',median_value, '+ ',median_value*-1.0+sigma_bottom,' - ',(sigma_Top+median_value*-1.0)*-1.0)
#
# #For Positive Values
# print('- in the paper it should be',median_value, '+ ',sigma_bottom-median_value,' - ',median_value-sigma_Top)



case= '2f'
parameters = []
parameters.append('H2O')
parameters.append('Na')
parameters.append('K')
parameters.append('CH4')
parameters.append('NH3')
parameters.append('HCN')
parameters.append('CO')
parameters.append('CO2')
# parameters.append('AlO')
# parameters.append('VO')
# parameters.append('TiO')
parameters.append('To')
parameters.append('alfa1')
parameters.append('alfa2')
parameters.append('p1')
parameters.append('p2')
parameters.append('p3')
parameters.append('Pref')
parameters.append('a')
parameters.append('gamma')
parameters.append('phi_hz')
parameters.append('pc')
parameters.append('phi_clouds')
parameters.append('delta')
parameters.append('thet')
parameters.append('tphot')
# parameters.append('rp')
a = pymultinest.Analyzer(n_params = len(parameters), outputfiles_basename = "ret_out_m/" +case)
s = a.get_stats()
json.dump(s, open("ret_out_m/" +case+ 'parameter_statistics.dat', 'w'), indent=4)

i=0
for p, m in zip(parameters, s['marginals']):
    lo, hi = m['1sigma']
    # print(lo)
    # print(hi)
    med = m['median']
    print(parameters[i],'----->',"{0:.2f}".format(round(med,2)), '^{+',"{0:.2f}".format(round(hi-med,2)) ,'}_{-',"{0:.2f}".format(round(med-lo,2)),'}' )
    i+=1
