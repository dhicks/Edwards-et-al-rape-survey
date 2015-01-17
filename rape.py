# -*- coding: utf-8 -*-
'''
Bayesian model for Edwards et al., "Denying Rape but Endorsing Forceful 
Intercourse: Exploring Differences Among Responders," Violence and Gender 
1:4, 2014. DOI:10.1089/vio.2014.0022.  

Created on Sat Jan 17 10:38:59 2015
Dan Hicks

Table 1: 
                                  YES                NO
Intentions to force a woman       31.7% (n = 26)     68.3% (n = 56)
to sexual intercourse

Any intentions to rape a woman    13.6% (n = 11)     86.4% (n = 70)


Model:  
rho = percentage of men in population who would answer 'yes' in table 1
r   = percentage of men in sample who answered 'yes' in table 1

Pr(rho | r) = Pr( r | rho) * Pr(rho) / Pr(r)
                    = binom(n, k) rho^k (1-rho)^(n-k) * Pr(rho) / Pr(r)
                        where binom is the binomial coefficient
'''

from __future__ import division        # Fix division
import matplotlib.pyplot as plt
import pandas as pd
from scipy.misc import comb            # Binomial coefficient

# Number of subintervals
bins = 100

# Define left endpoints of intervals
endpoints = [x/bins for x in range(bins)]
# Define uninformative (uniform) prior
priors = [1/bins] * bins
# Observational data
# Number who answered 'yes'
k = 26
# Total respondents
n = k + 56

rho = pd.DataFrame(columns=['endpoint', 'prior', 'likelihood', 'post'])
rho['endpoint'] = endpoints
rho['prior'] = priors
# Calculate likelihood: binom(n, k) rho^k (1-rho)^(n-k)
binom_coeff = comb(n, k)
rho['likelihood'] = binom_coeff * rho['endpoint']**k * (1-rho['endpoint'])**(n-k)
# Marginal = Pr(r) = sum over rho of Pr(r | rho) * Pr(rho)
marginal = sum(rho['likelihood'] * rho['prior'])
# Posterior = likelihood * prior / marginal
rho['post'] = rho['likelihood'] * rho['prior'] / marginal

# Plot the posterior distibution
plt.plot(rho['endpoint'], rho['prior'], color='red')
plt.plot(rho['endpoint'], rho['post'])
# How much of the plot density is to the left of .2?
print(sum(rho[rho['endpoint'] <= .2]['post']))