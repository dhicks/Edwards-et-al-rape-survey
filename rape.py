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
from numpy import inf                  # Infinity; for degenerate HDIs
import pandas as pd
from scipy.misc import comb            # Binomial coefficient
from scipy.optimize import minimize_scalar    # To calculate HDI
from scipy.stats import beta           # Beta distribution, for priors

def round_in_list(x, the_list, value=False) :
    '''
    Round x to the nearest value in the_list.  
    Assumes that the_list is a list of numbers (ints, floats, etc.), sorted 
    from smallest to largest.
    value determines whether the function returns the list index (default) or 
    the actual rounded value.  
    '''
    if len(the_list) == 0:
        raise ValueError('Empty list passed to round_in_list')
    # Find the first place where the list item is greater than x
    i = 0
    while (the_list[i] < x) and (i < len(the_list)-1):
        i += 1
    # If this happened with the 0th element, return now
    if i == 0:
        if value:
            return the_list[i]
        else:
            return i
    # If x is closer to the i-1st element, move back 1 step
    if (x - the_list[i-1]) < (the_list[i] - x):
        i = i-1
    if value:
        return the_list[i]
    else:
        return i

def density_interval_length (lend, pdf, coverage = 0.95):
    '''
    Takes a left endpoint, lend, a two-column numpy array representing a PDF, 
    and an optional coverage parameter.  
    Returns the length of the interval with that left endpoint that covers 
    at least (coverage) of the probability mass of the PDF.  This assumes 
    that the interval is indeed an interval.  
    Returns inf if no such interval exists.
    '''
    # Build the CDF
    cdf = []
    for i in range(pdf.shape[0]):   # pdf.shape[0] = # rows
        if i == 0:
            cdf.append(pdf[i, 1])   # row i, PDF value
        else:
            cdf.append(cdf[i-1] + pdf[i, 1])
    # Get the index for the x-value closest to lend
    # pdf[:,0].tolist() gives the x-values as a list
    index = round_in_list(lend, pdf[:,0].tolist())
    # If the cdf at the index is already too large, return inf
    if cdf[index] > 1-coverage:
        return inf
    # Get the right endpoint's probability value
    rprobvalue = cdf[index] + coverage
    # And get the corresponding index, this time using the list of CDF values
    rindex = round_in_list(rprobvalue, cdf)
    # Use this index to look up the x value for the right endpoint
    rend = pdf[rindex,0]
    # The length of the density interval is the difference between rend and lend
    return rend - lend
    

# Number of subintervals
bins = 1000
# Size of the HDI
coverage = .95

# Define left endpoints of intervals
endpoints = [x/bins for x in range(bins)]
# Define uninformative (uniform) prior
#priors = [1/bins] * bins
# Define skeptic's prior: left-heavy beta distribution
# Note that, the more skeptical the prior distribution, the narrower the 
# resulting posterior.  
# For the "intention to force" data, with a=1, b=50, the 95% HDI is (.14, .27); 
# with a=1, b=10, the 95% HDI is (.20, .38).  
# The skeptic needs about a=1, b>100 for the 95% HDI to be bounded above by .2.  
# I.e., the skeptic has to go in believing that fewer than 1% of men would 
# admit an intention to use force.  
priors = beta.pdf(endpoints, 1, 10)
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

# For readability, extract the posterior PDF before building the HDI
pdf = rho[['endpoint', 'post']].values
# The 95% HDI is the smallest interval that covers 95% of the probability mass. 
# To find this, we apply minimize_scalar to a function that calculates density 
# interval lengths. 
# TODO: abstract the HDI calculations into a function
hdi_left = minimize_scalar(
    lambda x : density_interval_length(x, pdf, coverage)
    )['x']
hdi_length = density_interval_length(hdi_left, pdf, coverage)
hdi_right = hdi_left + hdi_length
hdi_y = max([
    rho['post'][round_in_list(hdi_left, rho['endpoint'].tolist())], 
    rho['post'][round_in_list(hdi_right, rho['endpoint'].tolist())] ])
#print(hdi_left, hdi_right, hdi_length)


# Plot the prior and posterior distibutions
fig, ax1 = plt.subplots()
# Posterior, in blue
ax1.plot(rho['endpoint'], rho['post'], label=r'Pr($\rho$ | Data)')
ax1.set_ylabel(r'Pr($\rho$ | Data)', color='blue')
plt.xlabel(r'$\rho$')
# HDI
ax1.fill_between(rho['endpoint'], rho['post'], 
                 where=(rho['endpoint'] >= hdi_left) & 
                         (rho['endpoint'] < hdi_right), 
                 alpha=.5)
ax1.axhline(y = hdi_y, xmin = hdi_left, xmax = hdi_right, 
            color='blue', linewidth=2)
ax1.annotate('{:.2f}'.format(hdi_left), xy = (hdi_left, hdi_y), xytext = (0.5*hdi_left, hdi_y), color='blue')
ax1.annotate('{:.2f}'.format(hdi_right), xy = (hdi_right, hdi_y), xytext = (1.1*hdi_right, hdi_y), color='blue')
ax1.annotate('{:.0f}%'.format(coverage*100), xy = ((hdi_right + hdi_left)/2, 1.1*hdi_y), color='white')

ax2 = ax1.twinx()
# Prior, in red
ax2.plot(rho['endpoint'], rho['prior'], color='red', label=r'Pr($\rho$)')
ax2.set_ylabel(r'Pr($\rho$)', color='red')
plt.show()

