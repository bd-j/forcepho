# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import division

import math
import numpy as np
import os
import time
import matplotlib.pylab as plt
import cPickle as pickle
import pandas as pd
import scipy as sp
import datetime
import itertools

from scipy.optimize import fsolve

import astropy
from astropy.io import fits

np.set_printoptions(precision=3)

from pylab import figure, cm
from matplotlib.colors import LogNorm

from scipy.optimize import least_squares

os.chdir('/Users/Jerry/Dropbox/Harvard/Astro_Research')

# This is how to solve systems of equations in python
# will be useful if we want to match moments, but we are not doing that
def equations(p):
    x, y = p
    return (x+y**2-4, math.exp(x) + x*y - 3)

x, y =  fsolve(equations, (1, 1))

print equations((x, y))

# http://www.ics.uci.edu/~smyth/courses/cs274/notes/EMnotes.pdf


# download the FITS image
miri_f770 = fits.open('PSF_MIRI_F770W_revV-1.fits')
miri_f770.info()

image_large = miri_f770[0].data
image_large = image_large/np.sum(image_large)
image_small = image_large[325:400,325:400] # size 75 x 75
image_small = image_small/np.sum(image_small)



def noncentral_moment(image_data,exp_x,exp_y):
    x_max, y_max = image_data.shape
    loc_x = np.tile(np.arange(x_max),(y_max,1)).T
    loc_y = np.tile(np.arange(y_max),(x_max,1))
    return np.sum(image_data * (loc_x)**exp_x * (loc_y)**exp_y)

def central_moment(image_data,exp_x,exp_y):
    x_max, y_max = image_data.shape
    loc_x = np.tile(np.arange(x_max),(y_max,1)).T
    loc_y = np.tile(np.arange(y_max),(x_max,1))
    mean_x = np.sum(np.multiply(image_data,loc_x))/np.sum(image_data)
    mean_y = np.sum(np.multiply(image_data,loc_y))/np.sum(image_data)
    return np.sum(image_data * (loc_x-mean_x)**exp_x * (loc_y-mean_y)**exp_y)

# return mean_x, mean_y
def mean_params(image_data):
    return [noncentral_moment(image_data,1,0)/np.sum(image_data),noncentral_moment(image_data,0,1)/np.sum(image_data)]

# return sigma_x, sigma_y, rho
def cov_params(image_data):
    mean_x, mean_y = mean_params(image_data)
    x_max, y_max = image_data.shape
    loc_x = np.array([[x for y in xrange(y_max)] for x in xrange(x_max)])
    loc_y = np.array([[y for y in xrange(y_max)] for x in xrange(x_max)])
    sigma_x = np.sqrt(np.sum(image_data*(loc_x-mean_x)**2)/np.sum(image_data))
    sigma_y = np.sqrt(np.sum(image_data*(loc_y-mean_y)**2)/np.sum(image_data))
    rho = np.sum(image_data*(loc_x-mean_x)*(loc_y-mean_y)/np.sum(image_data))/(sigma_x*sigma_y)
    return [sigma_x,sigma_y,rho]

def easy_init(image_data):
    return [np.sum(image_data)] + mean_params(image_data)+cov_params(image_data)


def random_init(image_data):
    mean_x, mean_y = mean_params(image_data)
    return [np.random.uniform(low=0.0,high=1.0), mean_x + np.random.uniform(low=-5.0,high=5.0), 
            mean_y + np.random.uniform(low=-5.0,high=5.0), np.random.uniform(low=0.0,high=50.0),
            np.random.uniform(low=0.0,high=50.0), np.random.uniform(low=-1.0,high=1.0)]
    
    

def mvn_pdf(params):
    pos_x,pos_y,amp,mu_x,mu_y,sigma_x,sigma_y,rho = params
    return amp*1/(2*np.pi*sigma_x*sigma_y*np.sqrt(1-rho**2))*np.exp(
            -1/(2*(1-rho**2)) * ( (pos_x-mu_x)**2/(sigma_x**2) + (pos_y-mu_y)**2/(sigma_y**2) 
                - 2*rho*(pos_x-mu_x)*(pos_y-mu_y)/(sigma_x*sigma_y)  ) )

def mvn_pdf_2d(params,x_max,y_max): # 0,1,...,x_max-1
    amp,mu_x,mu_y,sigma_x,sigma_y,rho = params
    def temp_func(pos_x,pos_y):
        return mvn_pdf([pos_x,pos_y,amp,mu_x,mu_y,sigma_x,sigma_y,rho])
    pos_x_range = np.arange(x_max)
    pos_y_range = np.arange(y_max)
    result = temp_func(pos_x_range[:,None],pos_y_range[None,:])
    return result

def mvn_pdf_2d_mix_fn(num_mix,x_max,y_max):
    def ret_func(params):
        ans = np.zeros([x_max,y_max])
        for i in xrange(num_mix):
            ans += mvn_pdf_2d(params[(6*i):(6*i+6)],x_max,y_max)
        return ans
    return ret_func

# util function from https://stackoverflow.com/questions/21030391/how-to-normalize-array-numpy
def normalized(a, axis=-1, order=1):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def fit_mvn_mix(image_data, num_mix, method_opt, method_init, 
                repeat=1,returnfull=True,printint=True,printfinal=True):
    assert method_opt in ['scipy_ls','em']
    assert method_init in ['random','greedy']
    x_max, y_max = image_data.shape
    all_init = np.array([0.0]*(6*num_mix))
    ans_repeat = []
    def create_ans(params,original_dict):
        ans = original_dict
        recon_image = mvn_pdf_2d_mix_fn(num_mix,x_max,y_max)(params)
        new_dict = {     'fitted_params':params, 
                    'original_image':image_data,
                    'recon_image': recon_image,
                    'error_original': np.sqrt(np.sum(image_data**2)),
                    'error_residual': np.sqrt(np.sum((image_data-recon_image)**2))
                    }
        ans.update(new_dict)
        return ans
    for _ in xrange(repeat):
        if method_init == 'random':
            for i in range(num_mix):
                all_init[(6*i):(6*i+6)] = random_init(image_data) 
        elif method_init == 'greedy': 
            residual_image = np.copy(image_data)
            for i in range(num_mix):
                all_init[(6*i):(6*i+6)] = easy_init(residual_image)
                residual_image -= - mvn_pdf_2d(all_init[(6*i):(6*i+6)],x_max,y_max)
        if method_opt == 'scipy_ls':
            res = least_squares(lambda params: image_data.flatten() - mvn_pdf_2d_mix_fn(num_mix,x_max,y_max)(params).flatten()
            , all_init, bounds = ([-np.inf,0,0,0,0,-1]*num_mix, [np.inf, x_max-1, y_max-1, np.inf, np.inf, 1]*num_mix))
            ans_repeat.append(create_ans(res.x))
        elif method_opt == 'em':
            def log_likelihood(params):
                return np.sum(image_data*np.log(mvn_pdf_2d_mix_fn(num_mix,x_max,y_max)(params)))
            curr_params = all_init
            curr_log_likelihood = log_likelihood(curr_params)
            # weights = 3-dim array x,y,k - weights[x,y,k] propto N(x,y|param_k) and sum over k = 1
            # param_k is param[6*k:(6*k+6)]
            weights = np.empty([x_max,y_max,num_mix]) 
            loop_count = 0
            while loop_count < 20000:
                # recompute the weights, first the unscaled
                for k in range(num_mix):
                    weights[:,:,k] = mvn_pdf_2d(curr_params[6*k:(6*k+6)],x_max,y_max)
                # scale that for any x,y, summing over k results in 1
                weights = normalized(weights,axis=2,order=1)
                Nk_list = np.zeros(num_mix)
                for k in range(num_mix):
                    Nk_list[k] = np.sum(image_data*weights[:,:,k])
                # update params
                new_params = np.zeros(6*num_mix)
                # value at p=(x,y) is x for loc_x and y for loc_y (for efficient operations)
                loc_x = np.array([[x for y in xrange(y_max)] for x in xrange(x_max)])
                loc_y = np.array([[y for y in xrange(y_max)] for x in xrange(x_max)])
                # we update with for loop over k because we think num_mix is small
                # if this is a time choke we can try to rewrite this with numpy parallelization
                # note that amp_k, mu_{k,x}, mu_{k,y}, sigma_{k,x}, sigma_{k,y}, rho_k
                # have indices 6k, 6k+1, 6k+2, 6k+3, 6k+4, 6k+5 respectively
                # update amp_k
                for k in range(num_mix):
                    new_params[6*k] = Nk_list[k]/np.sum(Nk_list)
                # update mu_k (mu_{k,x},mu_{k,y})
                for k in range(num_mix):
                    new_params[6*k+1] = np.sum(image_data*weights[:,:,k]*loc_x)/Nk_list[k]
                    new_params[6*k+2] = np.sum(image_data*weights[:,:,k]*loc_y)/Nk_list[k]
                # update Sigma_k (sigma_{k,x},sigma_{k,y},rho_k)
                for k in range(num_mix):
                    mu_x_new, mu_y_new = new_params[6*k+1], new_params[6*k+2]
                    # decentered x^2, y^2, xy to compute updates
                    new_params[6*k+3] = np.sqrt(np.sum(image_data*weights[:,:,k]*(loc_x-mu_x_new)**2)/Nk_list[k])
                    new_params[6*k+4] = np.sqrt(np.sum(image_data*weights[:,:,k]*(loc_y-mu_y_new)**2)/Nk_list[k])
                    new_params[6*k+5] = np.sum(image_data*weights[:,:,k]*
                              (loc_x-mu_x_new)*(loc_y-mu_y_new))/(Nk_list[k]*new_params[6*k+3]*new_params[6*k+4])
                # compute new likelihood, compare to old likelihood
                new_log_likelihood = log_likelihood(new_params)
                if np.isnan(new_log_likelihood):
                    print 'WARNING log likelihood is np.nan'
                    print 'curr params', curr_params.reshape([num_mix,6])
                    print 'new params', new_params.reshape([num_mix,6])
                    # create answer with the param right before log likelihood is nan
                    ans_repeat.append(create_ans(curr_params,
                            {'loop_count':loop_count, 'final_log_likelihood':new_log_likelihood}))
                    break
                if printint and (loop_count+1) % 5000 == 0:
                    print 'loop count', loop_count
                    print 'curr log likelihood', curr_log_likelihood
                    print 'new log likelihood:', new_log_likelihood
                    print 'curr params', curr_params.reshape([num_mix,6])
                    print 'new params', new_params.reshape([num_mix,6])
                    print 'diff', np.abs(new_log_likelihood - curr_log_likelihood)
                if np.abs(new_log_likelihood - curr_log_likelihood) < 1e-8:
                    ans_repeat.append(create_ans(new_params,
                            {'loop_count':loop_count, 'final_log_likelihood':new_log_likelihood}))
                    if printfinal:
                        print 'total loop count', loop_count
                        print 'final log likelihood', new_log_likelihood
                        print 'final params', new_params.reshape([num_mix,6])
                    break 
                curr_params = new_params
                curr_log_likelihood = new_log_likelihood 
                loop_count += 1
    if returnfull:
        return ans_repeat
    else:
        error_residuals = [temp['error_residual'] for temp in ans_repeat]
        min_index = np.argmin(error_residuals)
        return ans_repeat[min_index]


            
plt.imshow(image_data[325:400,325:400], cmap='gray', norm=LogNorm()); plt.colorbar();
plt.imshow(image_small, cmap='gray'); plt.colorbar();

np.sqrt(np.sum(image_small**2))      

a = np.array([0,1,2])
a = [0,1,2]
np.tile(a,(3,1)).T
np.vstack(a,3)
np.tile(a,(3,1))

ans_all_em_random = {}
ans_all_em_random_large = {}
for i in [6,7,8,9,10]:
    ans_all_em_random[i] = fit_mvn_mix(image_small, i, method_opt='em', method_init='random',repeat=5, returnfull=True)
    print 'Done: EM random, i =', i

pickle.dump(ans_all_em_random, open('ans_all_em_random.p', 'wb'))

plt.imshow(image_small, cmap='gray'); plt.colorbar(); #airydisk_2D_kernel
i = 10
#plt.imshow(airy_em_random[i][0]['recon_image'], cmap='gray'); plt.colorbar();
plt.imshow(image_small-ans_all_em_random[i][0]['recon_image'], cmap='gray'); plt.colorbar();
for i in range(1,11):
    print i, len(ans_all_em_random[i])
    


ans_all_greedy = {}
for i in [4]:
    ans_all_greedy[i] = fit_mvn_mix(image_small, i, method_opt='scipy_ls', method_init='greedy',repeat=1)
    print 'Done: greedy, i =', i
 
ans_all_rand = {}    
for i in [1,2,3,4]:
    ans_all_rand[i] = fit_mvn_mix(image_small, i, method_opt='scipy_ls', method_init='random', repeat=5, returnfull=True)
    print 'Done: random, i =', i

temp3 = temp = fit_mvn_mix(image_small, 3, method='scipy_ls_greedy', repeat=1)
temp = fit_mvn_mix(image_small, 4, method='scipy_ls_greedy', repeat=1)
ans_all_rand[2][0]['fitted_params'].reshape([2,6])

pickle.dump(ans_all_greedy, open('ans_all_greedy.p','wb'))
pickle.dump(ans_all_rand, open('ans_all_rand.p','wb'))

from astropy.convolution import AiryDisk2DKernel
airydisk_2D_kernel = AiryDisk2DKernel(10).array # sum all pixels = 1
plt.imshow(airydisk_2D_kernel, interpolation='none', origin='lower')
plt.imshow(airydisk_2D_kernel, cmap='gray') #vmin=1e-8 #norm=LogNorm()
plt.xlabel('x [pixels]')
plt.ylabel('y [pixels]')
plt.colorbar()
plt.show()

airy_em_random = {}
for i in range(1,11):
    airy_em_random[i] = fit_mvn_mix(airydisk_2D_kernel, i, method_opt='em', method_init='random',repeat=5, returnfull=True)
    print 'Done: EM random, i =', i
    
pickle.dump(airy_em_random, open('airy_em_random.p', 'wb'))

for i in range(1,11):
    print i, [round(airy_em_random[i][j]['error_residual']/airy_em_random[i][j]['error_original'],4)
    for j in range(5)]

for i in range(1,11):
    print i, len(airy_em_random[i])
plt.imshow(airydisk_2D_kernel, cmap='gray'); plt.colorbar();
i = 2
plt.imshow(airy_em_random[i][0]['recon_image'], cmap='gray'); plt.colorbar();
plt.imshow(airydisk_2D_kernel-airy_em_random[i][0]['recon_image'], cmap='gray'); plt.colorbar();
for i in range(1,11):
    print i, airy_em_random[i][0]['error_residual']
