"""
Phil M. provided this code to obtain stellar mass fits
"""

import numpy as np
from scipy import interpolate

class UniverseMachineMStarFit(object):
    def __init__(self, scatter=0.2, alpha_z0=None, mode="all",
                 rng=np.random.default_rng()):
        self.scatter = scatter
        self.alpha_z0 = alpha_z0
        self.mode = mode
        self.rng = rng
    def m_star(self, mpeak, z, scatter=None):
        if scatter is None: scatter = self.scatter
        
        mpeak = mpeak
        
        a = 1/(1 + z)
 
        if self.mode == "all":
            e0 = -1.435
            al_lna = -1.732
 
            ea = 1.831
            alz = 0.178
 
            e_lna = 1.368
            b0 = 0.482
 
            ez = -0.217
            ba = -0.841
 
            m0 = 12.035
            bz = -0.471
 
            ma = 4.556
            d0 = 0.411
 
            m_lna = 4.417
            g0 = -1.034
 
            mz = -0.731
            ga = -3.100
 
            if self.alpha_z0 is None:
                al0 = 1.963
            else:
                al0 = self.alpha_z0
            gz = -1.055
 
            ala = -2.316
            
        elif self.mode == "sat":
            e0    = -1.449
            ea    = -1.256
            e_lna = -1.031
            ez    =  0.108
            
            m0    = 11.896
            ma    =  3.284
            m_lna =  3.413
            mz    = -0.580
            
            if self.alpha_z0 is None:
                al0 = 1.949
            else:
                al0 = self.alpha_z0
                
            ala    = -4.096
            al_lna = -3.226
            alz    =  0.401
            
            b0    =  0.477
            ba   =  0.046
            bz = -0.214
            
            d0    =  0.357
            
            g0 = -0.755
            ga =  0.461
            gz =  0.025
 
        elif self.mode == "cen":
            e0 = -1.435
            ea =  1.813
            e_lna =  1.353
            ez = -0.214
            
            m0 = 12.081
            ma = 4.696
            m_lna = 4.485
            mz = -0.740
 
            if self.alpha_z0 is None:
                al0 = 1.957
            else:
                al0 = self.alpha_z0            
            ala = -2.650
            al_lna = -1.953
            alz =  0.204
 
            b0 = 0.474
            ba = -0.903
            bz = -0.492
            
            d0 = 0.386
            
            g0 = -1.065
            ga = -3.243
            gz = -1.107
            
        log10_M1_Msun = m0 + ma*(a-1) - m_lna*np.log(a) + mz*z
        e = e0 + ea*(a - 1) - e_lna*np.log(a) + ez*z
        al = al0 + ala*(a - 1) - al_lna*np.log(a) + alz*z
        b = b0 + ba*(a - 1) + bz*z
        d = d0
        g = 10**(g0 + ga*(a - 1) + gz*z)
 
        x = np.log10(mpeak/10**log10_M1_Msun)
      
        log10_Ms_M1 = (e - np.log10(10**(-al*x) + 10**(-b*x)) +
                       g*np.exp(-0.5*(x/d)**2))
                       
        log10_Ms_Msun = log10_Ms_M1 + log10_M1_Msun
 
        if self.scatter > 0.0:
            log_scatter = scatter*self.rng.normal(
                0, 1, size=np.shape(mpeak))
            log10_Ms_Msun += log_scatter
        
        Ms = 10**log10_Ms_Msun
        
        return Ms
 
    def m_halo(self, m_star, z):
        mh_0 = 10**np.linspace(4, 17, 200)
        ms_0 = self.m_star(mh_0, z, scatter=0)
 
        dln_ms_dln_mh_0 = np.zeros(len(mh_0))
        dln_ms_dln_mh_0[1:-1] = ((np.log(ms_0[2:]) - np.log(ms_0[:-2])) /
                                 (np.log(mh_0[2:]) - np.log(mh_0[:-2])))
        dln_ms_dln_mh_0[0]  = dln_ms_dln_mh_0[1]
        dln_ms_dln_mh_0[-1] = dln_ms_dln_mh_0[-2]
        
        f_beta = interpolate.interp1d(np.log10(ms_0), dln_ms_dln_mh_0)
        f_ms = interpolate.interp1d(np.log10(ms_0), np.log10(mh_0))
        beta = f_beta(np.log10(m_star))
        ms_to_mh = lambda ms: 10**f_ms(np.log10(ms))
 
        # From Symphony
        alpha = -1.92
        return invert_smhm(m_star, alpha, beta, self.scatter, ms_to_mh)
 
def invert_smhm(m_star, alpha, beta, sigma_ms, ms_to_mh):
    # alpha is the log-slope of HMF, beta is the log-slope of the SMHM relation
    # sigma_ms is in dex, m_star is in Msun, and ms_to_mh is a function
    # with units Msun -> Msun
    inv_sigma = sigma_ms/beta
    inv_mu = 10**((sigma_ms/beta)**2 * (1 + alpha) +
                  np.log10(ms_to_mh(m_star)))
    return inv_mu, inv_sigma
