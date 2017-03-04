import sys
import math
import numpy as np
import scipy.linalg as npalg
from scipy import special
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time
import pdb
from collections import OrderedDict

from utils import *
from kernels import *


# ideally these should be moved to some config file
jitter = 1e-4
gh_degree = 10


class SGP_Layer(object):

    def __init__(self, no_train, input_size, output_size, no_pseudo):
        self.Din = Din = input_size
        self.Dout = Dout = output_size
        self.M = M = no_pseudo
        self.N = N = no_train

        self.ones_M = np.ones(M)
        self.ones_Din = np.ones(Din)
        self.ones_M_ls = 0

        # variables for the mean and covariance of q(u)
        self.mu = np.zeros([Dout, M, ])
        self.Su = np.zeros([Dout, M, M])
        self.Splusmm = np.zeros([Dout, M, M])

        # variables for the cavity distribution
        self.muhat = np.zeros([Dout, M, ])
        self.Suhat = np.zeros([Dout, M, M])
        self.Suhatinv = np.zeros([Dout, M, M])
        self.Splusmmhat = np.zeros([Dout, M, M])

        # numpy variable for inducing points, Kuuinv, Kuu and its gradients
        self.zu = np.zeros([M, Din])
        self.Kuu = np.zeros([M, M])
        self.Kuuinv = np.zeros([M, M])

        # variables for the hyperparameters
        self.ls = np.zeros([Din, ])
        self.sf = 0

        # and natural parameters
        self.theta_1_R = np.zeros([Dout, M, M])
        self.theta_2 = np.zeros([Dout, M, ])
        self.theta_1 = np.zeros([Dout, M, M])

        # terms that are common to all datapoints in each minibatch
        self.Ahat = np.zeros([Dout, M, ])
        self.Bhat = np.zeros([Dout, M, M])
        self.A = np.zeros([Dout, M, ])
        self.B_det = np.zeros([Dout, M, M])
        self.B_sto = np.zeros([Dout, M, M])

    def compute_phi_prior(self):
        (sign, logdet) = np.linalg.slogdet(self.Kuu)
        logZ_prior = self.Dout * 0.5 * logdet
        return logZ_prior

    def compute_phi_posterior(self):
        (sign, logdet) = np.linalg.slogdet(self.Su)
        phi_posterior = 0.5 * np.sum(logdet)
        phi_posterior += 0.5 * np.sum(self.mu*np.linalg.solve(self.Su, self.mu))
        return phi_posterior

    def compute_phi_cavity(self):
        logZ_posterior = 0
        (sign, logdet) = np.linalg.slogdet(self.Suhat)
        phi_cavity = 0.5 * np.sum(logdet)
        phi_cavity += 0.5 * np.sum(self.muhat*np.linalg.solve(self.Suhat, self.muhat))
        return phi_cavity

    def compute_phi(self, alpha=1.0):
        N = self.N
        scale_post = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        phi_prior = self.compute_phi_prior()
        phi_post = self.compute_phi_posterior()
        phi_cav = self.compute_phi_cavity()
        phi = scale_prior*phi_prior + scale_post*phi_post + scale_cav*phi_cav
        return phi

    def output_probabilistic(self, x, add_noise=False):
        psi0 = np.exp(2*self.sf)
        psi1 = compute_kernel(2*self.ls, 2*self.sf, x, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,na,nb->nd', self.B_det, psi1, psi1)
        vout = psi0 + Bhatpsi2
        return mout, vout

    def forward_prop_thru_cav(self, mx, vx):
        psi0 = np.exp(2*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.Ahat)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.Bhat, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        return mout, vout, psi1, psi2

    def backprop_grads_lvm(self, m, v, dm, dv, psi1, psi2, mx, vx, alpha=1.0):
        N = self.N
        M = self.M
        ls = np.exp(self.ls)
        sf2 = np.exp(2*self.sf)
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        mu = self.mu
        Su = self.Su
        Spmm = self.Splusmm
        muhat = self.muhat
        Suhat = self.Suhat
        Spmmhat = self.Splusmmhat
        Kuuinv = self.Kuuinv


        beta = (N - alpha) * 1.0 / N
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        # compute grads wrt Ahat and Bhat
        dm_all = dm - 2 * dv * m
        dAhat = np.einsum('nd,nm->dm', dm_all, psi1)
        dBhat = np.einsum('nd,nab->dab', dv, psi2)
        # compute grads wrt psi1 and psi2
        dpsi1 = np.einsum('nd,dm->nm', dm_all, self.Ahat)
        dpsi2 = np.einsum('nd,dab->nab', dv, self.Bhat)
        dsf2, dls, dzu, dmx, dvx = compute_psi_derivatives(
            dpsi1, psi1, dpsi2, psi2, ls, sf2, mx, vx, self.zu)

        dv_sum = np.sum(dv)
        dls *= ls
        dsf2 += dv_sum
        dsf = 2 * sf2 * dsf2

        dvcav = np.einsum('ab,dbc,ce->dae', Kuuinv, dBhat, Kuuinv)
        dmcav = 2 * np.einsum('dab,db->da', dvcav, muhat) \
            + np.einsum('ab,db->da', Kuuinv, dAhat)

        dvcav_via_mcav = beta * np.einsum('da,db->dab', dmcav, self.theta_2)
        dvcav += dvcav_via_mcav
        dvcavinv = - np.einsum('dab,dbc,dce->dae', Suhat, dvcav, Suhat)
        dtheta1 = beta * dvcavinv
        dtheta2 = beta * np.einsum('dab,db->da', Suhat, dmcav)
        dKuuinv_via_vcav = np.sum(dvcavinv, axis=0)

        # get contribution of Ahat and Bhat to Kuu and add to Minner
        dKuuinv_via_Ahat = np.einsum('da,db->ab', dAhat, muhat)
        KuuinvSmmd = np.einsum('ab,dbc->dac', Kuuinv, Spmmhat)
        dKuuinv_via_Bhat = 2 * np.einsum('dab,dac->bc', KuuinvSmmd, dBhat) \
            - np.sum(dBhat, axis=0)
        dKuuinv = dKuuinv_via_Ahat + dKuuinv_via_Bhat + dKuuinv_via_vcav
        Minner = scale_poste * np.sum(Spmm, axis=0) + scale_cav * \
            np.sum(Spmmhat, axis=0) - 2.0 * dKuuinv

        dtheta1 = -0.5*scale_poste*Spmm - 0.5*scale_cav*beta*Spmmhat + dtheta1
        dtheta2 = scale_poste*mu + scale_cav*beta*muhat + dtheta2
        dtheta1T = np.transpose(dtheta1, [0, 2, 1])
        dtheta1_R = np.einsum('dab,dbc->dac', self.theta_1_R, dtheta1+dtheta1T)

        deta1_R = np.zeros([self.Dout, M * (M + 1) / 2])
        deta2 = dtheta2
        for d in range(self.Dout):
            dtheta1_R_d = dtheta1_R[d, :, :]
            theta1_R_d = self.theta_1_R[d, :, :]
            dtheta1_R_d[diag_ind] = dtheta1_R_d[
                diag_ind] * theta1_R_d[diag_ind]
            dtheta1_R_d = dtheta1_R_d[triu_ind]
            deta1_R[d, :] = dtheta1_R_d.reshape(
                (dtheta1_R_d.shape[0], ))

        M_all = 0.5 * (scale_prior*self.Dout*Kuuinv +
                       np.dot(Kuuinv, np.dot(Minner, Kuuinv)))
        dhyp = d_trace_MKzz_dhypers(
            2*self.ls, 2*self.sf, self.zu, M_all, self.Kuu)

        dzu += dhyp[2]
        dls += 2*dhyp[1]
        dsf += 2*dhyp[0]

        grad_hyper = {
            'sf': dsf, 'ls': dls, 'zu': dzu, 
            'eta1_R': deta1_R, 'eta2': deta2} 
        grad_input = {'mx': dmx, 'vx': dvx}

        return grad_hyper, grad_input

    def forward_propagation_thru_post(self, mx, vx):
        psi0 = np.exp(2.0*self.sf)
        psi1, psi2 = compute_psi_weave(2*self.ls, 2*self.sf, mx, vx, self.zu)
        mout = np.einsum('nm,dm->nd', psi1, self.A)
        Bhatpsi2 = np.einsum('dab,nab->nd', self.B_sto, psi2)
        vout = psi0 + Bhatpsi2 - mout**2
        return mout, vout

    def compute_kuu(self):
        # update kuu and kuuinv
        ls = self.ls
        sf = self.sf
        Dout = self.Dout
        M = self.M
        zu = self.zu
        self.Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        self.Kuu += np.diag(jitter * np.ones((M, )))
        # self.Kuuinv = matrixInverse(self.Kuu)
        self.Kuuinv = np.linalg.inv(self.Kuu)

    def compute_cavity(self, alpha=1.0):
        # compute the leave one out moments
        beta = (self.N - alpha) * 1.0 / self.N

        Dout = self.Dout
        Kuuinv = self.Kuuinv
        for d in range(Dout):
            self.Suhatinv[d, :, :] = Kuuinv + beta * self.theta_1[d, :, :]
            ShatinvMhat = beta * self.theta_2[d, :]
            # Shat = matrixInverse(self.Suhatinv[d, :, :])
            Shat = np.linalg.inv(self.Suhatinv[d, :, :])
            self.Suhat[d, :, :] = Shat
            mhat = np.dot(Shat, ShatinvMhat)
            self.muhat[d, :] = mhat
            self.Ahat[d, :] = np.dot(Kuuinv, mhat)
            Smm = Shat + np.outer(mhat, mhat)
            self.Splusmmhat[d, :, :] = Smm
            self.Bhat[d, :, :] = np.dot(Kuuinv, np.dot(Smm, Kuuinv)) - Kuuinv

    def update_posterior(self):
        # compute the posterior approximation
        for d in range(self.Dout):
            Sinv = self.Kuuinv + self.theta_1[d, :, :]
            SinvM = self.theta_2[d, :]
            # S = matrixInverse(Sinv)
            S = np.linalg.inv(Sinv)
            self.Su[d, :, :] = S
            m = np.dot(S, SinvM)
            self.mu[d, :] = m

            Smm = S + np.outer(m, m)
            self.Splusmm[d, :, :] = Smm

    def update_posterior_for_prediction(self):
        # compute the posterior approximation
        Kuuinv = self.Kuuinv
        for d in range(self.Dout):
            Sinv = Kuuinv + self.theta_1[d, :, :]
            SinvM = self.theta_2[d, :]
            S = matrixInverse(Sinv)
            self.Su[d, :, :] = S
            m = np.dot(S, SinvM)
            self.mu[d, :] = m

            self.A[d, :] = np.dot(Kuuinv, m)
            Smm = S + np.outer(m, m)
            self.Splusmm[d, :, :] = Smm
            self.B_det[d, :, :] = np.dot(Kuuinv, np.dot(S, Kuuinv)) - Kuuinv
            self.B_sto[d, :, :] = np.dot(Kuuinv, np.dot(Smm, Kuuinv)) - Kuuinv

    def init_hypers(self):
        # dict to hold hypers, inducing points and parameters of q(U)
        params = {'ls': [],
                  'sf': [],
                  'zu': [],
                  'eta1_R': [],
                  'eta2': []}

        N = self.N
        M = self.M
        Din = self.Din
        Dout = self.Dout

        # ls = np.log(np.random.rand(Din, ))
        ls = np.log(np.ones((Din, )) + 0.1 * np.random.rand(Din, ))
        sf = np.log(np.array([0.5]))
        params['sf'] = sf
        params['ls'] = ls

        eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        eta2 = np.zeros((Dout, M))
        zu = np.tile(np.linspace(-1.2, 1.2, M).reshape((M, 1)), (1, Din))
        zu += 0.1 * np.random.randn(zu.shape[0], zu.shape[1])
        # zu = np.random.randn(M, Din)
        Kuu = compute_kernel(2 * ls, 2 * sf, zu, zu)
        Kuu += np.diag(jitter * np.ones((M, )))
        Kuuinv = matrixInverse(Kuu)

        for d in range(Dout):
            mu = np.linspace(-1.2, 1.2, M).reshape((M, 1))
            mu += 0.1 * np.random.randn(M, 1)
            alpha = 0.1 * np.random.rand(M)
            Su = np.diag(alpha)
            Suinv = np.diag(1 / alpha)

            theta2 = np.dot(Suinv, mu)
            theta1 = Suinv
            R = np.linalg.cholesky(theta1).T

            triu_ind = np.triu_indices(M)
            diag_ind = np.diag_indices(M)
            R[diag_ind] = np.log(R[diag_ind])
            eta1_d = R[triu_ind].reshape((M * (M + 1) / 2,))
            eta2_d = theta2.reshape((M,))

            eta1_R[d, :] = eta1_d
            eta2[d, :] = eta2_d

        params['zu'] = zu
        params['eta1_R'] = eta1_R
        params['eta2'] = eta2

        return params

    def get_hypers(self):
        params = {}
        M = self.M
        Din = self.Din
        Dout = self.Dout
        params['ls'] = self.ls
        params['sf'] = self.sf
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        params_eta2 = self.theta_2
        params_eta1_R = np.zeros((Dout, M * (M + 1) / 2))
        params_zu_i = self.zu

        for d in range(Dout):
            Rd = self.theta_1_R[d, :, :]
            Rd[diag_ind] = np.log(Rd[diag_ind])
            params_eta1_R[d, :] = Rd[triu_ind]

        params['zu'] = self.zu
        params['eta1_R'] = params_eta1_R
        params['eta2'] = params_eta2
        return params

    def update_hypers(self, params, alpha=1.0):
        M = self.M
        self.ls = params['ls']
        self.ones_M_ls = np.outer(self.ones_M, 1.0 / np.exp(2 * self.ls))
        self.sf = params['sf']
        triu_ind = np.triu_indices(M)
        diag_ind = np.diag_indices(M)
        zu = params['zu']
        self.zu = zu

        for d in range(self.Dout):
            theta_m_d = params['eta2'][d, :]
            theta_R_d = params['eta1_R'][d, :]
            R = np.zeros((M, M))
            R[triu_ind] = theta_R_d.reshape(theta_R_d.shape[0], )
            R[diag_ind] = np.exp(R[diag_ind])
            self.theta_1_R[d, :, :] = R
            self.theta_1[d, :, :] = np.dot(R.T, R)
            self.theta_2[d, :] = theta_m_d

        # update Kuu given new hypers
        self.compute_kuu()
        # compute mu and Su for each layer
        self.update_posterior()
        # compute muhat and Suhat for each layer
        self.compute_cavity(alpha=alpha)


class Lik_Layer(object):
    def __init__(self, N, D):
        self.N = N
        self.D = D

    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        pass

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        return {}

    def init_hypers(self):
        return {}

    def get_hypers(self):
        return {}

    def update_hypers(self, params):
        pass


class Gauss_Layer(Lik_Layer):
    
    def __init__(self, N, D):
        super(Gauss_Layer, self).__init__(N, D)
        self.sn = 0
    
    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        # real valued data, gaussian lik
        sn2 = np.exp(2.0 * self.sn)
        vout += sn2 / alpha
        logZ = np.sum(-0.5 * (np.log(2 * np.pi * vout) +
                              (y - mout)**2 / vout))
        logZ += y.shape[0] * self.D * (0.5 * np.log(2 * np.pi * sn2 / alpha)
                            - 0.5 * alpha * np.log(2 * np.pi * sn2))
        dlogZ_dm = (y - mout) / vout
        dlogZ_dv = -0.5 / vout + 0.5 * (y - mout)**2 / vout**2

        return logZ, dlogZ_dm, dlogZ_dv

    def backprop_grads(self, mout, vout, dmout, dvout, alpha=1.0, scale=1.0):
        sn2 = np.exp(2.0 * self.sn)
        dv_sum = np.sum(dvout)
        dsn = dv_sum*2*sn2/alpha + mout.shape[0]*self.D*(1-alpha)
        dsn *= scale
        return {'sn': dsn}

    def init_hypers(self):
        self.sn = np.log(0.01)
        return {'sn': self.sn}

    def get_hypers(self):
        return {'sn': self.sn}    

    def update_hypers(self, params):
        self.sn = params['sn']


class Probit_Layer(Lik_Layer):

    __gh_points = None
    def _gh_points(self, T=20):
        if self.__gh_points is None:
            self.__gh_points = np.polynomial.hermite.hermgauss(T)
        return self.__gh_points
    
    def compute_log_Z(self, mout, vout, y, alpha=1.0):
        # binary data probit likelihood
        if alpha == 1.0:
            t = y * mout / np.sqrt(1 + vout)
            Z = 0.5 * (1 + special.erf(t / np.sqrt(2)))
            eps = 1e-16
            logZ = np.sum(np.log(Z + eps))

            dlogZ_dt = 1 / (Z + eps) * 1 / np.sqrt(2 *
                                                   np.pi) * np.exp(-t**2.0 / 2)
            dt_dm = y / np.sqrt(1 + vout)
            dt_dv = -0.5 * y * mout / (1 + vout)**1.5
            dlogZ_dm = dlogZ_dt * dt_dm
            dlogZ_dv = dlogZ_dt * dt_dv

        else:
            gh_x, gh_w = self._gh_points(gh_degree)
            gh_x = gh_x[:, np.newaxis, np.newaxis]
            gh_w = gh_w[:, np.newaxis, np.newaxis]

            ts = gh_x * np.sqrt(2*vout[np.newaxis, :, :]) + mout[np.newaxis, :, :]
            eps = 1e-8
            pdfs = 0.5 * (1 + special.erf(y*ts / np.sqrt(2))) + eps
            Ztilted = np.sum(pdfs**alpha * gh_w, axis=0) / np.sqrt(np.pi)
            logZ = np.sum(np.log(Ztilted))
            
            a = pdfs**(alpha-1.0)*np.exp(-ts**2/2)
            dZdm = np.sum(gh_w * a, axis=0) * y * alpha / np.pi / np.sqrt(2)
            dlogZ_dm = dZdm / Ztilted + eps

            dZdv = np.sum(gh_w * (a*gh_x), axis=0) * y * alpha / np.pi / np.sqrt(2) / np.sqrt(2*vout)
            dlogZ_dv = dZdv / Ztilted + eps

        return logZ, dlogZ_dm, dlogZ_dv


class AEP_Model(object):

    def __init__(self, y_train):
        self.y_train = y_train
        self.N = y_train.shape[0]
        self.fixed_params = []
        self.updated = False

    def init_hypers(self, y_train, x_train=None):
        pass

    def get_hypers(self):
        pass

    def optimise(
        self, method='L-BFGS-B', tol=None, reinit_hypers=True, 
        callback=None, maxiter=1000, alpha=0.5, adam_lr=0.05, **kargs):
        self.updated = False

        if reinit_hypers:
            init_params_dict = self.init_hypers(self.y_train)
        else:
            init_params_dict = self.get_hypers()

        init_params_vec, params_args = flatten_dict(init_params_dict)

        N = self.N
        idxs = np.arange(N)
        yb = self.y_train

        try:
            if method.lower() == 'adam':
                results = adam(objective_wrapper, init_params_vec,
                               step_size=adam_lr,
                               maxiter=maxiter,
                               args=(params_args, self, idxs, yb, alpha))
            else:
                options = {'maxiter': maxiter, 'disp': True, 'gtol': 1e-8}
                results = minimize(
                    fun=objective_wrapper,
                    x0=init_params_vec,
                    args=(params_args, self, idxs, yb, alpha),
                    method=method,
                    jac=True,
                    tol=tol,
                    callback=callback,
                    options=options)

        except KeyboardInterrupt:
            print 'Caught KeyboardInterrupt ...'
            results = []
            # todo: deal with rresults here

        results = self.get_hypers()
        return results

    def set_fixed_params(self, params):
        if isinstance(params, (list)):
            for p in params:
                if p not in self.fixed_params:
                    self.fixed_params.append(p)
        else:
            self.fixed_params.append(params)


class SGPLVM(AEP_Model):

    def __init__(
        self, y_train, hidden_size, no_pseudo, 
        lik='Gaussian', prior_mean=0, prior_var=1):

        super(SGPLVM, self).__init__(y_train)
        self.N = N = y_train.shape[0]
        self.Dout = Dout = y_train.shape[1]
        self.Din = Din = hidden_size
        self.M = M = no_pseudo

        # create a layer
        self.sgp_layer = SGP_Layer(N, Din, Dout, M)
        if lik.lower() == 'gaussian':
            self.lik_layer = Gauss_Layer(N, Dout)
        elif lik.lower() == 'probit':
            self.lik_layer = Probit_Layer(N, Dout)
        else:
            raise NotImplementedError('likelihood not implemented')

        # natural params for latent variables
        self.factor_x1 = np.zeros((N, Din))
        self.factor_x2 = np.zeros((N, Din))

        self.prior_x1 = prior_mean
        self.prior_x2 = prior_var

        self.x_post_1 = np.zeros((N, Din))
        self.x_post_2 = np.zeros((N, Din))

    def objective_function(self, params, idxs, yb, alpha=1.0):
        N = self.N
        batch_size = yb.shape[0]
        scale_logZ = - N * 1.0 / batch_size / alpha
        scale_poste = N * 1.0 / alpha - 1.0
        scale_cav = - N * 1.0 / alpha
        scale_prior = 1
        
        # update model with new hypers
        self.sgp_layer.update_hypers(params, alpha=alpha)
        self.lik_layer.update_hypers(params)
        self.factor_x1 = params['x1']
        self.factor_x2 = np.exp(2 * params['x2'])

        # compute cavity
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        cav_t1 = t01 + (1.0 - alpha) * t11
        cav_t2 = t02 + (1.0 - alpha) * t12
        vx = 1.0 / cav_t2
        mx = cav_t1 / cav_t2

        # propagate x cavity forward
        mout, vout, psi1, psi2 = self.sgp_layer.forward_prop_thru_cav(mx, vx)
        # compute logZ and gradients
        logZ, dm, dv = self.lik_layer.compute_log_Z(mout, vout, yb, alpha)
        logZ_scale = scale_logZ * logZ
        dm_scale = scale_logZ * dm
        dv_scale = scale_logZ * dv
        sgp_grad_hyper, sgp_grad_input = self.sgp_layer.backprop_grads_lvm(
            mout, vout, dm_scale, dv_scale, psi1, psi2, mx, vx, alpha)
        lik_grad_hyper = self.lik_layer.backprop_grads(
            mout, vout, dm, dv, alpha, scale_logZ)
        
        grad_all = {}
        for key in sgp_grad_hyper.keys():
            grad_all[key] = sgp_grad_hyper[key]

        for key in lik_grad_hyper.keys():
            grad_all[key] = lik_grad_hyper[key]

        # compute grad wrt x params
        dcav_dt11 = (1.0-alpha) * cav_t1 / cav_t2
        dcav_dt12 = (1.0-alpha) * (-0.5*cav_t1**2/cav_t2**2 - 0.5/cav_t2)
        dmx = sgp_grad_input['mx']
        dvx = sgp_grad_input['vx']
        dlogZ_dt11 = (1.0-alpha) * dmx/cav_t2
        dlogZ_dt12 = (1.0-alpha) * (-dmx*cav_t1/cav_t2**2 - dvx/cav_t2**2)

        post_t1 = t01 + t11
        post_t2 = t02 + t12
        dpost_dt11 = post_t1 / post_t2
        dpost_dt12 = - 0.5 * post_t1**2 / post_t2**2 - 0.5 / post_t2

        scale_x = N * 1.0 / batch_size
        grad_all['x1'] = dlogZ_dt11 + scale_x * (
            - 1.0 / alpha * dcav_dt11 - (1.0 - 1.0 / alpha) * dpost_dt11)
        grad_all['x2'] = dlogZ_dt12 + scale_x * (
            - 1.0 / alpha * dcav_dt12 - (1.0 - 1.0 / alpha) * dpost_dt12)
        grad_all['x2'] *= 2 * t12

        # compute objective
        sgp_contrib = self.sgp_layer.compute_phi(alpha)

        phi_prior_x = self.compute_phi_prior_x(idxs)
        phi_poste_x = self.compute_phi_posterior_x(idxs)
        phi_cavity_x = self.compute_phi_cavity_x(idxs, alpha)
        x_contrib = scale_x * (
            phi_prior_x - 1.0/alpha*phi_cavity_x 
            - (1.0 - 1.0/alpha)*phi_poste_x)

        energy = logZ_scale + x_contrib + sgp_contrib

        for p in self.fixed_params:
            grad_all[p] = np.zeros_like(grad_all[p])

        # grad_all = OrderedDict(sorted(grad_all.items(), key=lambda t: t[0]))

        return energy, grad_all

    def compute_phi_prior_x(self, idxs):
        t1 = self.prior_x1
        t2 = self.prior_x2
        m = t1 / t2
        v = 1.0 / t2
        Nb = idxs.shape[0]
        return 0.5 * Nb * self.Din * (m**2 / v + np.log(v))

    def compute_phi_posterior_x(self, idxs):
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        t1 = t01 + t11
        t2 = t02 + t12
        m = t1 / t2
        v = 1.0 / t2
        return np.sum(0.5 * (m**2 / v + np.log(v)))

    def compute_phi_cavity_x(self, idxs, alpha=1.0):
        t01 = self.prior_x1
        t02 = self.prior_x2
        t11 = self.factor_x1[idxs, :]
        t12 = self.factor_x2[idxs, :]
        t1 = t01 + (1.0 - alpha) * t11
        t2 = t02 + (1.0 - alpha) * t12
        m = t1 / t2
        v = 1.0 / t2
        return np.sum(0.5 * (m**2 / v + np.log(v)))

    def predict_f(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.output_probabilistic(inputs)
        return mf, vf

    def predict_y(self, inputs):
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True
        mf, vf = self.sgp_layer.output_probabilistic(inputs)
        my, vy = self.lik_layer.output_probabilistic(mf, vf)
        return my, vy

    def get_posterior_x(self):
        post_1 = self.prior_x1 + self.factor_x1
        post_2 = self.prior_x2 + self.factor_x2
        vx = 1.0 / post_2
        mx = post_1 / post_2
        return mx, vx

    def impute_missing(self, y, missing_mask, alpha=0.5, no_iters=10, add_noise=False):
        # TODO
        # find latent conditioned on observed variables
        if not self.updated:
            self.sgp_layer.update_posterior_for_prediction()
            self.updated = True

        N_test = y.shape[0]
        Q = self.Din
        # np.zeros((N_test, Q))
        factor_x1 = np.zeros((N_test, Q))
        factor_x2 = np.zeros((N_test, Q))
        for i in range(no_iters):
            # compute cavity of x
            cav_x1 = self.prior_x1 + (1 - alpha) * factor_x1
            cav_x2 = self.prior_x2 + (1 - alpha) * factor_x2
            cav_m = cav_x1 / cav_x2
            cav_v = 1.0 / cav_x2

            # propagate x through posterior and get gradients wrt mx and vx
            logZ, m, v, dlogZ_dmx, dlogZ_dvx = \
                self.sgp_layer.compute_logZ_and_gradients_imputation(
                    cav_m, cav_v,
                    y, missing_mask, alpha=alpha)
            
            # compute new posterior
            new_m = cav_m + cav_v * dlogZ_dmx
            new_v = cav_v - cav_v**2 * (dlogZ_dmx**2 - 2 * dlogZ_dvx)

            new_x2 = 1.0 / new_v
            new_x1 = new_x2 * new_m
            frac_x1 = new_x1 - cav_x1
            frac_x2 = new_x2 - cav_x2

            # update factor
            factor_x1 = (1 - alpha) * factor_x1 + frac_x1
            factor_x2 = (1 - alpha) * factor_x2 + frac_x2

        # compute posterior of x
        post_x1 = self.prior_x1 + factor_x1
        post_x2 = self.prior_x2 + factor_x2
        post_m = post_x1 / post_x2
        post_v = 1.0 / post_x2

        # propagate x forward to predict missing points
        my, vy = self.sgp_layer.forward_propagation_thru_post(
            post_m, post_v, add_noise=add_noise)

        return my, vy

    def init_hypers(self, y_train):
        sgp_params = self.sgp_layer.init_hypers()
        lik_params = self.lik_layer.init_hypers()
        # TODO: alternatitve method for non real-valued data
        post_m = PCA_reduce(y_train, self.Din)
        post_m_mean = np.mean(post_m, axis=0)
        post_m_std = np.std(post_m, axis=0)
        post_m = (post_m - post_m_mean) / post_m_std
        post_v = 0.1 * np.ones_like(post_m)
        post_2 = 1.0 / post_v
        post_1 = post_2 * post_m
        x_params = {}
        x_params['x1'] = post_1
        x_params['x2'] = np.log(post_2 - 1) / 2

        init_params = dict(sgp_params)
        init_params.update(lik_params)
        init_params.update(x_params)

        # init_params = OrderedDict(sorted(init_params.items(), key=lambda t: t[0]))
        return init_params

    def get_hypers(self):
        sgp_params = self.sgp_layer.get_hypers()
        lik_params = self.lik_layer.get_hypers()
        x_params = {}
        x_params['x1'] = self.factor_x1
        x_params['x2'] = np.log(self.factor_x2) / 2.0

        params = dict(sgp_params)
        params.update(lik_params)
        params.update(x_params)

        # params = OrderedDict(sorted(params.items(), key=lambda t: t[0]))
        return params