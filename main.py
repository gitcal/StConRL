'''
This file runs the experimetns by generating data and solving the control task for specific chosen parameters

'''
import numpy as np
from cvxpy import *
import IPython
from cvxpy.atoms.norm_inf import norm_inf
from utils import *
import os

# Check if a storedData folder exist.	
if not os.path.exists('storedData'):
	os.makedirs('storedData')


class ML_Control(object):

	def __init__(
			self,
			n = 2, # state size
			m = 2, # input size
			T = 10, # duration of simulation
			dt = 1.0, # time step
			x_0 = 10.0, # initial state
			x_f = -1.0, # goal state
			v0 = 0.8, # initila speed
			mu = 0.0, # noise mean and variance
			sigma = 0.1, # noise level
			power = 1/3,  # dynamics cosntant
			root_cnst = 1.0, # dynamics constant
			u_thresh = 140.0, # control constraints
			n_sim = 30, # number of trajectories in dataset 
			infeas_flag = False, # optimization infeasibility flag
			method = 'random', # data acquisition method
			scale = 10.0, # kernel scaling parameter
			K = 10, # number of K closest data points for local learning 
			T_mpc = 4, # mpc horizon
			Q = 1.0, # state cost
			R = 100.0, # control cost
			mpc_sim = 8, # mpc simulations
			max_x_lin = 0.2): 

		self.n = n
		self.m = m 
		self.T = T 
		self.dt = dt 
		self.x_0 = x_0
		self.x_f = x_f
		self.v0 = v0
		self.mu = mu
		self.sigma = sigma
		self.power = power 
		self.u_thresh = u_thresh
		self.root_cnst = root_cnst
		self.n_sim = n_sim 
		self.infeas_flag = infeas_flag
		self.method = method
		self.scale = scale
		self.K = K
		self.T_mpc = T_mpc 
		self.Q = Q
		self.R = R
		self.mpc_sim = mpc_sim
		self.max_x_lin = max_x_lin
						
					


	def obtain_trajectories(self):

		# gather data either using a control law or using random control inputs
		x_0 = np.array([self.x_0+ 0.5*np.random.rand()])
		if self.method == 'control':
			x_trajectories, y_trajectories = gather_trajectories(x_0, self.x_f, self.dt, self.n_sim, self.T, self.root_cnst, self.mu, self.sigma, self.method)
		else:
			x_l = []
			y_l = []
			for i in range(10):
				x_trajectories, y_trajectories = gather_trajectories(x_0, self.x_f, self.dt, self.n_sim, self.T, self.root_cnst, self.mu, self.sigma, self.method)
				x_l.append(x_trajectories)
				y_l.append(y_trajectories)
		

		x_trajectories = np.concatenate( x_l, axis=0)
		y_trajectories = np.concatenate( y_l, axis=0)

		return x_trajectories, y_trajectories






	def control(self, x_trajectories, y_trajectories):

		x_mpc = np.zeros(self.mpc_sim+1) # mpc trajectory
		x_0 = self.x_0
		x_mpc[0] = x_0
		constr = [] # constraints
		IPython.embed()

		# Lipschitz constants
		Lip_sigma = 1
		Lip_grad_mu = 1


		# initialize linearization points for first mpc iteration
		init_traj = np.linspace(x_0+10**(-2), self.x_f+10**(-2), num=self.T_mpc)
		# kinearize around trajectory
		mus, J_x, J_u = get_Jacobian(x_trajectories, y_trajectories, init_traj)


		low_end = []
		high_end = []
		for j in range(self.mpc_sim):	
			x_max = init_traj + self.max_x_lin
			x_min = init_traj - self.max_x_lin
			u_Jac = np.zeros(len(init_traj))
			X2_big = np.stack((init_traj, u_Jac), axis=-1)# check inti traje and _0 oinde
			mus, sigmas = GP(x_trajectories, y_trajectories, X2_big, exp_kernel, self.sigma, self.scale)
			#mus, sigmas = STP_local(x_trajectories, y_trajectories, X2_big, self.K, exp_kernel, self.sigma, self.scale)
			IPython.embed()
			x_opt, u_opt = STP_MPC(mus, J_x, J_u, x_max, x_min, init_traj, self.T_mpc, x_0, self.x_f, self.Q, self.R)
			x_Jax_temp = x_opt[0]
			u_Jac_temp = u_opt[0]
			X2_big_temp = np.stack((x_Jax_temp, u_Jac_temp), axis=-1)
			X2_big_temp = np.expand_dims(X2_big_temp, 0)
			
			mus_temp, sigmas_temp = GP(x_trajectories, y_trajectories, X2_big_temp, exp_kernel, self.sigma, self.scale)
			#mus_temp, sigmas_temp = STP_local(x_trajectories, y_trajectories, X2_big_temp, self.K, exp_kernel, self.sigma, self.scale)
			# IPython.embed()
			unc_sets_up, unc_sets_low = get_traj_unc_sets_tp1(init_traj, mus_temp, sigmas_temp, self.max_x_lin, Lip_sigma, Lip_grad_mu)

			low_end.append(unc_sets_low)
			high_end.append(unc_sets_up)
			init_traj = x_opt[1:]
			mus, J_x, J_u = get_Jacobian(x_trajectories, y_trajectories, init_traj)
			noise = np.random.normal(self.mu, self.sigma, 1)

			# if (x_opt[0]>=-2) and (x_opt[0]<=2):
			# 	noise = 5*np.random.normal(self.mu, self.sigma, 1)
			x_0  =  5*np.cbrt(x_opt[0]) + u_opt[0] 

			
			x_0 = x_0 + noise#np.random.normal(mu, sigma, 1)
			x_mpc[j+1] = x_0# mpc trajectory
			# IPython.embed()

		low_end = np.array(low_end)
		high_end = np.array(high_end)
		print("Low end is {}".format(low_end.T))
		print("High end is {}".format(high_end.T))
		print("Real trajectory is {}".format(x_mpc[1:].T))

		return x_mpc, low_end, high_end



def main():
    alg = ML_Control()
    x_trajectories, y_trajectories = alg.obtain_trajectories()
    x_mpc, low_end, high_end = alg.control(x_trajectories, y_trajectories)




if __name__ == "__main__":
    main()








