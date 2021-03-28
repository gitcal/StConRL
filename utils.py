'''
This file contains a number of utility functions for the estimation and control tasks. More specifically:
1) gather_trajectories: generates trajectories of the unknown system either by following a feedback control law
	from different initial states or by applying random input in samples states of the unknown system.
2) exp_kernel: is the exponentia lkernel function used in GP and STP estimation
3) GP, STP, GP_local, STP_local are the Gaussian, Student-t, local Gaussian and local Student-t process regression
	estimation methods
4) get_traj_unc_sets, get_traj_unc_sets_tp1: the first returns the uncertainrty sets for all the MPC trajectory 
	while the latter only for the next time step
5) get_Jacobian: computes the Jacobian matrix around specified lienarization points
6) minkowski_sum: computes the Minkowski sum between two sets in R
7) STP_MPC: implements the MPC controller
'''

import numpy as np
import scipy as sc
import IPython
from scipy import spatial
from cvxpy import *
from cvxpy.atoms.norm_inf import norm_inf
from cvxpy.atoms.elementwise.power import power
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# compute function F(x)=\frac{\int_a^x(\hat{f}''(t))^{2/5}dt}{\int_a^b(\hat{f}''(t))^{2/5}dt}   (**)
# import matplotlib.pylab as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection




def gather_trajectories(x0, xf, dt, n_sim, T, alpha_c, mu, sigma, method='random'):


	if method == 'control':
		# simpler example/ create trajectories for estimation
		x_trajectories = [] # save all trajectories
		y_trajectories = []

		x_0 = np.array([x0 + init_var*np.random.rand()]) # initial state (position and velocity), 
		x = np.zeros((T,2)) # vector of states and control input
		y = np.zeros((T,1))
		x[0, 0] = x_0 # initial state
		x[0,1] = - np.cbrt(alpha_c*x[0,0]) - alpha*(x[0,0]-xf)
		# noise  = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		noise  = np.random.normal(mu, sigma, 1)
		y[0,0] = np.cbrt(alpha_c*x[0,0])  + dt*x[0,1] + noise

		for i in range(1,T):
			#u[:,i-1] = - np.cbrt(x[:, i-1]) - alpha*(x[:,i-1]-xf)
			x[i,0] = y[i-1,0]
			x[i,1] = - np.cbrt(alpha_c*x[i,0]) - alpha*(x[i,0]-xf)
			noise  = np.random.normal(mu, sigma, 1)
			y[i,0] = np.cbrt(alpha_c*x[i,0])  + dt*x[i,1] + noise

		# save trajectory
		x_trajectories = x
		y_trajectories = y   


	else:
		# generate random data
		x_0 = 0
		x = np.zeros((T,2)) # vector of states and control input
		y = np.zeros((T,1))
		noise  = np.random.normal(mu, sigma, 1)
		# if x_0>=-2 and x_0<=2:
		# 		noise = 30*np.random.normal(mu, sigma, 1)
		x[0, 0] = x_0 # initial state
		x[0,1] = np.random.uniform(low=-2.0, high=2.0)
		
		y[0,0] = 5*np.cbrt(alpha_c*x[0,0])  + dt*x[0,1] + noise
		
		for i in range(1,T):
			#u[:,i-1] = - np.cbrt(x[:, i-1]) - alpha*(x[:,i-1]-xf)
			
			x[i,0] = y[i-1,0]
			x[i,1] = np.random.uniform(low=-0.5, high=0.5)
			noise  = np.random.normal(mu, sigma, 1)#stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
			if (x[i,0]>=-2) and (x[i,0]<=2):
				noise = 5*np.random.normal(mu, sigma, 1)

			y[i,0] = 5*np.cbrt(alpha_c*x[i,0])  + dt*x[i,1] + noise#np.random.normal(mu, sigma, 1)

		# save trajectory
		x_trajectories = x
		y_trajectories = y
		print('random')

	return x_trajectories, y_trajectories



# Define the exponentiated quadratic 
def exp_kernel(xa, xb, scale):
	"""Exponentiated quadratic  with σ=1"""
	# L2 distance (Squared Euclidian)
	sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean')
	return np.exp(sq_norm/(2 * scale))



def GP(X1, y1, X2_big, kernel, sigma_noise, scale):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	#X2 = X2_big
	k1, _ = X1.shape
	k2, _ = X2_big.shape
	# X2_big = np.expand_dims(X2_big, 1)
	# X2_big = X2_big.T
	mus = np.zeros(k2)
	Sigmas = np.zeros(k2)
	ind = 0
	for X2 in X2_big:
	   
		Sigma11 = kernel(X1, X1, scale) + sigma_noise * np.eye(k1)
		# Kernel of observations vs to-predict
		X2 = np.expand_dims(X2, 1).T
		# IPython.embed()
		Sigma12 = kernel(X1, X2, scale)
		# Solve
		solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
	   
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = Sigma22 + sigma_noise  - (solved @ Sigma12)
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas 



def STP(X1, y1, X2_big, kernel, sigma_noise, scale):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	k1, _ = X1.shape
	k2, _ = X2_big.shape
	# X2_big = np.expand_dims(X2_big, 1)
	# X2_big = X2_big.T
	mus = np.zeros(k2)
	Sigmas = np.zeros(k2)
	ind = 0
	dof = 5
	for X2 in X2_big:
		# dists = np.linalg.norm(X2-X1_big, axis=1)
		# inds = np.argpartition(dists, k-1)
		# inds = inds[0:k]
		
		Sigma_11 = kernel(X1, X1, scale)# check whtehr these tw oshould break
		Sigma11 = Sigma_11 + sigma_noise * np.eye(k1)
		# Kernel of observations vs to-predict
		X2 = np.expand_dims(X2, 1).T
		Sigma12 = kernel(X1, X2, scale)
		# Solve
		solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = (dof+y1.T@sc.linalg.solve(Sigma11,y1)-2)/(dof+len(X1)-2)*(Sigma22 + sigma_noise - (solved @ Sigma12))
		#IPython.embed()
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas  



def GP_local(X1_big, y1_big, X2_big, K, kernel, sigma_noise, scale):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	#X2 = X2_big
	k1, _ = X1_big.shape
	k2, _ = X2_big.shape
	mus = np.zeros(k2)
	Sigmas = np.zeros(k2)
	ind = 0
	for X2 in X2_big:
		dists = np.linalg.norm(X2-X1_big, axis=1)
		inds = np.argpartition(dists, K-1)
		inds = inds[0:K]
		X1 = X1_big[inds]
		y1 = y1_big[inds]
		X2 = np.expand_dims(X2, 1).T
		Sigma11 = kernel(X1, X1, scale) + sigma_noise * np.eye(K)
		# Kernel of observations vs to-predict
		Sigma12 = kernel(X1, X2, scale)
		# Solve
		solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = Sigma22 + sigma_noise  - (solved @ Sigma12)
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas 



def STP_local(X1_big, y1_big, X2_big, K, kernel, sigma_noise, scale):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	k1, _ = X1_big.shape
	k2, _ = X2_big.shape
	mus = np.zeros(k2)
	Sigmas = np.zeros(k2)
	ind = 0
	dof = 5
	for X2 in X2_big:
		dists = np.linalg.norm(X2-X1_big, axis=1)
		inds = np.argpartition(dists, K-1)
		inds = inds[0:K]
		X1 = X1_big[inds]
		y1 = y1_big[inds]
		Sigma_11 = kernel(X1, X1, scale)# check whtehr these tw oshould break
		X2 = np.expand_dims(X2, 1).T
		Sigma11 = Sigma_11 + sigma_noise * np.eye(K)
		# Kernel of observations vs to-predict
		Sigma12 = kernel(X1, X2, scale)
		# Solve
		solved = sc.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = (dof+y1.T@sc.linalg.solve(Sigma11,y1)-2)/(dof+len(X1)-2)*(Sigma22 + sigma_noise - (solved @ Sigma12))
		#IPython.embed()
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas  



def plot_fit(X1_big, y1_big, sigma):
	x_test = np.random.uniform(low=np.min(X1_big[:,0]), high=np.max(X1_big[:,0]),size=20)#10*np.random.randn(20,2)
	y_test = np.random.uniform(low=np.min(X1_big[:,1]), high=np.max(X1_big[:,1]),size=20)#10*np.random.randn(20,2)
	X, Y = np.meshgrid(x_test, y_test)
	
	n = len(x_test)
	x_temp = np.reshape(X, (n*n))
	y_temp = np.reshape(Y, (n*n))

	X2_big = np.array(list(zip(x_temp, y_temp)))
	# outputs = STP(X1_big, y1_big, X2_big, exp_kernel, sigma, 10)
	K = 10
	outputs = STP_local(X1_big, y1_big, X2_big, K, exp_kernel, sigma, 10)

	############################
	# https://matplotlib.org/stable/gallery/mplot3d/pathpatch3d.html
	############################
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	
	# Data for a three-dimensional line
	# zline = np.linspace(0, 15, 1000)
	# xline = np.sin(zline)
	# yline = np.cos(zline)
	# ax.plot3D(xline, yline, zline, 'gray')

	# Data for three-dimensional scattered points
	zdata = y1_big
	xdata = X1_big[:,0]
	ydata = X1_big[:,1]
	ax.scatter3D(xdata, ydata, zdata,  c='b')#cmap='Greens'

	zdata2 = outputs[0]
	unc = outputs[1]
	xdata2 = X2_big[:,0]
	ydata2 = X2_big[:,1]
	# verts = [(X2_big[i,0],X2_big[i,1],outputs[0][i] + outputs[1][i]) for i in range(len(outputs[0]))] 
	# verts.append([(X2_big[i,0],X2_big[i,1],outputs[0][i] - outputs[1][i]) for i in range(len(outputs[0]))] )
	# ax.add_collection3d(Poly3DCollection([verts],color='orange')) # Add a polygon instead of fill_between

	ax.scatter3D(xdata2, ydata2, zdata2, c='r')# c=zdata2,
	# w = 3
	# x,y = np.arange(100), np.random.randint(0,100+w,100)
	# z = np.array([y[i-w:i+w].mean() for i in range(3,100+w)])
	# temp1 = zdata2 + outputs[1]
	# temp2 = zdata2 - outputs[1]
	# fig = plt.figure()
	# ax = fig.add_subplot(111, projection='3d')
	# ax.add_collection3d(plt.fill_between(temp1[0],temp1[0],-0.1, color='orange', alpha=0.3,label="filled plot"),1, zdir='y')
	#ax.add_collection3d(plt.fill_between(x,y,-0.1, color='orange', alpha=0.3,label="filled plot"),1, zdir='y')
	# Xverts = [(xdata2[i],ydata2[i],zdata2[i] + outputs[1][i]) for i in range(len(outputs[1]))] 
	# ax.plot_trisurf(xdata2,ydata2,zdata2 + outputs[1])
	# ax.plot_trisurf(xdata2,ydata2,zdata2 - outputs[1])
	for i in np.arange(0, len(zdata2)):
		# ax.plot([fx[i]+xerror[i], fx[i]-xerror[i]], [fy[i], fy[i]], [fz[i], fz[i]], marker="_")
		# ax.plot([fx[i], fx[i]], [fy[i]+yerror[i], fy[i]-yerror[i]], [fz[i], fz[i]], marker="_")
		ax.plot([xdata2[i], xdata2[i]], [ydata2[i], ydata2[i]], [zdata2[i]+outputs[1][i], zdata2[i]-outputs[1][i]], color='red', alpha=.5, marker="_")

	ax.set_xlabel('$x_t$')
	ax.set_ylabel('$u_t$')
	ax.set_zlabel('$x_{t+1}$')
	plt.show()



def get_traj_unc_sets(init_traj, mus, sigmas, max_x_lin, Lip_sigma, Lip_grad_mu):
	unc_up = []
	unc_low = []
	x_temp_up = 0
	x_temp_low = 0
	for i in range(len(init_traj)):
		ell_x_u = np.max([np.linalg.norm(init_traj[i]-max_x_lin), np.linalg.norm(init_traj[i]+max_x_lin)])
		ell_x_u = max_x_lin
		x_temp_1 = mus[i] + 2*(sigmas[i] + Lip_sigma*ell_x_u)+Lip_grad_mu*ell_x_u**2/2
		x_temp_2 = mus[i] - 2*(sigmas[i] + Lip_sigma*ell_x_u)+Lip_grad_mu*ell_x_u**2/2

		x_temp_up, x_temp_low = minkowski_sum(x_temp_up, x_temp_low, x_temp_1, x_temp_2)
		unc_up.append(x_temp_up)
		unc_low.append(x_temp_low)
		#IPython.embed()

	return np.array(unc_up), np.array(unc_low)



def get_traj_unc_sets_tp1(init_traj, mus, sigmas, max_x_lin, Lip_sigma, Lip_grad_mu):
	unc_up = []
	unc_low = []
	x_temp_up = 0
	x_temp_low = 0
	
	ell_x_u = np.max([np.linalg.norm(init_traj[0]-max_x_lin), np.linalg.norm(init_traj[0]+max_x_lin)])
	ell_x_u = max_x_lin
	x_temp_1 = mus[0] + 2*(sigmas[0] + Lip_sigma*ell_x_u)+Lip_grad_mu*ell_x_u**2/2
	x_temp_2 = mus[0] - 2*(sigmas[0] + Lip_sigma*ell_x_u)+Lip_grad_mu*ell_x_u**2/2
	unc_up.append(x_temp_1)
	unc_low.append(x_temp_2)
		#IPython.embed()

	return np.array(unc_up), np.array(unc_low)



def get_Jacobian(X1, y1, init_traj, eps=0.01):
	x_Jac = init_traj
	n = x_Jac.shape
	u_Jac = np.zeros(n)
	X2_big = np.stack((x_Jac+eps, u_Jac), axis=-1)
	mu_x_p, _ = GP(X1, y1, X2_big, exp_kernel, 0.2, 10)
	X2_big = np.stack((x_Jac-eps, u_Jac), axis=-1)
	mu_x_m, _ = GP(X1, y1, X2_big, exp_kernel, 0.2, 10)
	X2_big = np.stack((x_Jac, u_Jac+eps), axis=-1)
	mu_u_p, _ = GP(X1, y1, X2_big, exp_kernel, 0.2, 10)
	X2_big = np.stack((x_Jac, u_Jac-eps), axis=-1)
	mu_u_m, _ = GP(X1, y1, X2_big, exp_kernel, 0.2, 10)
	X2_big = np.stack((x_Jac, u_Jac), axis=-1)
	mu, _ = GP(X1, y1, X2_big, exp_kernel, 0.2, 10)

	J_x = 0.5*(mu_x_p-mu_x_m)/eps
	J_u = 0.5*(mu_u_p-mu_u_m)/eps
	return mu, J_x, J_u


def minkowski_sum(x_temp_up, x_temp_low, x_temp_1, x_temp_2):

	# minkowski sum function
	# minksowski sum between two intervals [x_temp_low,x_temp_up] and 
	# [x_temp_1, x_temp_2]
	vertex_1 = x_temp_up + x_temp_1
	vertex_2 = x_temp_up + x_temp_2
	vertex_3 = x_temp_low + x_temp_1
	vertex_4 = x_temp_low + x_temp_2
	x_temp_up = np.max((vertex_1, vertex_2, vertex_3, vertex_4)) 
	x_temp_low = np.min((vertex_1, vertex_2, vertex_3, vertex_4))
	return x_temp_up, x_temp_low
	



def STP_MPC(mu, J_x, J_u, x_max, x_min, init_traj, T_mpc, x_0, xf, Q, R):
	feas_flag = True
	Q = 10 # state cost
	R = 1 # control cost
	dt = 1.0 # time step
	#flag = 3
	u_thresh = 10.0

	x = Variable((1, T_mpc+1)) # state variable
	u = Variable((1, T_mpc)) # control variable
	#d = Variable((1, T_mpc+1)) # control invariant set
	cost = 0
	constr = []
	constr += [x[:,0] == x_0]
	for t in range(T_mpc):
		cost += Q*norm(x[:,t] - xf)**2 + R*norm(u[:,t])**2#Q*norm(x[:,t]-d[:,t])**2 +R*norm(u[:,t]-(-1/dt*np.cbrt(root_cnst*xf)-1/(dt*3)*np.cbrt(root_cnst)*(np.cbrt(xf))**(-2)*(d[:,t]-xf)+xf))**2 
		# nonlinear cases
		constr += [x[:,t+1] == mu[t] + J_x[t]*(x[:,t]-init_traj[t]) + J_u[t]*u[:,t],# x[:,t] <= x_max[t], x[:,t] >= x_min[t],
			   norm_inf(u[:,t]) <= u_thresh]
	constr += [x[:,T_mpc][0] <= xf + 0.1, x[:,T_mpc][0] >= xf - 0.1]
	cost += Q*norm(x[:,T_mpc] - xf)**2
	
	problem = Problem(Minimize(cost), constr)
	problem.solve(solver=OSQP, verbose=True, eps_abs = 1.0e-02, eps_rel = 1.0e-02, eps_prim_inf = 1.0e-02) 
	# print('Total cost in iteration {} is {}.'.format(j, problem.value))
	# print('******************************')
	# print('MPC iteration {}.'.format(j))
	# print('******************************')
	# print('Target state is {}.'.format(xf))
	# print('******************************')
	# print('Actual end State is {}.'.format(x.value[0][0]))
	# print('******************************')
	return x.value[0], u.value[0]



