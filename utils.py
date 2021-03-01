import numpy as np
import scipy as sc
from scipy.integrate import quad, simps
import IPython
from lwls import *
from scipy.interpolate import UnivariateSpline
import scipy.stats as stats
from scipy import spatial
from cvxpy import *
from cvxpy.atoms.norm_inf import norm_inf
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm
from scipy.optimize import curve_fit
from cvxpy.atoms.elementwise.power import power
from fractions import Fraction
from lwls import *
import sys
import seaborn as sns
#from video_plot.py import *
import matplotlib.cm as cm
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.animation as animation
from utils import *
import scipy.stats as stats
# compute function F(x)=\frac{\int_a^x(\hat{f}''(t))^{2/5}dt}{\int_a^b(\hat{f}''(t))^{2/5}dt}   (**)
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection # New import
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def integrand(x):
	# second derivative of third root of x raised to the -2/5 power
	return (((-2/9)**(2))**(1/5))*np.cbrt(x)**(-2)


def gather_trajectories(x0, xf, dt, n_sim, T, alpha_c, upper, lower, mu, sigma, method='random'):

	# mu = 0.0 # noise mean and variance
	# sigma = 0.1
	init_var = 2.5	
	if method == 'control':
		# simpler example/ create trajectories for estimation
		x_trajectories = [] # save all trajectories
		y_trajectories = []

		x_0 = np.array([x0 + init_var*np.random.rand()]) # initial state (position and velocity), 
		x = np.zeros((T,2)) # vector of states and control input
		y = np.zeros((T,1))
		alpha = 0.75 # control constant
		x[0, 0] = x_0 # initial state
		x[0,1] = - np.cbrt(alpha_c*x[0,0]) - alpha*(x[0,0]-xf)
		noise  = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
		y[0,0] = np.cbrt(alpha_c*x[0,0])  + dt*x[0,1] + noise.rvs(1)[0]#np.random.normal(mu, sigma, 1)
		
		for i in range(1,T):
			#u[:,i-1] = - np.cbrt(x[:, i-1]) - alpha*(x[:,i-1]-xf)
			x[i,0] = y[i-1,0]
			x[i,1] = - np.cbrt(alpha_c*x[i,0]) - alpha*(x[i,0]-xf)
			# upper = 0.05
			# lower = -0.05
			noise  = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
			y[i,0] = np.cbrt(alpha_c*x[i,0])  + dt*x[i,1] + noise.rvs(1)[0]#np.random.normal(mu, sigma, 1)

		# save trajectory
		x_trajectories = x
		y_trajectories = y


	else:
		# generate random data
		# maybe add here

		x_0 = 0#np.random.uniform(low=-6.0, high=6.0)# np.array([x0 + init_var*np.random.rand()]) # initial state (position and velocity), 
		x = np.zeros((T,2)) # vector of states and control input
		y = np.zeros((T,1))
		alpha = 0.75 # control constant
		noise  = np.random.normal(mu, sigma, 1)
		if x_0>=-10 and x_0<=-8:
				x_0 += 5*np.random.normal(mu, sigma, 1)
		x[0, 0] = x_0 # initial state
		x[0,1] = np.random.uniform(low=-2.0, high=2.0)
		
		y[0,0] = 5*np.cbrt(alpha_c*x[0,0])  + dt*x[0,1] + noise#np.random.normal(mu, sigma, 1)
		
		for i in range(1,T):
			#u[:,i-1] = - np.cbrt(x[:, i-1]) - alpha*(x[:,i-1]-xf)
			
			x[i,0] = y[i-1,0]
			x[i,1] = np.random.uniform(low=-0.5, high=0.5)
			# upper = 0.05
			# lower = -0.05
			noise  = np.random.normal(mu, sigma, 1)#stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
			if (x[i,0]>=-10) and (x[i,0]<=-8):
				x[i,0] += 5*np.random.normal(mu, sigma, 1)

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
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
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
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = (dof+y1.T@scipy.linalg.solve(Sigma11,y1)-2)/(dof+len(X1)-2)*(Sigma22 + sigma_noise - (solved @ Sigma12))
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
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
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
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel(X2, X2, scale)
		Sigma2 = (dof+y1.T@scipy.linalg.solve(Sigma11,y1)-2)/(dof+len(X1)-2)*(Sigma22 + sigma_noise - (solved @ Sigma12))
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


def plot_traj_unc_sets(X1_big,y1_big, x_traj,):


	fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
	ax = fig.add_subplot(111, projection='3d')

	coefs = (1, 2, 2)  # Coefficients in a0/c x**2 + a1/c y**2 + a2/c z**2 = 1 
	# Radii corresponding to the coefficients:
	rx, ry, rz = 1/np.sqrt(coefs)

	# Set of all spherical angles:
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)

	# Cartesian coordinates that correspond to the spherical angles:
	# (this is the equation of an ellipsoid):
	x = rx * np.outer(np.cos(u), np.sin(v))
	y = ry * np.outer(np.sin(u), np.sin(v))
	z = rz * np.outer(np.ones_like(u), np.cos(v))

	# Plot:
	ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b')

	# Adjustment of the axes, so that they all have the same span:
	max_radius = max(rx, ry, rz)
	for axis in 'xyz':
		getattr(ax, 'set_{}lim'.format(axis))((-max_radius, max_radius))

	plt.show()


def lin_dynamics(inter, curve_data):

	# return the linearized dynamics around the previous iteration points
	T  =len(inter)
	b1 = np.zeros(T)
	b2 = np.zeros(T)
	n = len(curve_data[0,:])
	max_val = np.max(curve_data[:,0])
	min_val = np.min(curve_data[:,0])
	for i in range(T):	
		ind = [j for j,v in enumerate(curve_data[:,0]) if v>inter[i]]
		if ind ==[]:
			# intercept nad slope is computed by averaging two adjacent points
			if inter[i] > max_val:
				b1[i]=(curve_data[-1,1]+curve_data[-2,1])/2
				b2[i]=(curve_data[-1,2]+curve_data[-2,2])/2
			elif inter[i] < min_val:
				b1[i]=(curve_data[0,1]+curve_data[1,1])/2
				b2[i]=(curve_data[0,2]+curve_data[1,2])/2
		else:
			ind = ind[0]
			b1[i]=(curve_data[ind,1]+curve_data[ind-1,1])/2
			b2[i]=(curve_data[ind,2]+curve_data[ind-1,2])/2
	return b1,b2


def lin_dynamics_local(inter, curve_data, l_x, l_y, F, h):

	# return the linearized dynamics around the previous iteration points
	T  = len(inter)
	b1 = np.zeros(T)
	b2 = np.zeros(T)
	x_min = np.zeros(T)
	x_max = np.zeros(T)
	max_val = np.max(l_x)
	min_val = np.min(l_x)
	for i in range(T):	
		ind = [j for j,v in enumerate(l_x) if v>=inter[i]]
		if ind ==[]:
			print("Check this, point out of domain")
			# if inter[i] > max_val:
			# 	b1[i]=(curve_data[-1,1]+curve_data[-2,1])/2
			# 	b2[i]=(curve_data[-1,2]+curve_data[-2,2])/2
			# elif inter[i] < min_val:
			# 	b1[i]=(curve_data[0,1]+curve_data[1,1])/2
			# 	b2[i]=(curve_data[0,2]+curve_data[1,2])/2
		else:
			ind = [j for j,v in enumerate(l_x) if v>=inter[i]][0]
			F_mid = F[ind]
			F_up = F_mid + h/2
			F_lo = F_mid - h/2
			if F_up > np.max(F):
				F_up = np.max(F) # for the points close to the right boundary of the domain	
			if F_lo <= np.min(F):
				F_lo = np.min(F) # for the points close to the left boundary of the domain	
			#IPython.embed()		
			ind_up = [j for j,v in enumerate(F) if v>=F_up][0]
			ind_lo = [j for j,v in enumerate(F) if v>=F_lo][0]
			slope = (l_y[ind_up]-l_y[ind_lo])/(l_x[ind_up]-l_x[ind_lo]) # slope of interpolation line
			intercept = l_y[ind_lo]-slope*l_x[ind_lo] # intercept of interpolation line
			x_min[i] = l_x[ind_lo]
			x_max[i] = l_x[ind_up]
			b1[i] = intercept
			b2[i] = slope
	return b1, b2, x_min, x_max# def interp_grid(F, x, N_interp, x_grid, y_grid):

def epsilon_stat(inter, l_x, low_ci, upp_ci, x_low, x_upp):

	# return vector of max statistical error in liearization regions
	n = len(inter)
	max_x = np.max(l_x)
	min_x = np.min(l_x)
	stat_err = np.zeros(len(inter))
	diff_ci = upp_ci-low_ci
	for i in range(n):
		x_lo = x_low[i]#inter[i] - hx[i]/2 # check not below above threshold
		x_up = x_upp[i]#inter[i] + hx[i]/2
		if x_lo < min_x:
			x_lo = x_min
		if x_up > max_x:
			x_up = max_x
		# or could return them from previous function
		ind_up = [j for j,v in enumerate(l_x) if v>=x_up][0]
		ind_lo = [j for j,v in enumerate(l_x) if v>=x_lo][0]
		stat_err[i] = np.max(diff_ci[ind_lo:ind_up+1])
	return stat_err

def epsilon_lin(inter, l_x, l_y, x_low, x_upp, b1, b2):

	# return vector if max interpolation error in liearization regions
	n = len(inter)
	max_x = np.max(l_x)
	min_x = np.min(l_x)
	lin_err = np.zeros(len(inter))
	for i in range(n):
		x_lo = x_low[i] # inter[i] - hx[i]/2# check not below above threshold
		x_up = x_upp[i] # inter[i] + hx[i]/2
		if x_lo < min_x:
			x_lo = x_min
		if x_up > max_x:
			x_up = max_x
		ind_up = [j for j, v in enumerate(l_x) if v >= x_up][0]
		ind_lo = [j for j, v in enumerate(l_x) if v >= x_lo][0]
		lin_x = np.linspace(l_x[ind_lo], l_x[ind_up], ind_up-ind_lo+1) # evaluate the function of a grid and get largest interpolation error
		# find index of max absolute difference between \hat{f}(x) and \hat{f}_{lin}(x) 
		ind_max = np.argmax(np.abs(l_y[ind_lo:ind_up+1]-b1[i]-b2[i]*lin_x))
		#lin_err[i] = np.max(np.abs(np.cbrt(l_x[ind_lo:ind_up+1])-l_y[ind_lo:ind_up+1]))
		temp = l_y[ind_lo:ind_up+1] - b1[i] - b2[i]*lin_x
		lin_err[i] = temp[ind_max] # return signed error
	return(lin_err)



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
	
def minkowski_diff(x_temp_up, x_temp_low, x_temp_1, x_temp_2):

	# minkowski difference function
	# minksowski difference between two intervals [x_temp_low,x_temp_up] and 
	# [x_temp_1, x_temp_2]
	vertex_1 = x_temp_up - x_temp_1
	vertex_2 = x_temp_up - x_temp_2
	vertex_3 = x_temp_low - x_temp_1
	vertex_4 = x_temp_low - x_temp_2
	x_temp_up =  np.min((vertex_1, vertex_2)) 
	x_temp_low =  np.max((vertex_3, vertex_4))
	return x_temp_up, x_temp_low	


def robust_eps(inter, l_x, l_y, T, stat_err, lin_err, b1, b2, upper_noise, lower_noise):
	# robust reachable sets 
	x_epsilon_up = np.zeros(len(inter)+1)
	x_epsilon_low = np.zeros(len(inter)+1)
	x_temp_up = 0
	x_temp_low = 0
	for i in range(0, len(inter)):
		# was
		#w_temp_up =  np.abs(lin_err[i]) + stat_err[i]/2 #+ 0.05#<-- very worst case # initialize upper and lower bound of Epsilon sets
		#w_temp_low =  - np.abs(lin_err[i]) - stat_err[i]/2 #- 0.05# <- no inter[i] + needed ?
		if lin_err[i]>0:
			w_temp_up =  np.abs(lin_err[i]) + stat_err[i]/2 + upper_noise#+ 0.02 #+ 0.05#<-- very worst case # initialize upper and lower bound of Epsilon sets
			w_temp_low =  - stat_err[i]/2 + lower_noise#- 0.02#- 0.05# <- no inter[i] + needed ?
		else:
			w_temp_up =  stat_err[i]/2 + upper_noise#+ 0.02 #+ 0.05#<-- very worst case # initialize upper and lower bound of Epsilon sets
			w_temp_low =  - np.abs(lin_err[i]) - stat_err[i]/2 + lower_noise#- 0.02#- 0.05# <- no inter[i] + needed ?
		#IPython.embed()
		# for j in range(i,T): # check indices
		#if(lin_err[j])>0: # max for now for lin but should be signed		
		x_temp_1 = b2[i]*x_epsilon_up[i]
		x_temp_2 = b2[i]*x_epsilon_low[i]
		x_temp_up, x_temp_low = minkowski_sum(x_temp_1, x_temp_2, w_temp_up, w_temp_low) # check order, was x_temp_up, x_temp_low, shrink?
		#IPython.embed()
		x_epsilon_up[i+1] = x_temp_up
		x_epsilon_low[i+1] = x_temp_low
	return x_epsilon_up, x_epsilon_low

				




def LR_MPC(x_upptemp, x_lowtemp, b1, b2, x_low, x_upp, x_epsilon_up, x_epsilon_low, x_max, x_min, T_mpc, x_0, j, xf, slope, intercept, flag):
	feas_flag = True
	root_cnst = 1.0
	Q = 1 # state cost
	R = 100 # control cost
	dt = 1.0 # time step
	#flag = 3
	u_thresh = 140.0

	x = Variable((1, T_mpc+1)) # state variable
	u = Variable((1, T_mpc)) # control variable
	#d = Variable((1, T_mpc+1)) # control invariant set
	cost = 0
	constr = []
	constr += [x[:,0] == x_0]
	for t in range(T_mpc):
		cost += Q*norm(x[:,t] - xf)**2 + R*norm(u[:,t])**2#Q*norm(x[:,t]-d[:,t])**2 +R*norm(u[:,t]-(-1/dt*np.cbrt(root_cnst*xf)-1/(dt*3)*np.cbrt(root_cnst)*(np.cbrt(xf))**(-2)*(d[:,t]-xf)+xf))**2 
		# nonlinear cases
		if flag <= 2:
			constr += [x[:,t+1] == b1[t]+b2[t]*x[:,t] + dt*u[:,t],
					   norm_inf(u[:,t]) <= u_thresh, norm_inf(x[:,t]) <= np.max(x_max)]
					  # d[:,t]<=x_upptemp, d[:,t]>=x_lowtemp]
		# just linear case
		else:	
			constr += [x[:,t+1] == intercept+slope*x[:,t] + dt*u[:,t],
					   norm_inf(u[:,t]) <= u_thresh, norm_inf(x[:,t]) <= np.max(x_max)]
					   #d[:,t]<=x_upptemp, d[:,t]>=x_lowtemp]
		if flag ==1 :
			if t>=1:
				diff = minkowski_diff(x_upp[t], x_low[t], x_epsilon_up[t], x_epsilon_low[t])
				# if set empty
				if diff[1] > diff[0]:
					print('------------ empty set ------------')
					#infeas[ii] = 1
					print('At MPC iteration {} the problem was infeasible.'.format(j))
					print('------------ empty set ------------')
					feas_flag = False
					#IPython.embed()
				constr += [x[:,t][0]<=diff[0], x[:,t][0]>=diff[1]]

	
	# Last constraint at time T	????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
	if flag <= 2:
		diff = minkowski_diff(x_upptemp, x_lowtemp, x_epsilon_up[T_mpc], x_epsilon_low[T_mpc])
		constr += [x[:,T_mpc][0]<=diff[0], x[:,T_mpc][0]>=diff[1]]#, d[:,T_mpc]<=x_upptemp, d[:,T_mpc]>=x_lowtemp] #x[:,T] <= 10**(-5),x[:,T] >= -10**(-5)
	else:
		#constr += [d[:,T_mpc] <= x_upptemp, d[:,T_mpc] >= x_lowtemp] 
		constr += [x[:,T_mpc][0]<=x_upptemp, x[:,T_mpc][0]>=x_lowtemp]
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
	return x.value[0], u.value[0], feas_flag