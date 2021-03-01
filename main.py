# Generate data for control problem.
import numpy as np
from cvxpy import *
import IPython
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
from LR_MPC import *
from LR_MPC_2 import *
import os

# Check if a storedData folder exist.	
if not os.path.exists('storedData'):
	os.makedirs('storedData')

#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#np.random.seed(3)
n = 2 # state size
m = 2 # input size
T = 10 # duration of simulation
dt = 1.0 # time step
x0 = 5.0 # starting region
xf = -1.0
v0 = 0.8 # initila speed
mu = 0.0 # noise mean and variance
sigma = 0.2
power = 1/3 
u_thresh = 140.0
root_cnst = 1.0
n_sim = 30 # number of trajectories to simulate, actual simulations are 
upper_noise_bnd = 0.05 # bounds on noise
lower_noise_bnd = -0.05
N_interp = 12 # interpolation points
gen_data = True
infeas_flag = False
method = 'random'



######################################
# gather data
######################################
x_0 = np.array([x0 + 0.5*np.random.rand()])[0]
if gen_data == True:
	
	if method == 'control':
		x_trajectories, y_trajectories = gather_trajectories(x0, xf, dt, n_sim, T, root_cnst, upper_noise_bnd, lower_noise_bnd, mu, sigma, method)
	else:
		x_l = []
		y_l = []
		for i in range(10):
			x_trajectories, y_trajectories = gather_trajectories(x0, xf, dt, n_sim, T, root_cnst, upper_noise_bnd, lower_noise_bnd, mu, sigma, method)
			x_l.append(x_trajectories)
			y_l.append(y_trajectories)
	
# 	np.savetxt('xdata.txt',x)
# 	np.savetxt('ydata.txt',y)
# x = np.loadtxt('xdata.txt')
# y = np.loadtxt('ydata.txt')

x_trajectories = np.concatenate( x_l, axis=0 )
y_trajectories = np.concatenate( y_l, axis=0 )


# y = GP(x_trajectories, y_trajectories, np.array([[0.1,0.2],[0.3,0.4]]), exp_kernel, 0.1, 0.2)
#plot_fit(x_trajectories, y_trajectories, sigma)
plot_traj_unc_sets()
IPython.embed()





### Estimation
# locally linear regression
# alpha neighborhood, poly_degree = 1 for linear regression 
# Obtain reference to LOESS x & y values (v & g).
l_x  = evalDF['v'].values # x actual
l_y  = evalDF['g'].values # y estimated
bb = evalDF['b'] # intercept and slope of fitted line (local on x)
ndata = len(l_x)
curve_data = np.zeros((ndata, 3))
curve_data[:,0]=l_x
curve_data[:,1]=[v[0] for i,v in enumerate(bb)] # save the intercept and slope
curve_data[:,2]=[v[1] for i,v in enumerate(bb)]
# already sorted vy loess
#curve_data = curve_data[np.argsort(curve_data[:, 0])]
#curve_data = sorted(curve_data, key=lambda a_entry: a_entry[0]) 
x_min = np.min(x)
x_max = np.max(x)
xp = np.linspace(x_min,x_max,len(l_x)) # domain
act_fun = np.cbrt(root_cnst*xp) # actual dynamics function to plot






######################################
# confidence intervals
######################################


######################################
# control
######################################

T_mpc=5 # mpc horizon
x = Variable((1, T_mpc+1))
u = Variable((1, T_mpc))
Q = 1 # state cost
R = 100 # control cost
cost = 0
constr = []
mpc_sim = 8 # mpc simulations
# initialize linearization points for first mpc iteration

x_mpc = np.zeros(mpc_sim+1) 
x_0 = np.random.uniform(-1, 5)
xf = np.random.uniform(-1, 5)
x_mpc[0] = x_0




######################################
Ntrials = 10
toler_list = np.zeros(Ntrials)
final_state_list = np.zeros(Ntrials)
cost_list = np.zeros(Ntrials)
cost2 = 0
toler = 1
for ii in range(Ntrials):
	toler = 1/(ii+1)
	#x_0_init = np.random.uniform(-5,5)
	x_0 = x_0_init
	print('Initial state {}'.format(x_0))
	xf = -1 #np.random.uniform(-1,5)
	# x_0 = 5.0 * np.random.randn()
	# xf = 5.0 * np.random.randn()
	#x_0 = np.array([x0 + 0.5*np.random.rand()])[0]


	# initialize linearization points for first mpc iteration
	inter = np.linspace(x_0+10**(-2), xf+10**(-2), num=T_mpc)
	#b1, b2 = lin_dynamics(inter, curve_data) # this uses the coefficients of the loess and not the interpolation
	b1, b2, x_low, x_upp = lin_dynamics_local(inter, curve_data, l_x, l_y, F, h)
	stat_err = epsilon_stat(inter, l_x, low_ci, upp_ci, x_low, x_upp) # vector of statistical errors in linearization regions
	lin_err = epsilon_lin(inter, l_x, l_y, x_low, x_upp, b1, b2) # vector of linearization errors in linearization regions
	x_epsilon_up, x_epsilon_low = robust_eps(inter, l_x, l_y, T_mpc, stat_err, lin_err, b1, b2, upper_noise_bnd, lower_noise_bnd) # robustify constraints

	one_step_pred_error = np.zeros(mpc_sim)
	lin_bound = 0.01 # constraint for x in mpc to lie withiin linearization accuracy (only first mpc interation)
	x_upptemp = xf + 0.1
	x_lowtemp = xf - 0.1
	flag = 1 # 1 nonlinear 2 nonlinear unconstrained 3 linear
	fail_times = 0
	for j in range(mpc_sim):	
		xval2, uval2, feas_flag2 = LR_MPC_2(x_upptemp, x_lowtemp, b1, b2, x_low, x_upp, x_epsilon_up, x_epsilon_low, x_max, x_min, T_mpc, x_0, j, xf, slope, intercept, flag, toler, inter)
		

		if feas_flag == True:
			# update new points of linear approximation
			inter = xval2
			inter_prev = inter
			inter = np.delete(inter, 0) # remove first element due to dynamics propagation
			# b1comp, b2comp = lin_dynamics(inter, curve_data) # linear dynamics from loess estimate
			b1, b2, x_low, x_upp = lin_dynamics_local(inter, curve_data, l_x, l_y, F, h) # linearization around inter coordinates	
			hx = x_upp - x_low # vector of domain discretization, not used for now	
			stat_err = epsilon_stat(inter, l_x, low_ci, upp_ci, x_low, x_upp) # vector of statistical errors in linearization regions
			lin_err = epsilon_lin(inter, l_x, l_y, x_low, x_upp, b1, b2) # vector of linearization errors in linearization regions
			x_epsilon_up, x_epsilon_low = robust_eps(inter, l_x, l_y, T_mpc, stat_err, lin_err, b1, b2, upper_noise_bnd, lower_noise_bnd) # robustify constraints
			x_0  =  np.cbrt(root_cnst*xval2[0])+ dt*uval2[0] 
			noise  = stats.truncnorm((lower_noise_bnd - mu) / sigma, (upper_noise_bnd - mu) / sigma, loc=mu, scale=sigma)
			x_0 = x_0 +  noise.rvs(1)[0]#np.random.normal(mu, sigma, 1)
			x_0 = x_0#[0] 
			x_mpc[j+1] = x_0# mpc trajectory
			T_mpc = T_mpc
		else:
			#IPython.embed()
			print('Infeasible')
			inter = inter_prev
			inter = np.delete(inter, 0) # remove first element due to dynamics propagation
			inter = np.delete(inter, 0)
			# b1comp, b2comp = lin_dynamics(inter, curve_data) # linear dynamics from loess estimate
			b1, b2, x_low, x_upp = lin_dynamics_local(inter, curve_data, l_x, l_y, F, h) # linearization around inter coordinates	
			hx = x_upp - x_low # vector of domain discretization, not used for now	
			stat_err = epsilon_stat(inter, l_x, low_ci, upp_ci, x_low, x_upp) # vector of statistical errors in linearization regions
			lin_err = epsilon_lin(inter, l_x, l_y, x_low, x_upp, b1, b2) # vector of linearization errors in linearization regions
			x_epsilon_up, x_epsilon_low = robust_eps(inter, l_x, l_y, T_mpc, stat_err, lin_err, b1, b2, upper_noise_bnd, lower_noise_bnd) # robustify constraints
			T_mpc = T_mpc - 1
			fail_times += 1
		# print(xval2)
		# print(uval2)
		#######################
		cost2+=Q*xval2[0]**2+R*uval2[0]**2
	toler_list[ii] = toler
	cost_list[ii] = cost2
	final_state_list[ii] = x_0
	#target_state[ii] = xf
	#actual_state[ii] = x_0
xfinal_simple_algo = x_0
print('******************************')		
print('Original algorithm actual end state {}'.format(xfinal_original_algo))
print('******************************')
print('Simple algorithm actual end state {}'.format(xfinal_simple_algo))
print('********************************')
print('Original algorithm cost {}'.format(cost1))
print('********************************')
print('Simple algorithm cost {}'.format(cost2))
print('***************END**************')
######################################
# plots
######################################

np.savetxt('storedData/toler_list.txt',toler_list)
np.savetxt('storedData/cost_list.txt',cost_list)
np.savetxt('storedData/final_state_list.txt',final_state_list)


IPython.embed()

f = plt.figure(2)
ax = f.add_subplot(411)
# for i in range(n_sim):
# 	plt.plot(x_trajectories[i,:])
# plt.ylabel(r"location", fontsize=16)
###
plt.subplot(4,1,2)
plt.plot(x_mpc)
plt.ylabel(r"mpc traj", fontsize=16)
plt.xlabel('iter', fontsize=16)
###
plt.subplot(4,1,3)
plt.plot(uval)
plt.ylabel(r"input", fontsize=16)
plt.xlabel('iter', fontsize=16)
###
plt.subplot(4,1,4)
plt.plot(one_step_pred_error)
plt.ylabel(r"one step err", fontsize=16)
plt.xlabel('iter', fontsize=16)
#plt.show()















