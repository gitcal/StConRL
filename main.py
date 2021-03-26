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
xf = -1.0
v0 = 0.8 # initila speed
mu = 0.0 # noise mean and variance
sigma = 0.1
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
x0 = 8
scale = 10

Lip_sigma = 1
Lip_grad_mu = 1
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
	

x_trajectories = np.concatenate( x_l, axis=0 )
y_trajectories = np.concatenate( y_l, axis=0 )



######################################
# control
######################################

T_mpc=4 # mpc horizon
x = Variable((1, T_mpc+1))
u = Variable((1, T_mpc))
Q = 1 # state cost
R = 100 # control cost
cost = 0
constr = []
mpc_sim = 8 # mpc simulations
# initialize linearization points for first mpc iteration

x_mpc = np.zeros(mpc_sim+1) 
x_0 = 10
x_mpc[0] = x_0


	


print('Initial state {}'.format(x_0))

max_x_lin = 0.2
xf = -10 #np.random.uniform(-1,5)



# initialize linearization points for first mpc iteration
init_traj = np.linspace(x_0+10**(-2), xf+10**(-2), num=T_mpc)
mus, J_x, J_u = get_Jacobian(x_trajectories, y_trajectories, init_traj)


low_end = []
high_end = []
for j in range(mpc_sim):	
	x_max = init_traj + max_x_lin
	x_min = init_traj - max_x_lin
	u_Jac = np.zeros(len(init_traj))
	X2_big = np.stack((init_traj, u_Jac), axis=-1)# check inti traje and _0 oinde
	#mus, sigmas = STP(x_trajectories, y_trajectories, X2_big, exp_kernel, sigma, scale)
	mus, sigmas = STP_local(x_trajectories, y_trajectories, X2_big, 10, exp_kernel, sigma, scale)
	x_opt, u_opt = STP_MPC(mus, J_x, J_u, x_max, x_min, init_traj, T_mpc, x_0, xf)
	x_Jax_temp = x_opt[0]
	u_Jac_temp = u_opt[0]
	X2_big_temp = np.stack((x_Jax_temp, u_Jac_temp), axis=-1)
	X2_big_temp = np.expand_dims(X2_big_temp, 0)
	
	#mus_temp, sigmas_temp = STP(x_trajectories, y_trajectories, X2_big_temp, exp_kernel, sigma, scale)
	mus_temp, sigmas_temp = STP_local(x_trajectories, y_trajectories, X2_big_temp, 10, exp_kernel, sigma, scale)
	# IPython.embed()
	unc_sets_up, unc_sets_low = get_traj_unc_sets_tp1(init_traj, mus_temp, sigmas_temp, max_x_lin, Lip_sigma, Lip_grad_mu)

	low_end.append(unc_sets_low)
	high_end.append(unc_sets_up)
	init_traj = x_opt[1:]
	mus, J_x, J_u = get_Jacobian(x_trajectories, y_trajectories, init_traj)
	noise = np.random.normal(mu, sigma, 1)

	if (x_opt[0]>=-2) and (x_opt[0]<=2):
		noise = 5*np.random.normal(mu, sigma, 1)
	x_0  =  5*np.cbrt(x_opt[0]) + u_opt[0] 

	
	x_0 = x_0 + noise#np.random.normal(mu, sigma, 1)
	x_mpc[j+1] = x_0# mpc trajectory
	# IPython.embed()

low_end = np.array(low_end)
high_end = np.array(high_end)
print("Low end is {}".format(low_end.T))
print("High end is {}".format(high_end.T))
print("Real trajectory is {}".format(x_mpc[1:].T))








