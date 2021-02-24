import numpy as np
import matplotlib.pyplot as plt
import IPython
import numpy as np
import matplotlib.pylab as plt
from scipy import spatial
import scipy
from scipy.stats import t


# def exponential_cov(x, y, params):
#     return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

# def predict(x, data, kernel, params, sigma, t):
#     k = [kernel(x, y, params) for y in data]
#     Sinv = np.linalg.inv(sigma)
#     y_pred = np.dot(k, Sinv).dot(t)
#     sigma_new = kernel(x, x, params) - np.dot(k, Sinv).dot(k)
#     return y_pred, sigma_new
 
 
# theta = [1, 10]
# sigma_0 = exponential_cov(0, 0, theta)
# xpts = np.arange(-3, 3, step=0.01)
# #plt.errorbar(xpts, np.zeros(len(xpts)), yerr=sigma_0, capsize=0)

# x_old = np.random.uniform(0,10,100)
# x_old = np.sort(x_old)
# x_old = x_old[::-1]
# y_old = np.zeros(len(x_old))
# for i in range(len(x_old)):
#     if x_old[i] <= 3:
#         y_old[i] = np.sin(x_old[i])
#     elif x_old[i] > 3 and x_old[i] <= 6:
#         y_old[i] = 0.01 * np.random.randn()
#     else:
#         y_old[i] = np.sin(x_old[i])

# sigma_noise = 0.1
# x_new = np.random.uniform(0,10,50)
# y_new = np.zeros(len(x_new))
# for i in range(len(x_new)):
#     if x_new[i] <= 3:
#         y_new[i] = np.sin(x_new[i]) + sigma_noise * np.random.randn()
#     elif x_new[i] > 3 and x_new[i] <= 6:
#         y_new[i] = 0.01  + sigma_noise * np.random.randn()
#     else:
#         y_new[i] = np.sin(x_new[i]) + sigma_noise * np.random.randn()




np.random.seed()

# x_pred = np.linspace(-3, 3, 1000)
# predictions = [predict(i, x, exponential_cov, theta, sigma_1, y) for i in x_pred]
# IPython.embed()
# y_pred, sigmas = np.transpose(predictions)
# plt.errorbar(x_pred, y_pred, yerr=sigmas, capsize=0)
# plt.plot(x, y, "ro")

# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb, scale):
	"""Exponentiated quadratic  with σ=1"""
	# L2 distance (Squared Euclidian)
	sq_norm = -0.5 * spatial.distance.cdist(xa, xb, 'sqeuclidean')
	#IPython.embed()
	#ell = 10
	#sq_norm = -0.5 * np.exp(np.linalg.norm(xa-xb)/ell)
	return np.exp(sq_norm/(2 * scale))


def GP_noise(X1_big, y1_big, X2_big, kernel_func, sigma_noise, scale):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	#X2 = X2_big
	k = 8#len(X1_big)
	mus = np.zeros(len(X2_big))
	Sigmas = np.zeros(len(X2_big))
	ind = 0
	for X2 in X2_big:
		dists = np.linalg.norm(X2-X1_big, axis=1)
		inds = np.argpartition(dists, k-1)
		inds = inds[0:k]
		X1 = X1_big[inds]
		y1 = y1_big[inds]
		Sigma11 = kernel_func(X1, X1, scale) + sigma_noise * np.eye(k)
		# Kernel of observations vs to-predict
		Sigma12 = kernel_func(X1, np.expand_dims(X2,1), scale)
		# Solve
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel_func(np.expand_dims(X2,1), np.expand_dims(X2,1), scale)
		Sigma2 = Sigma22 + sigma_noise  - (solved @ Sigma12)
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas  # mean, covariance


def Student_noise(X1_big, y1_big, X2_big, kernel_func, sigma_noise, scale, dof):
	"""
	Calculate the posterior mean and covariance matrix for y2
	based on the corresponding input X2, the noisy observations 
	(y1, X1), and the prior kernel function.
	"""
	# Kernel of the noisy observations
	k = 8#len(X1_big)
	mus = np.zeros(len(X2_big))
	Sigmas = np.zeros(len(X2_big))
	ind = 0
	for X2 in X2_big:
		dists = np.linalg.norm(X2-X1_big, axis=1)
		inds = np.argpartition(dists, k-1)
		inds = inds[0:k]
		X1 = X1_big[inds]
		y1 = y1_big[inds]
		Ker_temp = kernel_func(X1, X1, scale)# check whtehr these tw oshould break
		Sigma11 = Ker_temp + sigma_noise * np.eye(k)
		# Kernel of observations vs to-predict
		Sigma12 = kernel_func(X1, np.expand_dims(X2,1), scale)
		# Solve
		solved = scipy.linalg.solve(Sigma11, Sigma12, assume_a='pos').T
		# Compute posterior mean
		mu2 = solved @ y1
		# Compute the posterior covariance
		Sigma22 = kernel_func(np.expand_dims(X2,1), np.expand_dims(X2,1), scale)
		Sigma2 = (dof+y1.T@scipy.linalg.solve(Sigma11,y1)-2)/(dof+len(X1)-2)*(Sigma22 + sigma_noise - (solved @ Sigma12))
		#IPython.embed()
		mus[ind] = mu2
		Sigmas[ind] = Sigma2
		ind += 1
	return mus, Sigmas  # mean, covariance





# Define the true function that we want to regress on

n1 = 40  # Number of points to condition on (training points)
n2 = 20  # Number of points in posterior (test points)
sigma_noise = 0.5  # The variance of the noise
dof = 5
crit_val = t.ppf(0.975, dof+n1)
# Sample observations (X1, y1) on the function
# X1 = np.random.uniform(1, 6, size=(n1,1))
# X1 = np.sort(X1)
# X1 = X1[::-1]
# y1 = np.zeros(len(X1))
# for i in range(len(X1)):
#     if X1[i] <= 3.14:
#         y1[i] = 3*np.sin(X1[i]) + 1*np.random.randn()
#     elif X1[i] > 3.14 and X1[i] <= 6:
#         y1[i] = 0.01 * np.random.randn()
#     else:
#         y1[i] = 3*np.sin(X1[i])



# # Predict points at uniform spacing to capture function
# X2 = np.linspace(1, 6, n2).reshape(-1,1)
# y2 = np.zeros(len(X2))
# for i in range(len(X2)):
#     if X2[i] <= 3.14:
#         y2[i] = 3 * np.sin(X2[i]) + 1 * np.random.randn()
#     elif X2[i] > 3.14 and X2[i] <= 6:
#         y2[i] = 0.01 * np.random.randn()
#     else:
#         y2[i] = 3*np.sin(X2[i])

#########################################################################
# Homoscedastic and outliers
# X1 = np.random.uniform(1, 6, size=(n1,1))
# X1 = np.sort(X1,0)

# y1 = np.zeros(len(X1))
# for i in range(len(X1)):
# 	if np.random.rand()<0.3:
# 		y1[i] =  5*sigma_noise *np.random.randn()
# 	else:
# 		y1[i] =  sigma_noise *np.random.randn() #+ 3 * np.sin(X1[i])
	
# # create gap
# # index = list(range(100,450))
# # X1 = np.delete(X1, index)
# # y1 = np.delete(y1, index)
# # X1 = np.expand_dims(X1,1)
# # y1 = np.expand_dims(y1,1)

# # Predict points at uniform spacing to capture function
# X2 = np.linspace(1, 6, n2).reshape(-1,1)
# X2_test = np.linspace(1, 6, 200).reshape(-1,1)# just plotting
# y2 = np.zeros(len(X2))
# for i in range(len(X2)):
# 	y2[i] =  sigma_noise  * np.random.randn() #+ 3* np.sin(X2[i])
   

# heteroscedastic
X1 = np.random.uniform(1, 6, size=(n1,1))
X1 = np.sort(X1,0)

y1 = np.zeros(len(X1))
for i in range(len(X1)):
	if X1[i]<2:
		y1[i] =  10*sigma_noise *np.random.randn()
	elif X1[i] > 2 and X1[i]<4:
		y1[i] =  sigma_noise *np.random.randn() #+ 3 * np.sin(X1[i])
	else:
		y1[i] =  5*sigma_noise *np.random.randn()
	
# create gap
# index = list(range(100,450))
# X1 = np.delete(X1, index)
# y1 = np.delete(y1, index)
# X1 = np.expand_dims(X1,1)
# y1 = np.expand_dims(y1,1)

# Predict points at uniform spacing to capture function
X2 = np.linspace(1, 6, n2).reshape(-1,1)
X2_test = np.linspace(1, 6, 200).reshape(-1,1)# just plotting
y2 = np.zeros(len(X2))
for i in range(len(X2)):
	if X2[i]<2:
		y2[i] =  10*sigma_noise *np.random.randn()
	elif X2[i] > 2 and X2[i]<4:
		y2[i] =  sigma_noise *np.random.randn() #+ 3 * np.sin(X1[i])
	else:
		y2[i] =  5*sigma_noise *np.random.randn()

#########################################################################
# sine
# y1 = np.zeros(len(X1))
# for i in range(len(X1)):
#     if X1[i] <=2:
#         y1[i] =  4*X1[i] + sigma_noise  * np.random.randn()
#     elif X1[i] >2  and X1[i] <= 4:
#         y1[i] = 16 - 4*X1[i] + sigma_noise  * np.random.randn()
#     else:
#     	y1[i] = -16 + 4*X1[i] + sigma_noise  * np.random.randn()

# # Predict points at uniform spacing to capture function
# X2_test = np.linspace(1, 6, 200).reshape(-1,1)
# X2 = np.linspace(1, 6, n2).reshape(-1,1)
# y2 = np.zeros(len(X2))
# for i in range(len(X2)):
#     if X2[i] <=2:
#         y2[i] =  4*X2[i] + sigma_noise  * np.random.randn()
#     elif X2[i] >2  and X2[i] <= 4:
#         y2[i] = 16 - 4*X2[i] + sigma_noise  * np.random.randn()
#     else:
#     	y2[i] = -16 + 4*X2[i] + sigma_noise  * np.random.randn()


#########################################################################   

#########################################################################
# zig zag
# y1 = np.zeros(len(X1))
# for i in range(len(X1)):
#     if X1[i] <=2:
#         y1[i] =  4*X1[i] + sigma_noise  * np.random.randn()
#     elif X1[i] >2  and X1[i] <= 4:
#         y1[i] = 16 - 4*X1[i] + sigma_noise  * np.random.randn()
#     else:
#     	y1[i] = -16 + 4*X1[i] + sigma_noise  * np.random.randn()

# # Predict points at uniform spacing to capture function
# X2_test = np.linspace(1, 6, 200).reshape(-1,1)
# X2 = np.linspace(1, 6, n2).reshape(-1,1)
# y2 = np.zeros(len(X2))
# for i in range(len(X2)):
#     if X2[i] <=2:
#         y2[i] =  4*X2[i] + sigma_noise  * np.random.randn()
#     elif X2[i] >2  and X2[i] <= 4:
#         y2[i] = 16 - 4*X2[i] + sigma_noise  * np.random.randn()
#     else:
#     	y2[i] = -16 + 4*X2[i] + sigma_noise  * np.random.randn()


#########################################################################   
# Gaussian process posterior with noisy obeservations


# Compute the posterior mean and covariance
# Add noise kernel to the samples we sampled previously
#y1 = y1 + sigma_noise * np.random.randn(n1)
scale = 0.01
# Compute posterior mean and covariance
mu2, Sigma2 = GP_noise(X1, y1, X2_test, exponentiated_quadratic, sigma_noise, scale)

# Compute the standard deviation at the test points to be plotted
sigma2 = np.sqrt(Sigma2)

# Draw some samples of the posterior
#y2dr = np.random.multivariate_normal(mean=mu2, cov=Sigma2, size=ny)
####################### STUDENT T##########################

mut, Sigmat = Student_noise(X1, y1, X2_test, exponentiated_quadratic, sigma_noise, scale, dof)
sigmat = np.sqrt(Sigmat)

####################################################

# Plot the postior distribution and some samples
fig, (ax1, ax2) = plt.subplots(
	nrows=1, ncols=2, figsize=(12,6))
# Plot the distribution of the function (mean, covariance)
ax1.scatter(X2, y2, s=12, marker='x',c='red', label='$f(x)$')
ax1.fill_between(X2_test.flat, mu2-1.96*sigma2, mu2+1.96*sigma2, color='grey', 
				 alpha=0.15)#, label='$2\sigma_{\hat y|y}$'
ax1.plot(X2_test, mu2, 'b-', lw=1.5, label='$\hat f$')
ax1.scatter(X1, y1, s=12,facecolors='none', edgecolors='b',label='$y$')
ax1.set_xlabel('$x$', fontsize=13)
#ax1.set_ylabel('$y$', fontsize=13)
ax1.set_title('GPR')
ax1.axis([1, 6, -10, 10])
#ax1.legend(loc=1)
# Plot some samples from this function
ax2.scatter(X2, y2, s=12,  marker='x',c='red', label='$f(x)$')
ax2.fill_between(X2_test.flat, mut-crit_val*sigmat, mut+crit_val*sigmat, color='grey', 
				 alpha=0.15)
ax2.plot(X2_test, mut, 'b-', lw=1.5, label='$\\hat f$')
ax2.scatter(X1, y1, s=12,facecolors='none', edgecolors='b',label='$y$')
ax2.set_xlabel('$x$', fontsize=13)
#ax2.set_ylabel('$y$', fontsize=13)
ax2.set_title('TPR')
ax2.axis([1, 6, -10, 10])
#x2.legend()
plt.tight_layout()
plt.show()