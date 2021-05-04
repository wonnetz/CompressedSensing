import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as la
from sklearn import preprocessing

# Experiment Values
n = 1000
m = 100
p = 10
laambda = 2

# My Own Test Samples
#n = 20
#m = 15
#p = 2


# Functions


#def clip(vector, tau):
#    clipped = np.minimum(vector, tau)
#    clipped = np.maximum(clipped, -tau)
#    return clipped


# Proximal Operator
#def proximal(vector, tau):
#    x_k = vector - clip(vector,tau) # Possibly change the name for the x_k vector later
#    return x_k

def proximal(z, tau1):
    x = np.zeros(len(z))
    for i in range(len(z)):
        if z[i] > tau1:
            x[i] = z[i] - tau1
        elif z[i] < (-tau1):
            x[i] = z[i] + tau1
        else:
            x[i] = 0
    return x


# Evaluates the Gradient
def gradient(A, x, b):
    Axminusb = A.dot(x) - b
    A_T = np.transpose(A)
    grad = A_T.dot(Axminusb)
    return laambda * grad

"""
# I decided to use tau in place of 1/(theta_k * L)
def accel(y, z, tau2):
    x = np.zeros(len(z))
    gradY = gradient(A, y, b)
    for i in range(len(z)):
        if z[i] > (1 + gradY[i])/tau2:
            x[i] = z[i] - ((1 + gradY[i])/tau2)
        elif z[i] < -((1 + gradY[i])/tau2):
            x[i] = z[i] + ((1 - gradY[i])/tau2)
        else:
            x[i] = 0
    return x
"""


def accel(y, z, alpha):
    grad = gradient(A, y, b)
    for j in range(len(z)):
        if z[j] > ((1 + grad[j])/alpha):
            z[j] = z[j] -  ((1 + grad[j])/alpha)
        elif z[j] < -((1 - grad[j])/alpha):
            z[j] = z[j] + ((1 - grad[j])/alpha)
        else:
            z[j] = 0
    return z


# Evaluates the Function Value
def function(A, x_k, b):
    f = (np.linalg.norm(((A.dot(x_k)) - b))) ** 2
    f = (laambda/2) * f
    f = f + np.linalg.norm(x_k, ord=1)
    return f


# Initialization of an x_hat vector with p random elements in p random spots.
# This is our solution vector
x_hat = np.zeros(n)
for i in range(p):
    index = np.random.randint(1, len(x_hat))
    x_hat[index] = np.random.rand()

# Matrix A and A_T with IID Entries from the Standard Gaussian Distribution, i.e Normal Distribution
A = np.random.default_rng().standard_normal(size=(m, n))
# Normalizes A such that it's columns has l2_norm equal to 1
A = preprocessing.normalize(A, axis=0)
A_T = np.transpose(A)
# Creation of the b vector
b = A.dot(x_hat)


# The Actual Optimization Part
#x_0 = np.random.normal(0, 1, n)  # Random guess for x
x_0 = np.zeros(n)
eigs = la.eigs(A_T.dot(A), which='LR')  # Computation of the Eigen Values of A_T * A
eig_L = eigs[0][0].real  # Largest Eigen Value
L = laambda * eig_L  # Creation of the Lipschitz Constant
tau = np.random.uniform(0, (2/L))  # Creation of our Tau
x_k = x_0

prox_error = [(np.linalg.norm((x_0 - x_hat))/np.linalg.norm(x_hat))]
prox_func = [(np.abs((function(A, x_0, b) - function(A, x_hat, b))))]

# Accelerated Proximal Gradient Method
#theta_0 = np.random.uniform(0, 1)
theta_0 = (1 / L)
theta_k = theta_0
#z_0 = np.random.normal(0, 1, n)  # Random guess for z
z_0 = np.zeros(n)
x_k = x_0
z_k = z_0


accel_err = [(np.linalg.norm((x_0 - x_hat))/np.linalg.norm(x_hat))]
accel_func = [np.abs(function(A, x_0, b) - function(A, x_hat, b))]

for i in range(500):
    y_k = (1 - theta_k) * x_k + theta_k * z_k
    tau = theta_k * L
    z_k = accel(y_k, z_k, tau)
    x_k = (1 - theta_k) * x_k + theta_k * z_k

    theta_k = (np.sqrt(((theta_k ** 4) + (4 * theta_k ** 2))) - (theta_k ** 2)) / 2
    accel_err.append(np.linalg.norm(x_k - x_hat)/np.linalg.norm(x_hat))

x_k = x_0
# Proximal Gradient Method
for i in range(500):
    grad = gradient(A, x_k, b)  # This is correct
    input = x_k - (tau * grad)  # This should be fine
    x_k = proximal(input, tau)  #

    prox_error.append(np.linalg.norm(x_k - x_hat)/np.linalg.norm(x_hat))
    prox_func.append(np.abs(function(A, x_k, b) - function(A, x_hat, b)))
    #prox_func.append(np.abs(np.linalg.norm(x_k, ord=0) - np.linalg.norm(x_hat, ord=0)))




"""
# Accelerated Loop
for i in range(500):
    y_k = ((1 - theta_k) * x_k) + (theta_k * z_k)
    z_k = accel(y_k, z_k, (theta_k * L))
    x_k = ((1 - theta_k) * x_k) + (theta_k * z_k)
    theta_k = (np.sqrt(((theta_k**4) + (4*theta_k**2))) - (theta_k**2)) / 2  # This is correct

    accel_err.append(np.linalg.norm(x_k - x_hat)/np.linalg.norm(x_hat))
    accel_func.append(np.abs(function(A, x_k, b) - function(A, x_hat, b)))
"""



plt.plot(range(501), prox_error, label="Proximal Error")
plt.plot(range(501), accel_err, label="Accelerated Error")
plt.legend()
plt.show()

plt.plot(range(501), prox_func, label="Proximal Function Value Difference")
plt.plot(range(501), accel_func, label="Accelerated Function Value Difference")
plt.legend()
plt.show()





