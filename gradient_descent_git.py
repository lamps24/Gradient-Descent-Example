import numpy as np

np.random.seed(59802020)

n = 50
m = 200
r = 10

# generate a random dataset
A = np.random.randn(n, r)
B = np.random.randn(r, m)
X = np.matmul(A, B) + 0.01*np.random.randn(n, m)

# zero center
X -= np.mean(X, axis=1).reshape(n, 1)


########################################
### 1. Principal Components Analysis ###
########################################

# compute X*t(X)
Xt = np.transpose(X)
XXt = np.matmul(X, Xt)

# compute eigenvalues of X*t(X)
w,v = np.linalg.eig(XXt)

# first ten vectors of v are the first 10 principal components
A1 =  v[:,0:10]


#################################################
### 2. Gradient Descent to derive Autoencoder ###
#################################################

# initialize settings
a = np.random.randn(n, r)
rho = 0.95
c = 1e-7
epsilon = 10e6
iter = 1
at = np.transpose(a)
grad = -2*(((X - (a @ at @ X)) @ Xt @ a) + (X @ (Xt - (Xt @ a @ at)) @ a))

# while SC not satisfied, do
while epsilon > 1e-5:
     
    # back-tracking line search for t
    inner = np.trace((np.transpose(grad) @ -grad),)
    t = 1
    fnew = np.linalg.norm(X - ((a + t*-grad) @ np.transpose(a + t*-grad) @ X))
    f = np.linalg.norm(X - (a @ at @ X))
    diff = fnew - f
    comp = c * t * inner 

    # choose step size
    while diff > comp:
        t = rho*t
        fnew = np.linalg.norm(X - ((a + t*-grad) @ np.transpose(a + t*-grad) @ X))
        diff = fnew - f
        comp = c * t * inner 
        
    # take a step
    a = a - (t * grad)
    at = np.transpose(a)
    
    # updated gradient
    grad = -2*(((X - (a @ at @ X)) @ Xt @ a) + (X @ (Xt - (Xt @ a @ at)) @ a))

    # check epsilon
    epsilon = np.linalg.norm(grad)
    
    # update count
    iter = iter + 1


# difference metric between PCA and Auotencoder
A2 = a

A1pinvA1 = A1 @ np.linalg.pinv(A1)
A2pinvA2 = A2 @ np.linalg.pinv(A2)
result = np.linalg.norm(A1pinvA1 - A2pinvA2)

# they come to a very similar result
print(result)


####################################################
### 3. Gradient Descent to derive Factorization  ###
####################################################

### (ii) ###
# initialize settings
a = np.random.randn(n, r)
z = np.random.randn(r, m)
rho = 0.95
c = 1e-7
epsilon = 10e6
iter = 1
at = np.transpose(a)
zt = np.transpose(z)
grad_a = 2*((a @ z - X) @ zt)
grad_z = 2*(at @ (a @ z - X))

# while SC not satisfied, do
while epsilon > 1e3:
     
    # back-tracking line search for t_a
    inner_a = np.trace((np.transpose(grad_a) @ -grad_a),)
    t_a = 1
    fnew = np.linalg.norm(X - ((a + t_a*-grad_a) @ z))
    f = np.linalg.norm(X - (a @ z))
    diff = fnew - f
    comp = c * t_a * inner_a 

    # choose step size for t_a
    while diff > comp:
        t_a = rho*t_a
        fnew = np.linalg.norm(X - ((a + t_a*-grad_a) @ z))
        diff = fnew - f
        comp = c * t_a * inner_a 
        
    # back-tracking line search for t_z
    inner_z = np.trace((np.transpose(grad_z) @ -grad_z),)
    t_z = 1
    fnew = np.linalg.norm(X - a @ (z + t_z*-grad_z))
    f = np.linalg.norm(X - (a @ z))
    diff = fnew - f
    comp = c * t_z * inner_z 

    # choose step size for t_z
    while diff > comp:
        t_z = rho*t_z
        fnew = np.linalg.norm(X - a @ (z + t_z*-grad_z))
        diff = fnew - f
        comp = c * t_z * inner_z 
        
    # take a step
    a = a - (t_a * grad_a)
    at = np.transpose(a)
    z = z - (t_z * grad_z)
    zt = np.transpose(z)
    
    # updated gradients
    grad_a = 2*((a @ z - X) @ zt)
    grad_z = 2*(at @ (a @ z - X))

    # check epsilon
    epsilon = (np.linalg.norm(grad_a) + np.linalg.norm(grad_z)) / 2
    
    # update count
    iter = iter + 1


# differences between PCA, Autoencoder, and Factorization
A3 = a

A3pinvA3 = A3 @ np.linalg.pinv(A3)

# they all come to similar results
print(np.linalg.norm(A1pinvA1 - A2pinvA2))
print(np.linalg.norm(A2pinvA2 - A3pinvA3))
print(np.linalg.norm(A1pinvA1 - A3pinvA3))




