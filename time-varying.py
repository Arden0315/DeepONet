import deepxde as dde
import numpy as np
#from deepxde.backend import paddle as bkd

# Parameters
v = 0.5  # Advection velocity
alpha = 0.01  # Thermal diffusivity
f = 1 # Frequency
#dde.backend.set_default_backend("paddle")
# Define PDE
def pde(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)  # ∂u/∂t
    du_x = dde.grad.jacobian(u, x, i=0, j=0)  # ∂u/∂x
    du_xx = dde.grad.hessian(u, x, i=0, j=0)  # ∂²u/∂x²
    return du_t + v * du_x - alpha * du_xx # Differential equation (advection differential equation)

# Time-varying boundary condition
def boundary_L(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_R(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

def bc_func_L(x):
    return np.sin(2 * np.pi * f * x[:, 1:2])  # u(0, t) = sin(2πt)

def bc_func_R(x):
    return np.zeros((len(x), 1))  # u(1, t) = 0

Lx=1 # function domain
T=2

# Geometry and time domain
geom = dde.geometry.Interval(0, Lx)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define BCs and ICs
bc_L = dde.icbc.DirichletBC(geomtime, bc_func_L, boundary_L)
bc_R = dde.icbc.DirichletBC(geomtime, bc_func_R, boundary_R)
ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

Nt=160 # Number of gridpoints in output
Nx=100

# Compile data
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc_L, bc_R, ic],
    num_domain=1000,
    num_boundary=Nx,
    num_initial=Nt,
)

# Build and train the model (replace with DeepONet architecture)
net = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal") #fully connected neural network
model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
losshistory,train_state=model.train(iterations=2000)
dde.saveplot(losshistory, train_state, issave=True, isplot=False,output_dir='output_dir')


# Create test points
x = np.linspace(0, Lx, Nx)
t = np.linspace(0, T, Nt)
X, T = np.meshgrid(x, t)
xt = np.vstack((X.flatten(), T.flatten())).T

np.savetxt("output_dir/xt.txt",xt)
np.savetxt("x.txt", X)
np.savetxt("t.txt", T)
# Predict and plot
u_pred = model.predict(xt)
np.savetxt("output_dir/u_pred.txt",u_pred)
u_pred = u_pred.reshape(X.shape)
np.savetxt("output_dir/prediction.txt",u_pred)

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.pcolormesh(X, T, u_pred, shading="auto")
plt.colorbar(label="u(x,t)")
plt.xlabel("x")
plt.ylabel("t")
plt.title("Advection-Diffusion Solution")
plt.show()
#plt.savefig('advection-diffusion.png')

fig = plt.figure().add_subplot(projection='3d')
fig.scatter(X, T, u_pred, c=u_pred)
coordinates = np.loadtxt('output_dir/test.dat')
X_test = coordinates[:, 0]
T_test = coordinates[:, 1]
u_test = coordinates[:, 2]
fig.scatter(X_test, T_test, u_test, c='black')
plt.show()

# 1st step: plot 3d graph using matlab plot3, draw comparison figure
# 2nd step: try different architectures like DeepONet and PI-DeepONet
# 3rd step: try different parameters for v, alpha, f, etc. Sensitivity analysis (variation delT/delF, etc.)
# 4th step: automatic differentiation (???)