import deepxde as dde
import numpy as np
#from deepxde.backend import paddle as bkd

# Parameters
v = 0.5  # Advection velocity
alpha = 0.01  # Thermal diffusivity
f = 1 # Frequency
diff = 0.01
#dde.backend.set_default_backend("paddle")
# Define PDE
def pde1(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)  # ∂u/∂t
    du_x = dde.grad.jacobian(u, x, i=0, j=0)  # ∂u/∂x
    du_xx = dde.grad.hessian(u, x, i=0, j=0)  # ∂²u/∂x²
    return du_t + (v+diff) * du_x - alpha * du_xx # Differential equation (advection differential equation)

def pde2(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)  # ∂u/∂t
    du_x = dde.grad.jacobian(u, x, i=0, j=0)  # ∂u/∂x
    du_xx = dde.grad.hessian(u, x, i=0, j=0)  # ∂²u/∂x²
    return du_t + v * du_x - alpha * du_xx # Differential equation (advection differential equation)

# Time-varying boundary condition
def boundary_L(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_R(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

def bc_func_L1(x):
    return np.sin(2 * np.pi * f * x[:, 1:2])  # u(0, t) = sin(2πt)

def bc_func_L2(x):
    return np.sin(2 * np.pi * f * x[:, 1:2])

def bc_func_R(x):
    return np.zeros((len(x), 1))  # u(1, t) = 0

Lx=1 # function domain
T=2

# Geometry and time domain
geom = dde.geometry.Interval(0, Lx)
timedomain = dde.geometry.TimeDomain(0, T)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)


for i in [4]:
    f=i/10

    # Define BCs and ICs
    bc_L1 = dde.icbc.DirichletBC(geomtime, bc_func_L1, boundary_L)
    bc_L2 = dde.icbc.DirichletBC(geomtime, bc_func_L2, boundary_L)
    bc_R = dde.icbc.DirichletBC(geomtime, bc_func_R, boundary_R)
    ic = dde.icbc.IC(geomtime, lambda x: 0, lambda _, on_initial: on_initial)

    Nt=160 # Number of gridpoints in output
    Nx=100

    # Compile data
    data1 = dde.data.TimePDE(
        geomtime,
        pde1,
        [bc_L1, bc_R, ic],
        num_domain=1000,
        num_boundary=Nx,
        num_initial=Nt,
    )
    data2 = dde.data.TimePDE(
        geomtime,
        pde2,
        [bc_L2, bc_R, ic],
        num_domain=1000,
        num_boundary=Nx,
        num_initial=Nt,
    )

    # Build and train the model (replace with DeepONet architecture)
    net1 = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal") #fully connected neural network
    net2 = dde.nn.FNN([2] + [64] * 4 + [1], "tanh", "Glorot normal")
    model1 = dde.Model(data1, net1)
    model1.compile("adam", lr=1e-3)
    losshistory1,train_state1=model1.train(iterations=2000)
    dde.saveplot(losshistory1, train_state1, issave=True, isplot=False,output_dir='output_dir1')
    model2 = dde.Model(data2, net2)
    model2.compile("adam", lr=1e-3)
    losshistory2,train_state2=model2.train(iterations=2000)
    dde.saveplot(losshistory2, train_state2, issave=True, isplot=False,output_dir='output_dir2')

    # Create test points
    x = np.linspace(0, Lx, Nx)
    t = np.linspace(0, T, Nt)
    X, T = np.meshgrid(x, t)
    xt = np.vstack((X.flatten(), T.flatten())).T
    u_pred1 = model1.predict(xt)
    u_pred1 = u_pred1.reshape(X.shape)
    u_pred2 = model2.predict(xt)
    u_pred2 = u_pred2.reshape(X.shape)
    u_pred3 = (u_pred1 - u_pred2)/diff

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(X, T, u_pred3, shading="auto")
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("Advection-Diffusion Solution")
    plt.savefig('advection-diffusion-velocity'+str(i)+'.png')

    # 1st step: plot 3d graph using matlab plot3, draw comparison figure
    # 2nd step: try different architectures like DeepONet and PI-DeepONet
    # 3rd step: try different parameters for v, alpha, f, etc. Sensitivity analysis (variation delT/delF, etc.)
    # 4th step: automatic differentiation (???)