"""Backend supported: tensorflow.compat.v1, pytorch, paddle"""
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import paddle

dim_x = 5
sin = paddle.sin
cos = paddle.cos
concat = paddle.concat
v = 0.5
alpha = 0.01
# PDE
def pde(x, y, v):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + v * dy_x - alpha * dy_xx

geom = dde.geometry.Rectangle([0, 0], [1, 1])

'''
def func_ic(x, v):
    return np.sin(2*np.pi*v)


def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[1], 0)


ic = dde.icbc.DirichletBC(geom, func_ic, boundary)
'''
def boundary_L(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0)

def boundary_R(x, on_boundary):
    return on_boundary and np.isclose(x[0], 1)

def bc_func_L(x):
    return np.sin(2 * np.pi * x[:, 1:2])  # u(0, t) = sin(2πt)

def bc_func_R(x):
    return np.zeros((len(x), 1))  # u(1, t) = 0
bc_L = dde.icbc.DirichletBC(geom, bc_func_L, boundary_L)
bc_R = dde.icbc.DirichletBC(geom, bc_func_R, boundary_R)
ic = dde.icbc.DirichletBC(geom, bc_func_L, boundary_L)

pde = dde.data.PDE(geom, pde, ic, num_domain=50, num_boundary=50)

# Function space
func_space = dde.data.GRF(kernel="ExpSineSquared", length_scale=1)

# Data
eval_pts = np.linspace(0, 1, num=50)[:, None]
data = dde.data.PDEOperator(pde, func_space, eval_pts, 1000, function_variables=[0])

# Net
net = dde.nn.DeepONet(
    [50, 128, 128, 128],
    [dim_x, 128, 128, 128],
    "tanh",
    "Glorot normal",
)


def periodic(x):
    x, t = x[:, :1], x[:, 1:]
    x = x * 2 * np.pi
    return concat([cos(x), sin(x), cos(2 * x), sin(2 * x), t], 1)


net.apply_feature_transform(periodic)

model = dde.Model(data, net)
model.compile("adam", lr=0.0005)
losshistory, train_state = model.train(iterations=200)
dde.utils.plot_loss_history(losshistory)

x = np.linspace(0, 1, num=100)
t = np.linspace(0, 1, num=100)
u_true = np.sin(2 * np.pi * (x - t[:, None]))
plt.figure()
plt.imshow(u_true)
plt.colorbar()

v_branch = np.sin(2 * np.pi * eval_pts)[:, 0]
xv, tv = np.meshgrid(x, t)
x_trunk = np.vstack((np.ravel(xv), np.ravel(tv))).T
u_pred = model.predict((np.tile(v_branch, (100 * 100, 1)), x_trunk))
u_pred = u_pred.reshape((100, 100))
plt.figure()
plt.imshow(u_pred)
plt.colorbar()
plt.show()

print(dde.metrics.l2_relative_error(u_true, u_pred))
