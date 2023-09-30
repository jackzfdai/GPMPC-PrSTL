import numpy as np
import torch
import math
import gpytorch
from scipy.stats import norm
from scipy.signal import cont2discrete
import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
from casadi import *
from datetime import datetime

#wind params
K_vent = 0.5
var_vent = 0.005

#Generate GP training data
torch.manual_seed(0)

plot_gp_training = False

num_training_samples = 50
training_iter = 500

train_x1 = torch.rand(num_training_samples) * 7 
train_x2 = torch.rand(num_training_samples) * 7 
train_x3 = torch.rand(num_training_samples) * 7
# train_v1 = torch.rand(num_training_samples) * 4 - 2
# train_v2 = torch.rand(num_training_samples) * 4 - 2
# train_v3 = torch.rand(num_training_samples) * 4 - 2
# train_r = torch.rand(num_training_samples) * 2 - 1 
# train_dr = torch.rand(num_training_samples) * 2 - 1
# train_w = torch.rand(num_training_samples) * 2 - 1 
# train_dw = torch.rand(num_training_samples) * 2 - 1
# train_uf = torch.rand(num_training_samples) * 14.48 - 7.24
# train_ux = torch.rand(num_training_samples) * 7.24 - 3.62
# train_uy = torch.rand(num_training_samples) * 7.24 - 3.62

train_inputs = torch.stack([train_x1, train_x2, train_x3], dim=0) #, train_v1, train_v2, train_v3, train_r, train_dr, train_w, train_dw, train_uf, train_ux, train_uy], dim=0)
train_inputs = torch.transpose(train_inputs, dim0=0, dim1=1)

train_g_v1 = K_vent/(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7)) * (train_inputs[:, 0])/torch.sqrt(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7)) + torch.randn(num_training_samples) * math.sqrt(var_vent) 
train_g_v2 = torch.randn(num_training_samples) * math.sqrt(var_vent)
train_g_v3 = K_vent/(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7)) * (train_inputs[:, 2] - 7)/torch.sqrt(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7)) + torch.randn(num_training_samples) * math.sqrt(var_vent)

actual_g_v1 = K_vent/(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7)) * (train_inputs[:, 0])/torch.sqrt(torch.square(train_inputs[:, 0]) + torch.square(train_inputs[:, 2] - 7))

#Estimate GP from training data
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
# initialize likelihood and model
likelihood_g_v1 = gpytorch.likelihoods.GaussianLikelihood()
likelihood_g_v2 = gpytorch.likelihoods.GaussianLikelihood()
likelihood_g_v3 = gpytorch.likelihoods.GaussianLikelihood()
model_g_v1 = ExactGPModel(train_inputs, train_g_v1, likelihood_g_v1)
model_g_v2 = ExactGPModel(train_inputs, train_g_v2, likelihood_g_v2)
model_g_v3 = ExactGPModel(train_inputs, train_g_v3, likelihood_g_v3)

# Find optimal model hyperparameters
model_g_v1.train()
model_g_v2.train()
model_g_v3.train()
likelihood_g_v1.train()
likelihood_g_v2.train()
likelihood_g_v3.train()

#Train GP for v1 component
optimizer_v1 = torch.optim.Adam(model_g_v1.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll_v1 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_g_v1, model_g_v1)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer_v1.zero_grad()
    # Output from model
    output = model_g_v1(train_inputs)
    # Calc loss and backprop gradients
    loss = -mll_v1(output, train_g_v1)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_g_v1.covar_module.base_kernel.lengthscale.item(),
        model_g_v1.likelihood.noise.item()
    ))
    optimizer_v1.step()

optimizer_v2 = torch.optim.Adam(model_g_v2.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll_v2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_g_v2, model_g_v2)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer_v2.zero_grad()
    # Output from model
    output = model_g_v2(train_inputs)
    # Calc loss and backprop gradients
    loss = -mll_v2(output, train_g_v2)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_g_v2.covar_module.base_kernel.lengthscale.item(),
        model_g_v2.likelihood.noise.item()
    ))
    optimizer_v2.step()

optimizer_v3 = torch.optim.Adam(model_g_v3.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll_v3 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_g_v3, model_g_v3)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer_v3.zero_grad()
    # Output from model
    output = model_g_v3(train_inputs)
    # Calc loss and backprop gradients
    loss = -mll_v3(output, train_g_v3)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_g_v3.covar_module.base_kernel.lengthscale.item(),
        model_g_v3.likelihood.noise.item()
    ))
    optimizer_v3.step()

# Get into evaluation (predictive posterior) mode
model_g_v1.eval()
likelihood_g_v1.eval()
model_g_v2.eval()
likelihood_g_v2.eval()
model_g_v3.eval()
likelihood_g_v3.eval()
test_out = model_g_v1.covar_module(train_inputs)
test_out_dense = test_out.to_dense()
print(test_out_dense.size()[1])
print(model_g_v1.covar_module.base_kernel.lengthscale)
print(model_g_v1.covar_module.outputscale)
for param_name, param in model_g_v1.covar_module.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
# print(model_g1.likelihood.hyperparameters)
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x1 = torch.linspace(0, 7, 50)
    test_x2 = torch.linspace(0, 7, 50)
    test_x3 = torch.linspace(0, 7, 50)
    # test_v1 = torch.linspace(-1.5, 1.5, 50)
    # test_v2 = torch.linspace(-1.5, 1.5, 50)
    # test_v3 = torch.linspace(-1.5, 1.5, 50)
    # test_r = torch.linspace(-1, 1, 50)
    # test_dr = torch.linspace(-1, 1, 50)
    # test_w = torch.linspace(-1, 1, 50)
    # test_dw = torch.linspace(-1, 1, 50)
    # test_uf = torch.linspace(-4.545, 9.935, 50)
    # test_ux = torch.linspace(-3.62, 3.62, 50)
    # test_uy = torch.linspace(3.62, 3.62, 50)
    test_input = torch.stack([test_x1, test_x2, test_x3], dim=0) #, test_v1, test_v2, test_v3, test_r, test_dr, test_w, test_dw, test_uf, test_ux, test_uy], dim=0)
    test_input = torch.transpose(test_input, dim0=0, dim1=1)
    observed_pred = likelihood_g_v1(model_g_v1(test_input))
    observed_pred_v2 = likelihood_g_v2(model_g_v2(test_input))
    observed_pred_v3 = likelihood_g_v3(model_g_v3(test_input))

if plot_gp_training == True:
    with torch.no_grad():
        f, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.scatter3D(train_inputs[:,0].numpy(), train_inputs[:,2].numpy(), train_g_v1.numpy(), 'k*')
        ax.plot3D(test_input[:,0].numpy(), test_input[:,2].numpy(), observed_pred.mean.numpy())

        ax.set_zlim(-0.3, 0.3)
        ax.zaxis.set_major_formatter('{x:.02f}')
        ax.set_xlabel('x1')
        ax.set_ylabel('x3')
        ax.set_zlabel('v1')

        f1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
        # Plot the surface.
        ax1.scatter3D(train_inputs[:,0].numpy(), train_inputs[:,2].numpy(), train_g_v3.numpy(), 'k*')
        ax1.plot3D(test_input[:,0].numpy(), test_input[:,2].numpy(), observed_pred_v3.mean.numpy())

        ax1.set_zlim(-0.3, 0.3)
        ax1.zaxis.set_major_formatter('{x:.02f}')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x3')
        ax1.set_zlabel('v3')

        

        plt.show()

KzzPrior_v1 = model_g_v1.covar_module(train_inputs).to_dense()
dMu_v1 = torch.matmul(torch.linalg.inv(KzzPrior_v1 + model_g_v1.likelihood.noise.item()*torch.eye(KzzPrior_v1.size()[0])), train_g_v1) 
# print(dMu)
# print(dMu.size())
dSigma_v1 = torch.linalg.inv(KzzPrior_v1 + model_g_v1.likelihood.noise.item()*torch.eye(KzzPrior_v1.size()[0]))

KzzPrior_v2 = model_g_v2.covar_module(train_inputs).to_dense()
dMu_v2 = torch.matmul(torch.linalg.inv(KzzPrior_v2 + model_g_v2.likelihood.noise.item()*torch.eye(KzzPrior_v2.size()[0])), train_g_v2) 
# dSigma2 = torch.linalg.inv(KzzPrior2 + model_g2.covar_module.outputscale*torch.eye(KzzPrior2.size()[0]))
dSigma_v2 = torch.linalg.inv(KzzPrior_v2 + model_g_v2.likelihood.noise.item()*torch.eye(KzzPrior_v2.size()[0]))
# print(dSigma2)

KzzPrior_v3 = model_g_v3.covar_module(train_inputs).to_dense()
dMu_v3 = torch.matmul(torch.linalg.inv(KzzPrior_v3 + model_g_v3.likelihood.noise.item()*torch.eye(KzzPrior_v3.size()[0])), train_g_v3) 
# dSigma2 = torch.linalg.inv(KzzPrior2 + model_g2.covar_module.outputscale*torch.eye(KzzPrior2.size()[0]))
dSigma_v3 = torch.linalg.inv(KzzPrior_v3 + model_g_v3.likelihood.noise.item()*torch.eye(KzzPrior_v3.size()[0]))
# print(dSigma2)

l_v1 = model_g_v1.covar_module.base_kernel.lengthscale
sigma_v1 = model_g_v1.covar_module.outputscale

l_v2 = model_g_v2.covar_module.base_kernel.lengthscale
sigma_v2 = model_g_v2.covar_module.outputscale

l_v3 = model_g_v3.covar_module.base_kernel.lengthscale
sigma_v3 = model_g_v3.covar_module.outputscale
np_rng = np.random.default_rng(seed=0)

#Control parameters
T = 6 #Time horizon
surveillance_A_period_t = 3 #s
surveillance_B_period_t = 3 #s
dt = 0.2
N = int(T/dt) # Control intervals 
surveillance_A_period = int(surveillance_A_period_t/dt)
surveillance_B_period = int(surveillance_B_period_t/dt)
epsilon = 0.0
invCDFVarphiEpsilon_hi = norm.ppf(0.01) #satisfy probabilistic predicates by 1-p (enter p in norm.ppf())
invCDFVarphiEpsilon_lo = norm.ppf(0.2)
invCDFV = norm.ppf(0.25)
M = N

#quadrotor parameters (from Smooth Operator Codes)
mass = 0.5 #quadrotor mass in kg
g = 9.81 #gravitational acceleration in m/s^2

#x := [x, y, z, v_x, v_y, v_z]
#u := [theta, phi, thrust]

Ac = np.concatenate([np.concatenate([np.zeros((3, 3)), np.eye(3)], axis=1),
                     np.zeros((3, 6))], axis=0)

subB = np.array([[g, 0, 0], [0, -g, 0], [0, 0, 1/mass]])
Bc = np.concatenate([np.zeros((3, 3)),
                     subB], axis=0)

C = np.zeros((6, 6))
D = np.zeros((6, 3))

sysd = cont2discrete((Ac, Bc, C, D), dt=dt)
A = sysd[0]
B = sysd[1]
print(A)
print(B)

Bd = np.concatenate([dt*np.eye(3), np.eye(3)], axis=0)
# Bd = np.zeros((10, 3))

#cost function params
R = np.array([[1/4, 0, 0], [0, 1/2, 0], [0, 0, 1/2]])
R_theta = 1/4
R_phi = 1/4
R_thrust = 1/2

#Variable bounds
lbx = [0., 0., 0., -2.5, -2.5, -2.5]
ubx = [7., 7., 7., 2.5, 2.5, 2.5]

lbsigma = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
ubsigma = [10., 10., 10., 10., 10., 10.]

lbstddev = [1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]
ubstddev = [10., 10., 10., 10., 10., 10.]

maxAng = np.deg2rad(45)
minAng = -maxAng
maxThrust = mass*g
minThrust = -maxThrust
lbu = [minAng, minAng, minThrust]
ubu = [maxAng, maxAng, maxThrust]

x_lb = lbx[0] #Split into x1, x2 if needed later 
x_ub = ubx[0]
v_lb = lbx[3]
v_ub = ubx[3]
theta_lb = lbu[0]
theta_ub = ubu[0]
phi_lb = lbu[1]
phi_ub = ubu[1]
thrust_lb = lbu[2]
thrust_ub = ubu[2]

#Environment parameters
goal_A_polygon_x1 = [1.5, 3]
goal_A_polygon_x2 = [1.5, 3] 
goal_A_polygon_x3 = [3, 4.5]
goal_B_polygon_x1 = [4.5, 6]
goal_B_polygon_x2 = [4.5, 6]
goal_B_polygon_x3 = [1.5, 3]
obstacle_polygon_x1 = [2.5, 4]
obstacle_polygon_x2 = [3.5, 5]
obstacle_polygon_x3 = [0, 7]

def plotTraj(x1_sol, x2_sol, x3_sol):
    ax = plt.figure().add_subplot(projection='3d')
    
    #obstacle
    X, Y = np.meshgrid(obstacle_polygon_x1, obstacle_polygon_x2)
    Z_1 = obstacle_polygon_x3[0]*np.ones(4).reshape(2, 2)
    Z_2 = obstacle_polygon_x3[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Y,Z_1, alpha=0.5, color='black')
    ax.plot_wireframe(X,Y,Z_2, alpha=0.5, color='black')
    ax.plot_surface(X,Y,Z_1, alpha=0.5, color='r')
    ax.plot_surface(X,Y,Z_2, alpha=0.5, color='r')

    X, Y = np.meshgrid(obstacle_polygon_x1, obstacle_polygon_x3)
    Z_1 = obstacle_polygon_x2[0]*np.ones(4).reshape(2, 2)
    Z_2 = obstacle_polygon_x2[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Z_1,Y, alpha=0.5, color='black')
    ax.plot_wireframe(X,Z_2,Y, alpha=0.5, color='black')
    ax.plot_surface(X,Z_1,Y, alpha=0.5, color='r')
    ax.plot_surface(X,Z_2,Y, alpha=0.5, color='r')

    X, Y = np.meshgrid(obstacle_polygon_x2, obstacle_polygon_x3)
    Z_1 = obstacle_polygon_x1[0]*np.ones(4).reshape(2, 2)
    Z_2 = obstacle_polygon_x1[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(Z_1,X,Y, alpha=0.5, color='black')
    ax.plot_wireframe(Z_2,X,Y, alpha=0.5, color='black')
    ax.plot_surface(Z_1,X,Y, alpha=0.5, color='r')
    ax.plot_surface(Z_2,X,Y, alpha=0.5, color='r')

    #goal A
    X, Y = np.meshgrid(goal_A_polygon_x1, goal_A_polygon_x2)
    Z_1 = goal_A_polygon_x3[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_A_polygon_x3[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Y,Z_1, alpha=0.5, color='black')
    ax.plot_wireframe(X,Y,Z_2, alpha=0.5, color='black')
    ax.plot_surface(X,Y,Z_1, alpha=0.5, color='g')
    ax.plot_surface(X,Y,Z_2, alpha=0.5, color='g')

    X, Y = np.meshgrid(goal_A_polygon_x2, goal_A_polygon_x3)
    Z_1 = goal_A_polygon_x2[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_A_polygon_x2[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Z_1,Y, alpha=0.5, color='black')
    ax.plot_wireframe(X,Z_2,Y, alpha=0.5, color='black')
    ax.plot_surface(X,Z_1,Y, alpha=0.5, color='g')
    ax.plot_surface(X,Z_2,Y, alpha=0.5, color='g')

    X, Y = np.meshgrid(goal_A_polygon_x2, goal_A_polygon_x3)
    Z_1 = goal_A_polygon_x1[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_A_polygon_x1[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(Z_1,X,Y, alpha=0.5, color='black')
    ax.plot_wireframe(Z_2,X,Y, alpha=0.5, color='black')
    ax.plot_surface(Z_1,X,Y, alpha=0.5, color='g')
    ax.plot_surface(Z_2,X,Y, alpha=0.5, color='g')

    #goal B
    X, Y = np.meshgrid(goal_B_polygon_x1, goal_B_polygon_x2)
    Z_1 = goal_B_polygon_x3[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_B_polygon_x3[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Y,Z_1, alpha=0.5, color='black')
    ax.plot_wireframe(X,Y,Z_2, alpha=0.5, color='black')
    ax.plot_surface(X,Y,Z_1, alpha=0.5, color='b')
    ax.plot_surface(X,Y,Z_2, alpha=0.5, color='b')

    X, Y = np.meshgrid(goal_B_polygon_x2, goal_B_polygon_x3)
    Z_1 = goal_B_polygon_x2[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_B_polygon_x2[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(X,Z_1,Y, alpha=0.5, color='black')
    ax.plot_wireframe(X,Z_2,Y, alpha=0.5, color='black')
    ax.plot_surface(X,Z_1,Y, alpha=0.5, color='b')
    ax.plot_surface(X,Z_2,Y, alpha=0.5, color='b')

    X, Y = np.meshgrid(goal_B_polygon_x2, goal_B_polygon_x3)
    Z_1 = goal_B_polygon_x1[0]*np.ones(4).reshape(2, 2)
    Z_2 = goal_B_polygon_x1[1]*np.ones(4).reshape(2, 2)
    ax.plot_wireframe(Z_1,X,Y, alpha=0.5, color='black')
    ax.plot_wireframe(Z_2,X,Y, alpha=0.5, color='black')
    ax.plot_surface(Z_1,X,Y, alpha=0.5, color='b')
    ax.plot_surface(Z_2,X,Y, alpha=0.5, color='b')

    ax.set_xlim(x_lb, x_ub)
    ax.set_ylim(x_lb, x_ub)
    ax.set_zlim(x_lb - 0.1, x_ub)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    ax.scatter(x1_sol[0], x2_sol[0], x3_sol[0], s=120, facecolors='none', edgecolors='black')
    ax.set_aspect('equal')
    ax.plot(x1_sol, x2_sol, x3_sol, '-o')

    # ax.view_init(elev=90, azim=-90, roll=0)

    plt.grid(True)

def plotSol(x1_sol, x2_sol, x3_sol, v1_sol, v2_sol, v3_sol, uf_sol, ux_sol, uy_sol):
    tgrid = [k for k in range(N)]
    plt.figure()
    plt.plot(tgrid, x1_sol, '-')
    plt.plot(tgrid, x2_sol, '--')
    plt.plot(tgrid, x3_sol, '-')
    plt.plot(tgrid, v1_sol, '-')
    plt.plot(tgrid, v2_sol, '--')
    plt.plot(tgrid, v3_sol, '-')
    plt.step(tgrid, vertcat(DM.nan(1), uf_sol), '.')
    plt.step(tgrid, vertcat(DM.nan(1), ux_sol), '-.')
    plt.step(tgrid, vertcat(DM.nan(1), uy_sol), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2', 'x3', 'v1', 'v2', 'v3', 'uf', 'ux', 'uy'])
    plt.grid(True)


def solveMICP(plot, x0, v0):
    #Problem at time step k = 0 ____________________________________________
    m = gb.Model('integratorRobot2D')

    #Cost function (zero objective since we want to test feasibility)
    # zeroObjective = gb.LinExpr(0)
    # m.setObjective(zeroObjective)

    #Set x value to corresponding value in sequence
    x = m.addVars(N, 3, lb=x_lb, ub=x_ub, name="x")
    v = m.addVars(N, 3, lb=v_lb, ub=v_ub, name="v")
    theta = m.addVars(N-1-(len(x0) - 1), lb=theta_lb, ub=theta_ub, name="theta")
    phi = m.addVars(N-1-(len(x0) - 1), lb=phi_lb, ub=phi_ub, name="phi")
    thrust = m.addVars(N-1-(len(x0) - 1), lb=thrust_lb, ub=thrust_ub, name="thrust")

    for i in range(len(x0)):
        m.addConstr(x[i, 0] == x0[i, 0])
        m.addConstr(x[i, 1] == x0[i, 1])
        m.addConstr(x[i, 2] == x0[i, 2])
        m.addConstr(v[i, 0] == v0[i, 0])
        m.addConstr(v[i, 1] == v0[i, 1])
        m.addConstr(v[i, 2] == v0[i, 2])

    objective = gb.QuadExpr()
    zero_objective = gb.LinExpr(0)
    for i in range(N-1-(len(x0) - 1)):
        objective_i = gb.QuadExpr(R_theta*(theta[i]**2) + R_phi*(phi[i]**2) + R_thrust*(thrust[i]**2)) #gb.QuadExpr(thrust[i]**2)
        objective.add(objective_i)
    # for i in range(N):
    #     objective_i = gb.QuadExpr(v[i, 0]**2 + v[i, 1]**2 + v[i, 2]**2)
    #     objective.add(objective_i)

    m.setObjective(objective)

    for i in range(len(x0) - 1, N-1):
        m.addConstr(x[i+1, 0] == A[0, 0]*x[i, 0] + A[0, 1]*x[i, 1] + A[0, 2]*x[i, 2] + A[0, 3]*v[i, 0] + A[0, 4]*v[i, 1] + A[0, 5]*v[i, 2] + B[0, 0]*theta[i-(len(x0) - 1)] + B[0, 1]*phi[i-(len(x0) - 1)] + B[0, 2]*thrust[i-(len(x0) - 1)])
        m.addConstr(x[i+1, 1] == A[1, 0]*x[i, 0] + A[1, 1]*x[i, 1] + A[1, 2]*x[i, 2] + A[1, 3]*v[i, 0] + A[1, 4]*v[i, 1] + A[1, 5]*v[i, 2] + B[1, 0]*theta[i-(len(x0) - 1)] + B[1, 1]*phi[i-(len(x0) - 1)] + B[1, 2]*thrust[i-(len(x0) - 1)])
        m.addConstr(x[i+1, 2] == A[2, 0]*x[i, 0] + A[2, 1]*x[i, 1] + A[2, 2]*x[i, 2] + A[2, 3]*v[i, 0] + A[2, 4]*v[i, 1] + A[2, 5]*v[i, 2] + B[2, 0]*theta[i-(len(x0) - 1)] + B[2, 1]*phi[i-(len(x0) - 1)] + B[2, 2]*thrust[i-(len(x0) - 1)])
        m.addConstr(v[i+1, 0] == A[3, 0]*x[i, 0] + A[3, 1]*x[i, 1] + A[3, 2]*x[i, 2] + A[3, 3]*v[i, 0] + A[3, 4]*v[i, 1] + A[3, 5]*v[i, 2] + B[3, 0]*theta[i-(len(x0) - 1)] + B[3, 1]*phi[i-(len(x0) - 1)] + B[3, 2]*thrust[i-(len(x0) - 1)])
        m.addConstr(v[i+1, 1] == A[4, 0]*x[i, 0] + A[4, 1]*x[i, 1] + A[4, 2]*x[i, 2] + A[4, 3]*v[i, 0] + A[4, 4]*v[i, 1] + A[4, 5]*v[i, 2] + B[4, 0]*theta[i-(len(x0) - 1)] + B[4, 1]*phi[i-(len(x0) - 1)] + B[4, 2]*thrust[i-(len(x0) - 1)])
        m.addConstr(v[i+1, 2] == A[5, 0]*x[i, 0] + A[5, 1]*x[i, 1] + A[5, 2]*x[i, 2] + A[5, 3]*v[i, 0] + A[5, 4]*v[i, 1] + A[5, 5]*v[i, 2] + B[5, 0]*theta[i-(len(x0) - 1)] + B[5, 1]*phi[i-(len(x0) - 1)] + B[5, 2]*thrust[i-(len(x0) - 1)])
        
    qVarphi1 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi1")
    qVarphi2 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi2")
    qVarphi3 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi3")
    qVarphi4 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi4")
    # qVarphi5 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi5")
    # qVarphi6 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi6")
    # pPhi1 = m.addVars(N, vtype=GRB.BINARY, name="pPhi1")
    # pPhi2 = m.addVars(N, vtype=GRB.BINARY, name="pPhi2")


    # pPhi1_sum = gb.LinExpr()
    # pPhi2_sum = gb.LinExpr()

    for i in range(0, N):
        # Avoid predicates
        m.addConstr(0 <= M*(1 - qVarphi1[i]) - epsilon + (obstacle_polygon_x1[0] - x[i, 0]))
        m.addConstr(0 <= M*(1 - qVarphi2[i]) - epsilon + (x[i, 0] - obstacle_polygon_x1[1]))
        m.addConstr(0 <= M*(1 - qVarphi3[i]) - epsilon + (obstacle_polygon_x2[0] - x[i, 1]))
        m.addConstr(0 <= M*(1 - qVarphi4[i]) - epsilon + (x[i, 1] - obstacle_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - qVarphi5[i]) - epsilon + (obstacle_polygon_x3[0] - x[i, 2]))
        # m.addConstr(0 <= M*(1 - qVarphi6[i]) - epsilon + (x[i, 2] - obstacle_polygon_x3[1]))

        #Avoid disjunctions
        m.addConstr(1 <= qVarphi1[i] + qVarphi2[i] + qVarphi3[i] + qVarphi4[i]) # + qVarphi5[i] + qVarphi6[i])

        # #Reach predicates
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 0] - goal_A_polygon_x1[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 0] + goal_A_polygon_x1[1]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 1] - goal_A_polygon_x2[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 1] + goal_A_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 2] - goal_A_polygon_x3[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 2] + goal_A_polygon_x3[1]))

        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 0] - goal_B_polygon_x1[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 0] + goal_B_polygon_x1[1]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 1] - goal_B_polygon_x2[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 1] + goal_B_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 2] - goal_B_polygon_x3[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 2] + goal_B_polygon_x3[1]))

        # #pPhi1_sum
        # pPhi1_sum.add(pPhi1[i])

        # #pPhi2_sum
        # pPhi2_sum.add(pPhi2[i])
    
    # m.addConstr(1 <= pPhi1_sum)
    # m.addConstr(1 <= pPhi2_sum)

    for i in range(0, N-surveillance_A_period+1):
        pPhi1_i = m.addVars(surveillance_A_period, vtype=GRB.BINARY, name="pPhi1_{:d}".format(i))
        pPhi1_i_sum = gb.LinExpr()

        for j in range (surveillance_A_period):
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 0] - goal_A_polygon_x1[0]))
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 0] + goal_A_polygon_x1[1]))
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 1] - goal_A_polygon_x2[0]))
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 1] + goal_A_polygon_x2[1]))
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 2] - goal_A_polygon_x3[0]))
            m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 2] + goal_A_polygon_x3[1]))

            pPhi1_i_sum.add(pPhi1_i[j])

        m.addConstr(1 <= pPhi1_i_sum)

    for i in range(0, N-surveillance_B_period+1):
        pPhi2_i = m.addVars(surveillance_B_period, vtype=GRB.BINARY, name="pPhi2_{:d}".format(i))
        pPhi2_i_sum = gb.LinExpr()

        for j in range (surveillance_B_period):         
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 0] - goal_B_polygon_x1[0]))
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 0] + goal_B_polygon_x1[1]))
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 1] - goal_B_polygon_x2[0]))
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 1] + goal_B_polygon_x2[1]))
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 2] - goal_B_polygon_x3[0]))
            m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 2] + goal_B_polygon_x3[1]))

            pPhi2_i_sum.add(pPhi2_i[j])

        m.addConstr(1 <= pPhi2_i_sum)

    m.optimize()
    
    x1_sol = []
    x2_sol = []
    x3_sol = []
    v1_sol = []
    v2_sol = []
    v3_sol = []
    theta_sol = []
    phi_sol = []
    thrust_sol = []
    qVarphi1_sol = []
    qVarphi2_sol = []
    qVarphi3_sol = []
    qVarphi4_sol = []
    # qVarphi5_sol = []
    # qVarphi6_sol = []
    pPhi1_sol = []
    pPhi2_sol = []

    # solver_runtime = m.getAttr('Runtime')
    solver_status = m.getAttr('Status')
    if solver_status == GRB.OPTIMAL:
        print("m results________________________________________________")
        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X))

        print("Runtime: " + str(m.Runtime) + "s")

        for i in range(N):
            x1_sol_i = m.getVarByName("x[{:d},0]".format(i))
            x2_sol_i = m.getVarByName("x[{:d},1]".format(i))
            x3_sol_i = m.getVarByName("x[{:d},2]".format(i))
            x1_sol.append(x1_sol_i.X)
            x2_sol.append(x2_sol_i.X)
            x3_sol.append(x3_sol_i.X)
            v1_sol_i = m.getVarByName("v[{:d},0]".format(i))
            v2_sol_i = m.getVarByName("v[{:d},1]".format(i))
            v3_sol_i = m.getVarByName("v[{:d},2]".format(i))
            v1_sol.append(v1_sol_i.X)
            v2_sol.append(v2_sol_i.X)
            v3_sol.append(v3_sol_i.X)

            qVarphi1_sol_i = m.getVarByName("qVarphi1[{:d}]".format(i))
            qVarphi1_sol.append(qVarphi1_sol_i.X)
            qVarphi2_sol_i = m.getVarByName("qVarphi2[{:d}]".format(i))
            qVarphi2_sol.append(qVarphi2_sol_i.X) 
            qVarphi3_sol_i = m.getVarByName("qVarphi3[{:d}]".format(i))
            qVarphi3_sol.append(qVarphi3_sol_i.X)
            qVarphi4_sol_i = m.getVarByName("qVarphi4[{:d}]".format(i))
            qVarphi4_sol.append(qVarphi4_sol_i.X)
            # qVarphi5_sol_i = m.getVarByName("qVarphi5[{:d}]".format(i))
            # qVarphi5_sol.append(qVarphi5_sol_i.X)
            # qVarphi6_sol_i = m.getVarByName("qVarphi6[{:d}]".format(i))
            # qVarphi6_sol.append(qVarphi6_sol_i.X)
            
        for i in range(N-surveillance_A_period+1):
            pPhi1_sol_i = [] 
            for j in range(surveillance_A_period):
                pPhi1_sol_i_j = m.getVarByName("pPhi1_{:d}[{:d}]".format(i, j))
                pPhi1_sol_i.append(pPhi1_sol_i_j.X)
            pPhi1_sol.append(pPhi1_sol_i)
        
        for i in range(N-surveillance_B_period+1):
            pPhi2_sol_i = [] 
            for j in range(surveillance_B_period):
                pPhi2_sol_i_j = m.getVarByName("pPhi2_{:d}[{:d}]".format(i, j))
                pPhi2_sol_i.append(pPhi2_sol_i_j.X)
            pPhi2_sol.append(pPhi2_sol_i)

        for i in range(N-1-(len(x0) - 1)):
            theta_sol_i = m.getVarByName("theta[{:d}]".format(i))
            phi_sol_i = m.getVarByName("phi[{:d}]".format(i))
            thrust_sol_i = m.getVarByName("thrust[{:d}]".format(i))
            theta_sol.append(theta_sol_i.X)
            phi_sol.append(phi_sol_i.X)
            thrust_sol.append(thrust_sol_i.X)

        # print(x1_sol)
        # print(x2_sol)
        # print(x3_sol)
        
        if plot == True:
            plotTraj(x1_sol, x2_sol, x3_sol)
            print(x1_sol)
            print(x2_sol)
            print(x3_sol)

    return theta_sol, phi_sol, thrust_sol, qVarphi1_sol, qVarphi2_sol, qVarphi3_sol, qVarphi4_sol, pPhi1_sol, pPhi2_sol

def solveNLP(plot, x0, v0, sigma_x0, sigma_v0, uf_milp, ux_milp, uy_milp, qVarphi1_milp, qVarphi2_milp, qVarphi3_milp, qVarphi4_milp, pPhi1_milp, pPhi2_milp): #qVarphi5_milp, qVarphi6_milp, 
    feasible = False

    #Match control horizon indexing
    local_N = N - 1

    # Declare model variables
    x = SX.sym('x', 6)
    sigma = SX.sym('sigma', 6) #assuming sigma_uf, sigma_ux, sigma_uf = 0
    u = SX.sym('u', 3)

    # Model equations
    x_next = vertcat(dot(A[0, :], x) + dot(B[0, :], u) + Bd[0, 0]*(sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v1.detach().numpy())),
                    dot(A[1, :], x) + dot(B[1, :], u) + Bd[1, 1]*(sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v2.detach().numpy())),
                    dot(A[2, :], x) + dot(B[2, :], u) + Bd[2, 2]*(sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v3.detach().numpy())),
                    dot(A[3, :], x) + dot(B[3, :], u) + Bd[3, 0]*(sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v1.detach().numpy())),
                    dot(A[4, :], x) + dot(B[4, :], u) + Bd[4, 1]*(sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v2.detach().numpy())),
                    dot(A[5, :], x) + dot(B[5, :], u) + Bd[5, 2]*(sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu_v3.detach().numpy())))
    
    # x_next = vertcat(dot(A[0, :], x) + dot(B[0, :], u),
    #                 dot(A[1, :], x) + dot(B[1, :], u),
    #                 dot(A[2, :], x) + dot(B[2, :], u),
    #                 dot(A[3, :], x) + dot(B[3, :], u),
    #                 dot(A[4, :], x) + dot(B[4, :], u),
    #                 dot(A[5, :], x) + dot(B[5, :], u))
    
    # r = x[0] ** 2 + (x[2] - ubx[2]) ** 2
    #mu_vdx = K_vent/r * x[0] / r 
    # mu_vdy = 0
    # mu_vdz = K_vent/r * (x[2] - ubx[2]) / r
    # x_next = vertcat(dot(A[0, :], x) + dot(B[0, :], u) + Bd[0, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[0, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)),
    #                 dot(A[1, :], x) + dot(B[1, :], u) + Bd[1, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[1, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)),
    #                 dot(A[2, :], x) + dot(B[2, :], u) + Bd[2, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[2, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)),
    #                 dot(A[3, :], x) + dot(B[3, :], u) + Bd[3, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[3, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)),
    #                 dot(A[4, :], x) + dot(B[4, :], u) + Bd[4, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[4, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)),
    #                 dot(A[5, :], x) + dot(B[5, :], u) + Bd[5, 0] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * x[0] / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)) + Bd[5, 2] * (K_vent / (x[0] ** 2 + (x[2] - ubx[2]) ** 2) * (x[2] - ubx[2]) / (x[0] ** 2 + (x[2] - ubx[2]) ** 2)))

    sigma_next = vertcat(dot(A[0, :], sigma) + Bd[0, 0]*(sigma_v1.item() - (sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v1.detach().numpy()) @ (sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[1, :], sigma) + Bd[1, 1]*(sigma_v2.item() - (sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v2.detach().numpy()) @ (sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[2, :], sigma) + Bd[2, 2]*(sigma_v3.item() - (sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v3.detach().numpy()) @ (sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[3, :], sigma) + Bd[3, 0]*(sigma_v1.item() - (sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v1.detach().numpy()) @ (sigma_v1.item()*exp((-1/(2*(l_v1.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[4, :], sigma) + Bd[4, 1]*(sigma_v2.item() - (sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v2.detach().numpy()) @ (sigma_v2.item()*exp((-1/(2*(l_v2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[5, :], sigma) + Bd[5, 2]*(sigma_v3.item() - (sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma_v3.detach().numpy()) @ (sigma_v3.item()*exp((-1/(2*(l_v3.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T))
    
    # sigma_next = vertcat(dot(A[0, :], sigma),
    #                      dot(A[1, :], sigma),
    #                      dot(A[2, :], sigma),
    #                      dot(A[3, :], sigma),
    #                      dot(A[4, :], sigma),
    #                      dot(A[5, :], sigma))
    
    # var_vents = np.array([var_vent, var_vent, var_vent])
    # sigma_next = vertcat(dot(A[0, :], sigma) + np.dot(Bd[0, :], var_vents),
    #                      dot(A[1, :], sigma) + np.dot(Bd[1, :], var_vents),
    #                      dot(A[2, :], sigma) + np.dot(Bd[2, :], var_vents),
    #                      dot(A[3, :], sigma) + np.dot(Bd[3, :], var_vents),
    #                      dot(A[4, :], sigma) + np.dot(Bd[4, :], var_vents),
    #                      dot(A[5, :], sigma) + np.dot(Bd[5, :], var_vents))

    # Objective term
    # L = x[3] ** 2 + x[4] ** 2 + x[5] ** 2
    L = R_theta*(u[0]**2) + R_phi*(u[1]**2) + R_thrust*(u[2]**2)

    F = Function('F', [x, sigma, u], [x_next, sigma_next, L], ['x_k', 'sigma_k', 'u_k'], ['x_k+1', 'sigma_k+1', 'L'])

    stddev = sqrt(sigma)

    sigmaToStddev = Function('sigmaToStddev', [sigma], [stddev], ['sigma_k'], ['stddev_k'])

    # Initial guess for u
    # u_start = [DM([0., 0., 0.])] * local_N
    u_start = []
    for i in range(local_N-(len(x0) - 1)):
        u_start.append(DM([uf_milp[i], ux_milp[i], uy_milp[i]]))
    # DM([u1_milp[k], u2_milp[k]])
    # Get a feasible trajectory as an initial 
    x_start = []
    sigma_start = []
    stddev_start = []
    for i in range(len(x0)):
        xk = DM(np.concatenate([x0[i], v0[i]], axis=0))
        x_start += [xk]
        sigmak = DM(np.concatenate([sigma_x0, sigma_v0], axis=0)) #assuming sigma_x0, v0 is always 0
        sigma_start += [sigmak]
        stddevk = sigmaToStddev(sigma_k=sigmak)['stddev_k']
        stddev_start += [stddevk]

    for k in range(local_N-(len(x0) - 1)):
        xk = F(x_k=xk, sigma_k=sigmak, u_k=u_start[k])['x_k+1']
        x_start += [xk]
        sigmak = F(x_k=xk, sigma_k=sigmak, u_k=u_start[k])['sigma_k+1']
        sigma_start += [sigmak]
        stddevk = sigmaToStddev(sigma_k=sigmak)['stddev_k']
        stddev_start += [stddevk]

    # Start with an empty NLP
    # w=[]
    # w0 = []
    # lbw = []
    # ubw = []

    opt_vars = []
    opt_vars0 = []
    lbopt_vars = []
    ubopt_vars = []

    discrete = []
    J = 0
    g=[]
    lbg = []
    ubg = []

    # "Lift" initial conditions
    for i in range(len(x0)):
        X0_i = MX.sym('X0_' + str(i), 6)
        opt_vars += [X0_i]
        lbopt_vars += [x0[i, 0], x0[i, 1], x0[i,2], v0[i, 0], v0[i, 1], v0[i, 2]]
        ubopt_vars += [x0[i, 0], x0[i, 1], x0[i, 2], v0[i, 0], v0[i, 1], v0[i, 2]]
        opt_vars0 += [x_start[i]]
        discrete += [False, False, False, False, False, False]

        SIGMA0_i = MX.sym('sigma0_' + str(i), 6)
        opt_vars += [SIGMA0_i]
        lbopt_vars += [sigma_x0[0], sigma_x0[1], sigma_x0[2], sigma_v0[0], sigma_v0[1], sigma_v0[2]]
        ubopt_vars += [sigma_x0[0], sigma_x0[1], sigma_x0[2], sigma_v0[0], sigma_v0[1], sigma_v0[2]]
        opt_vars0 += [sigma_start[i]]
        discrete += [False, False, False, False, False, False]

        STDDEV0_i = MX.sym('stddev0_' + str(i), 6)
        opt_vars += [STDDEV0_i]
        lbopt_vars += [np.sqrt(sigma_x0[0]), np.sqrt(sigma_x0[1]), np.sqrt(sigma_x0[2]), np.sqrt(sigma_v0[0]), np.sqrt(sigma_v0[1]), np.sqrt(sigma_v0[2])]
        ubopt_vars += [np.sqrt(sigma_x0[0]), np.sqrt(sigma_x0[1]), np.sqrt(sigma_x0[2]), np.sqrt(sigma_v0[0]), np.sqrt(sigma_v0[1]), np.sqrt(sigma_v0[2])]
        opt_vars0 += [stddev_start[i]]
        discrete += [False, False, False, False, False, False]

        #dummy U variables
        if i < len(x0) - 1:
            Uk_i = MX.sym('U_' + str(i), 3)
            opt_vars   += [Uk_i]
            lbopt_vars += [0, 0, 0]
            ubopt_vars += [0, 0, 0]
            opt_vars0  += [DM(np.array([0, 0, 0]))]
            discrete += [False, False, False]

        # Formulate the NLP
        Xk = X0_i
        Sigmak = SIGMA0_i
        Stddevk = STDDEV0_i

    for k in range(local_N - (len(x0) - 1)):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k + len(x0) - 1), 3)
        opt_vars   += [Uk]
        lbopt_vars += lbu
        ubopt_vars += ubu
        opt_vars0  += [u_start[k]]
        discrete += [False, False, False]

        # Integrate till the end of the interval
        Fk = F(x_k=Xk, sigma_k = Sigmak, u_k=Uk)
        Xk_end = Fk['x_k+1']
        Sigmak_end = Fk['sigma_k+1']
        J=J+Fk['L']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str((len(x0) - 1) + k + 1), 6)
        opt_vars   += [Xk]
        lbopt_vars += lbx
        ubopt_vars += ubx
        opt_vars0  += [x_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False, False, False]

        Sigmak = MX.sym('Sigma_' + str((len(x0) - 1) + k + 1), 6)
        opt_vars += [Sigmak]
        lbopt_vars += lbsigma
        ubopt_vars += ubsigma
        opt_vars0 += [sigma_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False, False, False]

        Stddevk = MX.sym('Stddev_' + str((len(x0) - 1) + k + 1), 6)
        opt_vars += [Stddevk]
        lbopt_vars += lbstddev
        ubopt_vars += ubstddev
        opt_vars0 += [stddev_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False, False, False]
        
        Stddev_val = sigmaToStddev(sigma_k=Sigmak)['stddev_k']

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        g += [Sigmak_end - Sigmak]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        g += [Stddevk - Stddev_val]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [0, 0, 0, 0, 0, 0]

        g += [invCDFV * Stddevk[3] + (Xk[3] - lbx[3])]
        g += [invCDFV * Stddevk[4] + (Xk[4] - lbx[4])]
        g += [invCDFV * Stddevk[5] + (Xk[5] - lbx[5])]
        g += [invCDFV * Stddevk[3] + (- Xk[3] + ubx[3])]
        g += [invCDFV * Stddevk[4] + (- Xk[4] + ubx[4])]
        g += [invCDFV * Stddevk[5] + (- Xk[5] + ubx[5])]
        lbg += [0, 0, 0, 0, 0, 0]
        ubg += [2*M, 2*M, 2*M, 2*M, 2*M, 2*M]

    #STL constraints
    for i in range(len(x0), local_N + 1):
        invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
        if i > len(x0) + 5:
            invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
        g += [M*(1 - qVarphi1_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*i][0] + (- opt_vars[0 + 4*i][0] + obstacle_polygon_x1[0])]
        lbg += [0]
        ubg += [2*M]

    for i in range(len(x0), local_N + 1):
        invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
        if i > len(x0) + 5:
            invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
        g += [M*(1 - qVarphi2_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*i][0] + (opt_vars[0 + 4*i][0] - obstacle_polygon_x1[1])]
        lbg += [0]
        ubg += [2*M]

    for i in range(len(x0), local_N + 1):
        invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
        if i > len(x0) + 5:
            invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
        g += [M*(1 - qVarphi3_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*i][1] + (- opt_vars[0 + 4*i][1] + obstacle_polygon_x2[0])]
        lbg += [0]
        ubg += [2*M]

    for i in range(len(x0), local_N + 1):
        invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
        if i > len(x0) + 5:
            invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
        g += [M*(1 - qVarphi4_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*i][1] + (opt_vars[0 + 4*i][1] - obstacle_polygon_x2[1])]
        lbg += [0]
        ubg += [2*M]

    # for i in range(local_N + 1):
    #     g += [M*(1 - qVarphi5_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[1 + 3*i][2] + (- opt_vars[0 + 3*i][2] + obstacle_polygon_x3[0])]
    #     lbg += [0]
    #     ubg += [2*M]

    # for i in range(local_N + 1):
    #     g += [M*(1 - qVarphi6_milp[i]) - epsilon + invCDFVarphiEpsilon * opt_vars[1 + 3*i][2] + (opt_vars[0 + 3*i][2] - obstacle_polygon_x3[1])]
    #     lbg += [0]
    #     ubg += [2*M]

    for i in range(local_N + 2 - surveillance_A_period):
        for j in range(surveillance_A_period):
            if (i + j >= len(x0)):
                invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
                if i + j > len(x0) + 5:
                    invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][0] + (opt_vars[0 + 4*(i+j)][0] - goal_A_polygon_x1[0])]
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][0] + (- opt_vars[0 + 4*(i+j)][0] + goal_A_polygon_x1[1])]
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][1] + (opt_vars[0 + 4*(i+j)][1] - goal_A_polygon_x2[0])]
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][1] + (- opt_vars[0 + 4*(i+j)][1] + goal_A_polygon_x2[1])]
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][2] + (opt_vars[0 + 4*(i+j)][2] - goal_A_polygon_x3[0])]
                g += [M*(1 - pPhi1_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][2] + (- opt_vars[0 + 4*(i+j)][2] + goal_A_polygon_x3[1])]
                lbg += [0, 0, 0, 0, 0, 0]
                ubg += [2*M, 2*M, 2*M, 2*M, 2*M, 2*M]

    for i in range(local_N + 2 - surveillance_B_period):
        for j in range(surveillance_B_period):
            if (i + j >= len(x0)):
                invCDFVarphiEpsilon = invCDFVarphiEpsilon_hi
                if i + j > len(x0) + 5:
                    invCDFVarphiEpsilon = invCDFVarphiEpsilon_lo
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][0] + (opt_vars[0 + 4*(i+j)][0] - goal_B_polygon_x1[0])]
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][0] + (- opt_vars[0 + 4*(i+j)][0] + goal_B_polygon_x1[1])]
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][1] + (opt_vars[0 + 4*(i+j)][1] - goal_B_polygon_x2[0])]
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][1] + (- opt_vars[0 + 4*(i+j)][1] + goal_B_polygon_x2[1])]
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][2] + (opt_vars[0 + 4*(i+j)][2] - goal_B_polygon_x3[0])]
                g += [M*(1 - pPhi2_milp[i][j]) - epsilon + invCDFVarphiEpsilon * opt_vars[2 + 4*(i+j)][2] + (- opt_vars[0 + 4*(i+j)][2] + goal_B_polygon_x3[1])]
                lbg += [0, 0, 0, 0, 0, 0]
                ubg += [2*M, 2*M, 2*M, 2*M, 2*M, 2*M]

    # Concatenate decision variables and constraint terms
    opt_vars = vertcat(*opt_vars)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': opt_vars, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete});
    # nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
    # nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*opt_vars0), lbx=lbopt_vars, ubx=ubopt_vars, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())
    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']

    opt_vars1_opt = sol['x']
    lam_w_opt = sol['lam_x']
    lam_g_opt = sol['lam_g']

    # u0_opt = w1_opt[0:6*(N-1)+4][4::6]
    # u1_opt = w1_opt[0:6*(N-1)+4][5::6]
    # xGuess, sigmaGuess = makeTrajectoryGP(u0_opt, u1_opt)

    # print("Optimal result")
    # print(w1_opt)

    uTheta_opt = []
    uPhi_opt = []
    uThrust_opt = []
    feasible = np.any(opt_vars1_opt.full().flatten())

    if feasible == True:
        opt_vars1_opt = opt_vars1_opt.full().flatten()
        x1_opt = opt_vars1_opt[0:21*(N-1)+18][0::21]
        x2_opt = opt_vars1_opt[0:21*(N-1)+18][1::21]
        x3_opt = opt_vars1_opt[0:21*(N-1)+18][2::21]
        v1_opt = opt_vars1_opt[0:21*(N-1)+18][3::21]
        v2_opt = opt_vars1_opt[0:21*(N-1)+18][4::21]
        v3_opt = opt_vars1_opt[0:21*(N-1)+18][5::21]

        sigma_x1_opt = opt_vars1_opt[0:21*(N-1)+18][6::21]
        sigma_x2_opt = opt_vars1_opt[0:21*(N-1)+18][7::21]
        sigma_x3_opt = opt_vars1_opt[0:21*(N-1)+18][8::21]
        sigma_v1_opt = opt_vars1_opt[0:21*(N-1)+18][9::21]
        sigma_v2_opt = opt_vars1_opt[0:21*(N-1)+18][10::21]
        sigma_v3_opt = opt_vars1_opt[0:21*(N-1)+18][11::21]

        uTheta_opt = opt_vars1_opt[0:21*(N-1)][18::21]
        uPhi_opt = opt_vars1_opt[0:21*(N-1)][19::21]
        uThrust_opt = opt_vars1_opt[0:21*(N-1)][20::21]
        if plot == True:
            plotSol(x1_opt, x2_opt, x3_opt, v1_opt, v2_opt, v3_opt, uTheta_opt, uPhi_opt, uThrust_opt)
            plotTraj(x1_opt, x2_opt, x3_opt)
            
    return proc_runtime, uTheta_opt, uPhi_opt, uThrust_opt, feasible

def simNextStep(uncertainty, x, v, u):
    if uncertainty == True:
        r = (x[0]) ** 2 + (x[2] - ubx[2]) ** 2
        mu_vdx = K_vent/r * (x[0])/ r 
        mu_vdy = 0
        mu_vdz = K_vent/r * (x[2] - ubx[2]) / r
        sigma_vd = math.sqrt(var_vent)
        vdx = np_rng.normal(mu_vdx, sigma_vd, 1)
        vdy = np_rng.normal(mu_vdy, sigma_vd, 1)
        vdz = np_rng.normal(mu_vdz, sigma_vd, 1) 
        vd = np.concatenate([vdx, vdy, vdz])
        state = np.concatenate([x, v])
        state_nominal = (A @ state) + (B @ u) #comparison purposes
        state_next = (A @ state) + (B @ u) + (Bd @ vd)
    else: 
        state = np.concatenate([x, v])
        state_next = (A @ state) + (B @ u)
    return state_next 

def verifySTLSAT(x, v):
    #Problem at time step k = 0 ____________________________________________
    m_sat = gb.Model('integratorRobot2D_satCheck')

    #Cost function (zero objective since we want to test feasibility)
    zero_objective = gb.LinExpr(0)
    m_sat.setObjective(zero_objective)

    qVarphi1 = m_sat.addVars(N, vtype=GRB.BINARY, name="qVarphi1")
    qVarphi2 = m_sat.addVars(N, vtype=GRB.BINARY, name="qVarphi2")
    qVarphi3 = m_sat.addVars(N, vtype=GRB.BINARY, name="qVarphi3")
    qVarphi4 = m_sat.addVars(N, vtype=GRB.BINARY, name="qVarphi4")
    # qVarphi5 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi5")
    # qVarphi6 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi6")
    # pPhi1 = m.addVars(N, vtype=GRB.BINARY, name="pPhi1")
    # pPhi2 = m.addVars(N, vtype=GRB.BINARY, name="pPhi2")


    # pPhi1_sum = gb.LinExpr()
    # pPhi2_sum = gb.LinExpr()

    for i in range(0, N):
        # Avoid predicates
        m_sat.addConstr(0 <= M*(1 - qVarphi1[i]) - epsilon + (obstacle_polygon_x1[0] - x[i, 0]))
        m_sat.addConstr(0 <= M*(1 - qVarphi2[i]) - epsilon + (x[i, 0] - obstacle_polygon_x1[1]))
        m_sat.addConstr(0 <= M*(1 - qVarphi3[i]) - epsilon + (obstacle_polygon_x2[0] - x[i, 1]))
        m_sat.addConstr(0 <= M*(1 - qVarphi4[i]) - epsilon + (x[i, 1] - obstacle_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - qVarphi5[i]) - epsilon + (obstacle_polygon_x3[0] - x[i, 2]))
        # m.addConstr(0 <= M*(1 - qVarphi6[i]) - epsilon + (x[i, 2] - obstacle_polygon_x3[1]))

        #Avoid disjunctions
        m_sat.addConstr(1 <= qVarphi1[i] + qVarphi2[i] + qVarphi3[i] + qVarphi4[i]) # + qVarphi5[i] + qVarphi6[i])

        # #Reach predicates
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 0] - goal_A_polygon_x1[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 0] + goal_A_polygon_x1[1]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 1] - goal_A_polygon_x2[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 1] + goal_A_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 2] - goal_A_polygon_x3[0]))
        # m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (- x[i, 2] + goal_A_polygon_x3[1]))

        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 0] - goal_B_polygon_x1[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 0] + goal_B_polygon_x1[1]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 1] - goal_B_polygon_x2[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 1] + goal_B_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (x[i, 2] - goal_B_polygon_x3[0]))
        # m.addConstr(0 <= M*(1 - pPhi2[i]) - epsilon + (- x[i, 2] + goal_B_polygon_x3[1]))

        # #pPhi1_sum
        # pPhi1_sum.add(pPhi1[i])

        # #pPhi2_sum
        # pPhi2_sum.add(pPhi2[i])
    
    # m.addConstr(1 <= pPhi1_sum)
    # m.addConstr(1 <= pPhi2_sum)

    for i in range(0, N-surveillance_A_period+1):
        pPhi1_i = m_sat.addVars(surveillance_A_period, vtype=GRB.BINARY, name="pPhi1_{:d}".format(i))
        pPhi1_i_sum = gb.LinExpr()

        for j in range (surveillance_A_period):
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 0] - goal_A_polygon_x1[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 0] + goal_A_polygon_x1[1]))
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 1] - goal_A_polygon_x2[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 1] + goal_A_polygon_x2[1]))
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 2] - goal_A_polygon_x3[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 2] + goal_A_polygon_x3[1]))

            pPhi1_i_sum.add(pPhi1_i[j])

        m_sat.addConstr(1 <= pPhi1_i_sum)

    for i in range(0, N-surveillance_B_period+1):
        pPhi2_i = m_sat.addVars(surveillance_B_period, vtype=GRB.BINARY, name="pPhi2_{:d}".format(i))
        pPhi2_i_sum = gb.LinExpr()

        for j in range (surveillance_B_period):         
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 0] - goal_B_polygon_x1[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 0] + goal_B_polygon_x1[1]))
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 1] - goal_B_polygon_x2[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 1] + goal_B_polygon_x2[1]))
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 2] - goal_B_polygon_x3[0]))
            m_sat.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 2] + goal_B_polygon_x3[1]))

            pPhi2_i_sum.add(pPhi2_i[j])

        m_sat.addConstr(1 <= pPhi2_i_sum)

    m_sat.optimize()

    qVarphi1_sol = []
    qVarphi2_sol = []
    qVarphi3_sol = []
    qVarphi4_sol = []
    # qVarphi5_sol = []
    # qVarphi6_sol = []
    pPhi1_sol = []
    pPhi2_sol = []

    feasible = False
    solver_status = m_sat.getAttr('Status')
    if solver_status == GRB.OPTIMAL:
        feasible = True
        print("m results________________________________________________")
        for var in m_sat.getVars():
            print('%s %g' % (var.VarName, var.X))

        print("Runtime: " + str(m_sat.Runtime) + "s")

        for i in range(N):
            qVarphi1_sol_i = m_sat.getVarByName("qVarphi1[{:d}]".format(i))
            qVarphi1_sol.append(qVarphi1_sol_i.X)
            qVarphi2_sol_i = m_sat.getVarByName("qVarphi2[{:d}]".format(i))
            qVarphi2_sol.append(qVarphi2_sol_i.X) 
            qVarphi3_sol_i = m_sat.getVarByName("qVarphi3[{:d}]".format(i))
            qVarphi3_sol.append(qVarphi3_sol_i.X)
            qVarphi4_sol_i = m_sat.getVarByName("qVarphi4[{:d}]".format(i))
            qVarphi4_sol.append(qVarphi4_sol_i.X)
            # qVarphi5_sol_i = m.getVarByName("qVarphi5[{:d}]".format(i))
            # qVarphi5_sol.append(qVarphi5_sol_i.X)
            # qVarphi6_sol_i = m.getVarByName("qVarphi6[{:d}]".format(i))
            # qVarphi6_sol.append(qVarphi6_sol_i.X)
            
        for i in range(N-surveillance_A_period+1):
            pPhi1_sol_i = [] 
            for j in range(surveillance_A_period):
                pPhi1_sol_i_j = m_sat.getVarByName("pPhi1_{:d}[{:d}]".format(i, j))
                pPhi1_sol_i.append(pPhi1_sol_i_j.X)
            pPhi1_sol.append(pPhi1_sol_i)
        
        for i in range(N-surveillance_B_period+1):
            pPhi2_sol_i = [] 
            for j in range(surveillance_B_period):
                pPhi2_sol_i_j = m_sat.getVarByName("pPhi2_{:d}[{:d}]".format(i, j))
                pPhi2_sol_i.append(pPhi2_sol_i_j.X)
            pPhi2_sol.append(pPhi2_sol_i)

    return feasible, qVarphi1_sol, qVarphi2_sol, qVarphi3_sol, qVarphi4_sol, pPhi1_sol, pPhi2_sol



def runControlLoop(plot, x0, v0):
    sigma_x0 = [1e-6, 1e-6, 1e-6]
    sigma_v0 = [1e-6, 1e-6, 1e-6]

    runtimes = []
    
    theta_micp, phi_micp, thrust_micp, qVarphi1_micp, qVarphi2_micp, qVarphi3_micp, qVarphi4_micp, pPhi1_micp, pPhi2_micp = solveMICP(plot, x0, v0) #qVarphi5_micp, qVarphi6_micp, 

    isFeasible = False
    satSTL = False
    for i in range(0, N-1):
        runtime, theta_nlp, phi_nlp, thrust_nlp, isFeasible = solveNLP(False, x0, v0, sigma_x0, sigma_v0, theta_micp[i:], phi_micp[i:], thrust_micp[i:], qVarphi1_micp, qVarphi2_micp, qVarphi3_micp, qVarphi4_micp, pPhi1_micp, pPhi2_micp) #qVarphi5_micp, qVarphi6_micp,
        runtimes.append(runtime)

        if isFeasible == True:
            xv_next = simNextStep(True, x0[i], v0[i], np.array([theta_nlp[i], phi_nlp[i], thrust_nlp[i]]))
            x0 = np.concatenate([x0, [xv_next[:3]]], axis=0)
            v0 = np.concatenate([v0, [xv_next[3:]]], axis=0)
            print("NLP feasible at time step: ", str(i))
        else:
            print("NLP infeasible at time step: ", str(i))
            break

    if plot == True:
        plotTraj(x0[:,0], x0[:,1], x0[:,2])
    
    if isFeasible:
        satSTL, qVarphi1_sat, qVarphi2_sat, qVarphi3_sat, qVarphi4_sat, pPhi1_sat, pPhi2_sat = verifySTLSAT(x0, v0)
        
        if satSTL:
            print("STL satisfied")
            # print(qVarphi1_sat)
            # print(qVarphi2_sat)
            # print(qVarphi3_sat)
            # print(qVarphi4_sat)
            # print(pPhi1_sat)
            # print(pPhi2_sat)
        else:
            print("STL not satisfied")
    else: 
        satSTL = False

    print(x0)
    print("___")
    print(v0)

    avg_runtime = -1
    if len(runtimes) > 0:
        avg_runtime = sum(runtimes)/len(runtimes)
        print("Average runtime: ", avg_runtime)
    
    return satSTL, isFeasible, avg_runtime 

# x0 = np.array([[5., 2., 1.]])
# v0 = np.array([[0., 0., 0.]])

# runControlLoop(True, x0, v0)

singleTest = True
numTestIts = 100

if singleTest == True:
    numTestIts = 1

currIt = 0
# sigmax1Guess = np.zeros(N)
# sigmax2Guess = np.zeros(N)

feas_x0List = []
infeas_x0List = []
nlpInfeas_x0List = []
numNLPInfeas = 0

for i in range(numTestIts):
    currIt += 1
    x0_i = np.array([[5., 2., 1.]])
    if singleTest == False:
        x0_i = np.random.rand(3)*ubx[0]
        while (x0_i[0] >= obstacle_polygon_x1[0] - epsilon and x0_i[0] <= obstacle_polygon_x1[1] + epsilon and x0_i[1] >= obstacle_polygon_x2[0] + - epsilon and x0_i[1] <= obstacle_polygon_x2[1] + epsilon):
            x0_i = np.random.rand(3)*ubx[0]
        
        x0_i = np.array([x0_i])
    
    v0_i = np.array([[0., 0., 0.]])
    satisfiedSTL, NLPFeasOnTermination, avg_runtime = runControlLoop(singleTest, x0_i, v0_i)
    if satisfiedSTL:
        feas_x0List.append(x0_i)
    elif NLPFeasOnTermination:
        infeas_x0List.append(x0_i)
    else:
        nlpInfeas_x0List.append(x0_i)

    if NLPFeasOnTermination == False:
        numNLPInfeas += 1

    plt.show()
    
print("Total sat: ", len(feas_x0List))
print("Total unsat: ", len(infeas_x0List))
print("Total unsat + NLP infeas: ", len(nlpInfeas_x0List))
print("Total unsat due to infeas: ", str(numNLPInfeas))
print("Unsat list: ", infeas_x0List)
print("Unsat + infeas list: ", nlpInfeas_x0List)
