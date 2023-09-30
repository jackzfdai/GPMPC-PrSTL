import numpy as np
import torch
import math
import gpytorch
from scipy.stats import norm
from scipy.signal import cont2discrete
import gurobipy as gb
from gurobipy import GRB
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import *

from datetime import datetime

np_rng = np.random.default_rng(seed=0)

#GP Parameter Estimation
var_residual = 0.05
K_residual = 0.25

torch.manual_seed(0)

plot_gp_training = False

num_training_samples = 50

train_x1 = torch.rand(num_training_samples) * 10 #torch.linspace(0, 10, 5)
train_x2 = torch.rand(num_training_samples) * 10 #torch.linspace(0, 10, 5)
train_x3 = torch.rand(num_training_samples) * 10 - 5
train_x4 = torch.rand(num_training_samples) * 10 - 5
train_u1 = torch.rand(num_training_samples) - 0.5 #torch.linspace(-1, 1, 5)
train_u2 = torch.rand(num_training_samples) - 0.5

train_inputs = torch.stack([train_x1, train_x2, train_x3, train_x4, train_u1, train_u2], dim=0)
train_inputs = torch.transpose(train_inputs, dim0=0, dim1=1)
# print(train_inputs)
A1 = 0.05
A2 = 0.05
Sigma_w = 0.01
train_g1 = K_residual*torch.sin(train_inputs[:,0]) + torch.randn(train_inputs[:,0].size()) * math.sqrt(var_residual)
train_g2 = K_residual*torch.sin(train_inputs[:,1]) + torch.randn(train_inputs[:,1].size()) * math.sqrt(var_residual)

actual_g1 = K_residual*torch.sin(train_inputs[:,0]) 

# We will use the simplest form of GP model, exact inference
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
likelihood_g1 = gpytorch.likelihoods.GaussianLikelihood()
likelihood_g2 = gpytorch.likelihoods.GaussianLikelihood()
model_g1 = ExactGPModel(train_inputs, train_g1, likelihood_g1)
model_g2 = ExactGPModel(train_inputs, train_g2, likelihood_g2)

# Find optimal model hyperparameters
model_g1.train()
model_g2.train()
likelihood_g1.train()
likelihood_g2.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model_g1.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_g1, model_g1)

training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model_g1(train_inputs)
    # Calc loss and backprop gradients
    loss = -mll(output, train_g1)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_g1.covar_module.base_kernel.lengthscale.item(),
        model_g1.likelihood.noise.item()
    ))
    optimizer.step()

optimizer = torch.optim.Adam(model_g2.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood_g2, model_g2)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model_g2(train_inputs)
    # Calc loss and backprop gradients
    loss = -mll(output, train_g2)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model_g2.covar_module.base_kernel.lengthscale.item(),
        model_g2.likelihood.noise.item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model_g1.eval()
likelihood_g1.eval()
test_out = model_g1.covar_module(train_inputs)
test_out_dense = test_out.to_dense()
print(test_out_dense.size()[1])
print(model_g1.covar_module.base_kernel.lengthscale)
print(model_g1.covar_module.outputscale)
for param_name, param in model_g1.covar_module.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')
# print(model_g1.likelihood.hyperparameters)
# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x1 = torch.linspace(0, 10, 50)
    test_x2 = torch.linspace(0, 10, 50)
    test_x3 = torch.linspace(-5, 5, 50)
    test_x4 = torch.linspace(-5, 5, 50)
    test_u1 = torch.linspace(-0.5, 0.5, 50)
    test_u2 = torch.linspace(-0.5, 0.5, 50)
    test_input = torch.stack([test_x1, test_x2, test_x3, test_x4, test_u1, test_u2], dim=0)
    test_input = torch.transpose(test_input, dim0=0, dim1=1)
    observed_pred = likelihood_g1(model_g1(test_input))

if plot_gp_training == True:
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_inputs[:,0].numpy(), train_g1.numpy(), 'k*')
        ax.plot(train_inputs[:,0].numpy(), actual_g1.numpy(), 'k+')
        # Plot predictive means as blue line
        ax.plot(test_input[:,0].numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_input[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-0.3, 0.3])
        ax.legend(['Observed Data', 'Actual Function Samples','Mean', 'Confidence'])
        # plt.show()

KzzPrior1 = model_g1.covar_module(train_inputs).to_dense()
print("_________1___________")
print(KzzPrior1)
dMu = torch.matmul(torch.linalg.inv(KzzPrior1 + model_g1.likelihood.noise.item()*torch.eye(KzzPrior1.size()[0])), train_g1) 
print(dMu)
print(dMu.size())
# dSigma = torch.linalg.inv(KzzPrior1 + model_g1.covar_module.outputscale*torch.eye(KzzPrior1.size()[0]))
dSigma = torch.linalg.inv(KzzPrior1 + model_g1.likelihood.noise.item()*torch.eye(KzzPrior1.size()[0]))

KzzPrior2 = model_g2.covar_module(train_inputs).to_dense()
dMu2 = torch.matmul(torch.linalg.inv(KzzPrior2 + model_g2.likelihood.noise.item()*torch.eye(KzzPrior2.size()[0])), train_g2) 
# dSigma2 = torch.linalg.inv(KzzPrior2 + model_g2.covar_module.outputscale*torch.eye(KzzPrior2.size()[0]))
dSigma2 = torch.linalg.inv(KzzPrior2 + model_g2.likelihood.noise.item()*torch.eye(KzzPrior2.size()[0]))
print(dSigma2)

test_x1_new = torch.linspace(0, 10, 50)
test_x2_new = torch.linspace(0, 10, 50)
test_x3_new = torch.linspace(-5, 5, 50)
test_x4_new = torch.linspace(-5, 5, 50)
test_u1_new = torch.linspace(-0.5, 0.5, 50)
test_u2_new = torch.linspace(-0.5, 0.5, 50)
test_inputs = torch.stack([test_x1_new, test_x2_new, test_x3_new, test_x4_new, test_u1_new, test_u2_new], dim=0)
test_inputs = torch.transpose(test_inputs, dim0=0, dim1=1)
# dMu_tile = dMu.tile((50, 1))
# print(dMu_tile.size())
# print(test_x1_new.size())
# test_x1_mu = torch.matmul(dMu_tile, test_x1_new)
# print(test_x1_mu)
l = model_g1.covar_module.base_kernel.lengthscale
sigma_1 = model_g1.covar_module.outputscale
sigma_2 = model_g2.covar_module.outputscale
l2 = model_g2.covar_module.base_kernel.lengthscale

resultingMus = []

#debug usage
resultingMusTest = []
for i in range(50):
    test_row_i = torch.tile(test_inputs[i], (num_training_samples, 1))
    norms = torch.linalg.norm(train_inputs - test_row_i, dim=1)
    Kzz_row_i = sigma_1*torch.exp((-1.0/(2.0**(l**2)))*torch.square(norms))
    resultingMu = torch.matmul(Kzz_row_i, dMu)
    resultingMusTest.append(resultingMu.item())
print("test")
print(resultingMusTest)
#end debug usage


for i in range(50):
    list_k = torch.tensor([])
    for j in range(num_training_samples):
        k_j = sigma_1*math.exp((-1/(2*(l**2)))*((test_x1_new[i] - train_x1[j])**2 + (test_x2_new[i]-train_x2[j])**2 + (test_u1_new[i] - train_u1[j])**2 + (test_u2_new[i] - train_u2[j])**2))
        list_k = torch.cat((list_k, torch.tensor([k_j])), 0)
    print(list_k)
    resultingMu = torch.matmul(list_k, dMu)
    resultingSigma = torch.tensor([sigma_1]) - torch.matmul(list_k, torch.matmul(dSigma, list_k))
    print(resultingSigma)
    print(resultingMu)
    resultingMus.append(resultingMu.item())
    print(resultingMus)

diff = np.linalg.norm(observed_pred.mean.numpy() - np.asarray(resultingMus))
print("diff is: ", diff)
print(model_g1.likelihood.noise.item())
print("original")
print(observed_pred.mean.numpy() )
print("manual")
print(np.asarray(resultingMus))


inputs_row_0 = torch.tile(train_inputs[0], (num_training_samples, 1))
norms = torch.linalg.norm(train_inputs - inputs_row_0, dim=1)
Kzz_row_0 = sigma_1*torch.exp((-1.0/(2.0**(l**2)))*torch.square(norms))
# print("_________2___________")
# print(Kzz_row_0)

if plot_gp_training == True:
    f1, ax1 = plt.subplots(1, 1, figsize=(4, 3))

    # Plot predictive means as blue line
    ax1.plot(train_inputs[:,0].numpy(), train_g1.numpy(), 'k*')
    ax1.plot(train_inputs[:,0].numpy(), actual_g1.numpy(), 'k+')
    ax1.plot(test_input[:,0].numpy(), resultingMus, 'b')
    # Shade between the lower and upper confidence bounds
    ax1.set_ylim([-0.3, 0.3])
    ax1.legend(['Observed Data', 'Actual Function Sample','Mean', 'Confidence'])
    plt.show()

def calcGPMu_x1(x1, x2, u1, u2):
    x1 = np.array(x1).item()
    x2 = np.array(x1).item()
    u1 = np.array(u1).item()
    u2 = np.array(u2).item()

    list_k = torch.tensor([])
    for j in range(num_training_samples):
        k_j = sigma_1*math.exp((-1/(2*(l**2)))*((torch.tensor([x1]) - train_x1[j])**2 + (torch.tensor([x2]) - train_x2[j])**2 + (torch.tensor([u1]) - train_u1[j])**2 + (torch.tensor([u2]) - train_u2[j])**2))
        list_k = torch.cat((list_k, torch.tensor([k_j])), 0)
    print(list_k)
    resultingMu_x1 = torch.matmul(list_k, dMu)
    # resultingSigma = torch.tensor([sigma_1]) - torch.matmul(list_k, torch.matmul(dSigma, list_k))
    print(resultingMu_x1)
    return resultingMu_x1.item()


# def calcGPMu_x2(x1, x2, u1, u2):
#     x1 = np.array(x1).item()
#     x2 = np.array(x1).item()
#     u1 = np.array(u1).item()
#     u2 = np.array(u2).item()

#     list_k = torch.tensor([])
#     for j in range(num_training_samples):
#         k_j = sigma_2*math.exp((-1/(2*(l**2)))*((torch.tensor([x1]) - train_x1[j])**2 + (torch.tensor([x2]) - train_x2[j])**2 + (torch.tensor([u1]) - train_u1[j])**2 + (torch.tensor([u2]) - train_u2[j])**2))
#         list_k = torch.cat((list_k, torch.tensor([k_j])), 0)
#     print(list_k)
#     resultingMu_x2 = torch.matmul(list_k, dMu2)
#     # resultingSigma = torch.tensor([sigma_1]) - torch.matmul(list_k, torch.matmul(dSigma, list_k))
#     print(resultingMu_x2)
#     return resultingMu_x2.item()

def calcGPSigma_x1(x1, x2, x3, x4, u1, u2):
    list_k = torch.tensor([])
    for j in range(num_training_samples):
        k_j = sigma_1*math.exp((-1/(2*(l**2)))*((x1 - train_x1[j])**2 + (x2 - train_x2[j])**2 + (x3 - train_x3[j])**2 + (x4 - train_x4[j])**2 + (u1 - train_u1[j])**2 + (u2 - train_u2[j])**2))
        list_k = torch.cat((list_k, torch.tensor([k_j])), 0)
    print(list_k)
    # resultingMu_x1 = torch.matmul(list_k, dMu)
    resultingSigma_x1 = torch.tensor([sigma_1]) - torch.matmul(list_k, torch.matmul(dSigma, list_k))
    print(resultingSigma_x1)
    return resultingSigma_x1

# def calcGPSigma_x2(x1, x2, u1, u2):
#     list_k = torch.tensor([])
#     for j in range(num_training_samples):
#         k_j = sigma_2*math.exp((-1/(2*(l**2)))*((x1 - train_x1[j])**2 + (x2 - train_x2[j])**2 + (u1 - train_u1[j])**2 + (u2 - train_u2[j])**2))
#         list_k = torch.cat((list_k, torch.tensor([k_j])), 0)
#     print(list_k)
#     # resultingMu_x1 = torch.matmul(list_k, dMu)
#     resultingSigma_x2 = torch.tensor([sigma_2]) - torch.matmul(list_k, torch.matmul(dSigma2, list_k))
#     print(resultingSigma_x2)
#     return resultingSigma_x2

#Control parameters
T = 3 #Time horizon
dt = 0.1
N = int(T/dt)
epsilon = 0.0
M = N
invCDFVarphiEpsilon = norm.ppf(0.15)

#System Parameters
Ac = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [0, 0, 0, 0], 
              [0, 0, 0, 0]])
Bc = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])

C = np.zeros((4, 4))
D = np.zeros((4, 2))

sysd = cont2discrete((Ac, Bc, C, D), dt=dt)
A = sysd[0]
B = sysd[1]
print(A)
print(B)

#Residual parameters:
Bd = np.concatenate([1/2*(dt**2)*np.eye(2), dt*np.eye(2)], axis=0)

#Variable bounds
lbx = [0., 0., -5., -5.]
ubx = [10., 10., 5., 5.]

lbu = [-2.5, -2.5]
ubu = [2.5, 2.5]

lbsigma = [1e-6, 1e-6, 1e-6, 1e-6]
ubsigma = [10., 10., 10., 10.]

lbstddev = [1e-6, 1e-6, 1e-6, 1e-6]
ubstddev = [10., 10., 10., 10.]

x_lb = lbx[0] #Split into x1, x2 if needed later 
x_ub = ubx[0]
v_lb = lbx[2]
v_ub = ubx[2]
u_lb = lbu[0]
u_ub = ubu[0]

#Reach avoid parameters
goal_A_polygon_x1 = [7, 8.5]
goal_A_polygon_x2 = [3, 4.5] 
obstacle_polygon_x1 = [2, 6]
obstacle_polygon_x2 = [2, 6]

def solveMILP(plot, x0, v0):
    #Problem at time step k = 0 ____________________________________________
    m = gb.Model('integratorRobot2D')

    #Cost function (zero objective since we want to test feasibility)
    # zeroObjective = gb.LinExpr(0)
    # m.setObjective(zeroObjective)

    #Set x value to corresponding value in sequence
    x = m.addVars(N, 2, lb=x_lb, ub=x_ub, name="x")
    v = m.addVars(N, 2, lb=v_lb, ub=v_ub, name="v")
    u = m.addVars(N-1-(len(x0) - 1), 2, lb=u_lb, ub=u_ub, name="u")

    for i in range(len(x0)):
        m.addConstr(x[i, 0] == x0[i, 0])
        m.addConstr(x[i, 1] == x0[i, 1])
        m.addConstr(v[i, 0] == v0[i, 0])
        m.addConstr(v[i, 1] == v0[i, 1])
        
    objective = gb.QuadExpr()
    zero_objective = gb.LinExpr(0)
    for i in range(N-1-(len(x0) - 1)):
        objective_i = gb.QuadExpr(u[i, 0] ** 2 + u[i, 1] ** 2)
        objective.add(objective_i)

    m.setObjective(objective)

    #variables 
    qVarphi1 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi1")
    qVarphi2 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi2")
    qVarphi3 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi3")
    qVarphi4 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi4")
    pPhi1 = m.addVars(N, vtype=GRB.BINARY, name="pPhi1")

    for i in range(len(x0) - 1, N-1):
        m.addConstr(x[i+1, 0] == A[0, 0]*x[i, 0] + A[0, 1]*x[i, 1] + A[0, 2]*v[i, 0] + A[0, 3]*v[i, 1] + B[0, 0]*u[i-(len(x0) - 1), 0] + B[0, 1]*u[i-(len(x0) - 1), 1])
        m.addConstr(x[i+1, 1] == A[1, 0]*x[i, 0] + A[1, 1]*x[i, 1] + A[1, 2]*v[i, 0] + A[1, 3]*v[i, 1] + B[1, 0]*u[i-(len(x0) - 1), 0] + B[1, 1]*u[i-(len(x0) - 1), 1])
        m.addConstr(v[i+1, 0] == A[2, 0]*x[i, 0] + A[2, 1]*x[i, 1] + A[2, 2]*v[i, 0] + A[2, 3]*v[i, 1] + B[2, 0]*u[i-(len(x0) - 1), 0] + B[2, 1]*u[i-(len(x0) - 1), 1])
        m.addConstr(v[i+1, 1] == A[3, 0]*x[i, 0] + A[3, 1]*x[i, 1] + A[3, 2]*v[i, 0] + A[3, 3]*v[i, 1] + B[3, 0]*u[i-(len(x0) - 1), 0] + B[3, 1]*u[i-(len(x0) - 1), 1])

    pPhi1_sum = gb.LinExpr()

    for i in range(0, N):
        #Avoid predicates
        m.addConstr(0 <= M*(1 - qVarphi1[i]) - epsilon + (obstacle_polygon_x1[0] - x[i, 0]))
        m.addConstr(0 <= M*(1 - qVarphi2[i]) - epsilon + (x[i, 0] - obstacle_polygon_x1[1]))
        m.addConstr(0 <= M*(1 - qVarphi3[i]) - epsilon + (obstacle_polygon_x2[0] - x[i, 1]))
        m.addConstr(0 <= M*(1 - qVarphi4[i]) - epsilon + (x[i, 1] - obstacle_polygon_x2[1]))

        #Reach predicates
        m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 0] - goal_A_polygon_x1[0]))
        m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (goal_A_polygon_x1[1] - x[i, 0]))
        m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (x[i, 1] - goal_A_polygon_x2[0]))
        m.addConstr(0 <= M*(1 - pPhi1[i]) - epsilon + (goal_A_polygon_x2[1] - x[i, 1]))

        #Avoid disjunctions
        m.addConstr(1 <= qVarphi1[i] + qVarphi2[i] + qVarphi3[i] + qVarphi4[i])

        #pPhi1_sum
        pPhi1_sum.add(pPhi1[i])

    m.addConstr(1 <= pPhi1_sum)

    m.optimize()

    x1_sol = []
    x2_sol = []
    u1_sol = []
    u2_sol = []
    qVarphi1_sol = []
    qVarphi2_sol = []
    qVarphi3_sol = []
    qVarphi4_sol = []
    pPhi1_sol = []

    solver_status = m.getAttr('Status')
    solver_runtime = m.getAttr('Runtime')
    if solver_status == GRB.OPTIMAL:
        print("m results________________________________________________")
        for v in m.getVars():
            print('%s %g' % (v.VarName, v.X))

        print("Runtime: " + str(m.Runtime) + "s")

        for i in range(N):
            x1_sol_i = m.getVarByName("x[{:d},0]".format(i))
            x2_sol_i = m.getVarByName("x[{:d},1]".format(i))
            x1_sol.append(x1_sol_i.X)
            x2_sol.append(x2_sol_i.X)
            qVarphi1_sol_i = m.getVarByName("qVarphi1[{:d}]".format(i))
            qVarphi1_sol.append(qVarphi1_sol_i.X)
            qVarphi2_sol_i = m.getVarByName("qVarphi2[{:d}]".format(i))
            qVarphi2_sol.append(qVarphi2_sol_i.X) 
            qVarphi3_sol_i = m.getVarByName("qVarphi3[{:d}]".format(i))
            qVarphi3_sol.append(qVarphi3_sol_i.X)
            qVarphi4_sol_i = m.getVarByName("qVarphi4[{:d}]".format(i))
            qVarphi4_sol.append(qVarphi4_sol_i.X)
            pPhi1_sol_i = m.getVarByName("pPhi1[{:d}]".format(i))
            pPhi1_sol.append(pPhi1_sol_i.X)
            # zLambda1_sol_i = m.getVarByName("zLambda1[{:d}]".format(i))
            # zLambda1_sol.append(zLambda1_sol_i.X)
            # zLambda2_sol_i = m.getVarByName("zLambda2[{:d}]".format(i))
            # zLambda2_sol.append(zLambda2_sol_i.X)


        for i in range(N-1-(len(x0) - 1)):
            u1_sol_i = m.getVarByName("u[{:d},0]".format(i))
            u2_sol_i = m.getVarByName("u[{:d},1]".format(i))
            u1_sol.append(u1_sol_i.X)
            u2_sol.append(u2_sol_i.X)
            
        if plot == True:
            plotTraj(x1_sol, x2_sol)
            objective_val = m.getAttr('ObjVal')
            print("Open loop objective: ", str(objective_val))

    return solver_runtime, u1_sol, u2_sol, qVarphi1_sol, qVarphi2_sol, qVarphi3_sol, qVarphi4_sol, pPhi1_sol

def solveNLP(plot, x0, v0, sigma_x0, sigma_v0, u1_milp, u2_milp, qVarphi1_milp, qVarphi2_milp, qVarphi3_milp, qVarphi4_milp, pPhi1_milp):
    feasible = False

    #Match control horizon indexing
    local_N = N - 1

    # Declare model variables
    x = SX.sym('x', 4)
    sigma = SX.sym('sigma', 4)
    u = SX.sym('u', 2)

    # Model equations
    x_next = vertcat(dot(A[0, :], x) + dot(B[0, :], u) + Bd[0, 0]*(sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu.detach().numpy())),
                    dot(A[1, :], x) + dot(B[1, :], u) + Bd[1, 1]*(sigma_2.item()*exp((-1/(2*(l2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu2.detach().numpy())),
                    dot(A[2, :], x) + dot(B[2, :], u) + Bd[2, 0]*(sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu.detach().numpy())),
                    dot(A[3, :], x) + dot(B[3, :], u) + Bd[3, 1]*(sigma_2.item()*exp((-1/(2*(l2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T)) @ DM(dMu2.detach().numpy())))

    # x_next = vertcat(dot(A[0, :], x) + dot(B[0, :], u) + Bd[0, 0]*K_residual*sin(x[0]) + Bd[0, 1]*K_residual*sin(x[1]),
    #                 dot(A[1, :], x) + dot(B[1, :], u) + Bd[1, 0]*K_residual*sin(x[0]) + Bd[1, 1]*K_residual*sin(x[1]), 
    #                 dot(A[2, :], x) + dot(B[2, :], u) + Bd[2, 0]*K_residual*sin(x[0]) + Bd[2, 1]*K_residual*sin(x[1]),
    #                 dot(A[3, :], x) + dot(B[3, :], u) + Bd[3, 0]*K_residual*sin(x[0]) + Bd[3, 1]*K_residual*sin(x[1]))

    sigma_next = vertcat(dot(A[0, :], sigma) + Bd[0, 0]*(sigma_1.item() - (sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma.detach().numpy()) @ (sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[1, :], sigma) + Bd[1, 1]*(sigma_2.item() - (sigma_2.item()*exp((-1/(2*(l2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma2.detach().numpy()) @ (sigma_2.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[2, :], sigma) + Bd[2, 0]*(sigma_1.item() - (sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma.detach().numpy()) @ (sigma_1.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T),
                         dot(A[3, :], sigma) + Bd[3, 1]*(sigma_2.item() - (sigma_2.item()*exp((-1/(2*(l2.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))) @ DM(dSigma2.detach().numpy()) @ (sigma_2.item()*exp((-1/(2*(l.item()**2)))*(sum2((repmat(horzcat(x[0], x[1], x[2], x[3], u[0], u[1]), num_training_samples, 1) - DM(train_inputs.detach().numpy()))**2).T))).T))

    # sigma_d = np.array([math.sqrt(var_residual), math.sqrt(var_residual)])

    # sigma_next = vertcat(dot(A[0, :], sigma) + np.dot(Bd[0, :], sigma_d),
    #                      dot(A[1, :], sigma) + np.dot(Bd[1, :], sigma_d),
    #                      dot(A[2, :], sigma) + np.dot(Bd[2, :], sigma_d),
    #                      dot(A[3, :], sigma) + np.dot(Bd[3, :], sigma_d))

    # Objective term
    L = u[0] ** 2 + u[1] ** 2

    F = Function('F', [x, sigma, u], [x_next, sigma_next, L], ['x_k', 'sigma_k', 'u_k'], ['x_k+1', 'sigma_k+1', 'L'])

    # Initial guess for u
    u_start = [] 
    for i in range(local_N-(len(x0) - 1)):
        u_start.append(DM([u1_milp[i], u2_milp[i]]))
    
    # DM([u1_milp[k], u2_milp[k]])
    # Get a feasible trajectory as an initial guess
    x_start = []
    sigma_start = []
    stddev_start = []
    for i in range(len(x0)):
        xk = DM(np.concatenate([x0[i], v0[i]], axis=0))
        x_start += [xk]
        sigmak = DM(np.concatenate([sigma_x0, sigma_v0], axis=0)) #assuming sigma_x0, v0 is always 0
        sigma_start += [sigmak]
        stddev_k = sqrt(sigmak)
        stddev_start += [stddev_k]

    for k in range(local_N-(len(x0) - 1)):
        xk = F(x_k=xk, sigma_k=sigmak, u_k=u_start[k])['x_k+1']
        x_start += [xk]
        sigmak = F(x_k=xk, sigma_k=sigmak, u_k=u_start[k])['sigma_k+1']
        sigma_start += [sigmak]
        stddev_k = sqrt(sigmak)
        stddev_start += [stddev_k]

    # Start with an empty NLP
    w=[]
    w0 = []
    lbw = []
    ubw = []
    discrete = []
    J = 0
    g=[]
    lbg = []
    ubg = []

# "Lift" initial conditions
    for i in range(len(x0)):
        X0_i = MX.sym('X0_' + str(i), 4)
        w += [X0_i]
        lbw += [x0[i, 0], x0[i, 1], v0[i, 0], v0[i, 1]]
        ubw += [x0[i, 0], x0[i, 1], v0[i, 0], v0[i, 1]]
        w0 += [x_start[i]]
        discrete += [False, False, False, False]

        SIGMA0_i = MX.sym('sigma0_' + str(i), 4)
        w += [SIGMA0_i]
        lbw += [sigma_x0[0], sigma_x0[1], sigma_v0[0], sigma_v0[1]]
        ubw += [sigma_x0[0], sigma_x0[1], sigma_v0[0], sigma_v0[1]]
        w0 += [sigma_start[i]]
        discrete += [False, False, False, False]

        STDDEV0_i = MX.sym('stddev0_' + str(i), 4)
        w += [STDDEV0_i]
        lbw += [np.sqrt(sigma_x0[0]), np.sqrt(sigma_x0[1]), np.sqrt(sigma_v0[0]), np.sqrt(sigma_v0[1])]
        ubw += [np.sqrt(sigma_x0[0]), np.sqrt(sigma_x0[1]), np.sqrt(sigma_v0[0]), np.sqrt(sigma_v0[1])]
        w0 += [stddev_start[i]]
        discrete += [False, False, False, False]

        #dummy U variables
        if i < len(x0) - 1:
            Uk_i = MX.sym('U_' + str(i), 2)
            w   += [Uk_i]
            lbw += [0, 0]
            ubw += [0, 0]
            w0  += [DM(np.array([0, 0]))]
            discrete += [False, False]

        # Formulate the NLP
        Xk = X0_i
        Sigmak = SIGMA0_i
        Stddevk = STDDEV0_i

    for k in range(local_N - (len(x0) - 1)):
        # New NLP variable for the control
        Uk = MX.sym('U_' + str(k + len(x0) - 1), 2)
        w   += [Uk]
        lbw += lbu
        ubw += ubu
        w0  += [u_start[k]]
        discrete += [False, False]

        # Integrate till the end of the interval
        Fk = F(x_k=Xk, sigma_k = Sigmak, u_k=Uk)
        Xk_end = Fk['x_k+1']
        Sigmak_end = Fk['sigma_k+1']
        J=J+Fk['L']

        # New NLP variable for state at end of interval
        Xk = MX.sym('X_' + str((len(x0) - 1) + k + 1), 4)
        w   += [Xk]
        lbw += lbx
        ubw += ubx
        w0  += [x_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False]

        Sigmak = MX.sym('Sigma_' + str((len(x0) - 1) + k + 1), 4)
        w += [Sigmak]
        lbw += lbsigma
        ubw += ubsigma
        w0 += [sigma_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False]

        Stddevk = MX.sym('Stddev_' + str((len(x0) - 1) + k + 1), 4)
        w += [Stddevk]
        lbw += lbstddev
        ubw += ubstddev
        w0 += [stddev_start[k+1+(len(x0) - 1)]]
        discrete += [False, False, False, False]

        # Add equality constraint
        g   += [Xk_end-Xk]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        g += [Sigmak_end - Sigmak]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

        g += [Stddevk - sqrt(Sigmak)]
        lbg += [0, 0, 0, 0]
        ubg += [0, 0, 0, 0]

    #STL constraints
    for i in range(local_N + 1):
        g += [M*(1 - qVarphi1_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][0] + (- w[0 + 4*i][0] + obstacle_polygon_x1[0])]
        lbg += [0]
        ubg += [2*M]

    for i in range(local_N + 1):
        g += [M*(1 - qVarphi2_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][0] + (w[0 + 4*i][0] - obstacle_polygon_x1[1])]
        lbg += [0]
        ubg += [2*M]

    for i in range(local_N + 1):
        g += [M*(1 - qVarphi3_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][1] + (- w[0 + 4*i][1] + obstacle_polygon_x2[0])]
        lbg += [0]
        ubg += [2*M]

    for i in range(local_N + 1):
        g += [M*(1 - qVarphi4_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][1] + (w[0 + 4*i][1] - obstacle_polygon_x2[1])]
        lbg += [0]
        ubg += [2*M]

    for i in range(local_N + 1):
        g += [M*(1 - pPhi1_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][0] + (w[0 + 4*i][0] - goal_A_polygon_x1[0])]
        g += [M*(1 - pPhi1_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][0] + (- w[0 + 4*i][0] + goal_A_polygon_x1[1])]
        g += [M*(1 - pPhi1_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][1] + (w[0 + 4*i][1] - goal_A_polygon_x2[0])]
        g += [M*(1 - pPhi1_milp[i]) - epsilon + invCDFVarphiEpsilon * w[2 + 4*i][1] + (- w[0 + 4*i][1] + goal_A_polygon_x2[1])]
        lbg += [0, 0, 0, 0]
        ubg += [2*M, 2*M, 2*M, 2*M]

    # Concatenate decision variables and constraint terms
    w = vertcat(*w)
    g = vertcat(*g)

    # Create an NLP solver
    nlp_prob = {'f': J, 'x': w, 'g': g}
    nlp_solver = nlpsol('nlp_solver', 'bonmin', nlp_prob, {"discrete": discrete});
    # nlp_solver = nlpsol('nlp_solver', 'knitro', nlp_prob, {"discrete": discrete});
    # nlp_solver = nlpsol('nlp_solver', 'ipopt', nlp_prob); # Solve relaxed problem

    # Plot the solution
    tgrid = [k for k in range(local_N+1)]

    # Solve the NLP
    sol = nlp_solver(x0=vertcat(*w0), lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)

    print(nlp_solver.stats())

    solver_stats = nlp_solver.stats()
    proc_runtime = solver_stats['t_proc_total']

    w1_opt = sol['x']
    lam_w_opt = sol['lam_x']
    lam_g_opt = sol['lam_g']

    u1_ret = 0
    u2_ret = 0
    x1_ret = np.array([])
    x2_ret = np.array([])
    
    feasible = np.any(w1_opt.full().flatten())
    if feasible == True:
        w1_opt = w1_opt.full().flatten()
        x1_opt = w1_opt[0:14*(N-1)+12][0::14]
        x2_opt = w1_opt[0:14*(N-1)+12][1::14]
        v1_opt = w1_opt[0:14*(N-1)+12][2::14]
        v2_opt = w1_opt[0:14*(N-1)+12][3::14]
        u1_opt = w1_opt[0:14*(N-1)+12][12::14]
        u2_opt = w1_opt[0:14*(N-1)+12][13::14]
        sigma1_opt = w1_opt[0:14*(N-1)+12][4::14]
        sigma2_opt = w1_opt[0:14*(N-1)+12][5::14]

        if plot == True:
            plotSol(x1_opt, x2_opt, v1_opt, v2_opt, u1_opt, u2_opt)
            plotTraj(x1_opt, x2_opt)
    
        u1_ret = u1_opt[len(x0) - 1]
        u2_ret = u2_opt[len(x0) - 1]
        x1_ret = x1_opt
        x2_ret = x2_opt
    return proc_runtime, feasible, u1_ret, u2_ret, x1_ret, x2_ret

def plotTraj(x1_sol, x2_sol, x1_openloops = [], x2_openloops = []):
    goal_A_polygon_x1_plot = [goal_A_polygon_x1[0], goal_A_polygon_x1[0], goal_A_polygon_x1[1], goal_A_polygon_x1[1]]
    goal_A_polygon_x2_plot = [goal_A_polygon_x2[1], goal_A_polygon_x2[0], goal_A_polygon_x2[0], goal_A_polygon_x2[1]] 
    obstacle_polygon_x1_plot = [obstacle_polygon_x1[0], obstacle_polygon_x1[0], obstacle_polygon_x1[1], obstacle_polygon_x1[1]]
    obstacle_polygon_x2_plot = [obstacle_polygon_x2[1], obstacle_polygon_x2[0], obstacle_polygon_x2[0], obstacle_polygon_x2[1]]
    plt.figure()
    ax = plt.gca()
    plt.xlim(lbx[0], ubx[0])
    plt.ylim(lbx[1], ubx[1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.fill(goal_A_polygon_x1_plot, goal_A_polygon_x2_plot, 'g', alpha=0.5)
    plt.plot(goal_A_polygon_x1_plot + [goal_A_polygon_x1[0]], goal_A_polygon_x2_plot + [goal_A_polygon_x2[1]], 'g')
    # plt.fill(goal_B_polygon_x1, goal_B_polygon_x2, 'g')
    plt.fill(obstacle_polygon_x1_plot, obstacle_polygon_x2_plot, 'r', alpha=0.5)
    plt.plot(obstacle_polygon_x1_plot + [obstacle_polygon_x1[0]], obstacle_polygon_x2_plot + [obstacle_polygon_x2[1]], 'r')
    plt.scatter(x1_sol[0], x2_sol[0], s=120, facecolors='none', edgecolors='black')
    ax.set_aspect('equal')
    plt.plot(x1_sol, x2_sol, '-o')
    for i in range(len(x1_openloops)):
        if i % 5 == 0: #plot open loops only 1 every 5 control intervals for clarity 
            plt.plot(x1_openloops[i], x2_openloops[i], alpha=0.5, linestyle='-', linewidth=1)
    plt.grid(True)

def plotSol(x1_sol, x2_sol, v1_sol, v2_sol, u1_sol, u2_sol):
    tgrid = [k for k in range(N)]
    plt.figure()
    plt.plot(tgrid, x1_sol, '-')
    plt.plot(tgrid, x2_sol, '--')
    plt.plot(tgrid, v1_sol, '-')
    plt.plot(tgrid, v2_sol, '--')
    plt.step(tgrid, vertcat(DM.nan(1), u1_sol), '.')
    plt.step(tgrid, vertcat(DM.nan(1), u2_sol), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2', 'v1', 'v2', 'u1', 'u2'])
    plt.grid(True)

def simNextStep(uncertainty, x, v, u):
    if uncertainty == True:
        state = np.concatenate([x, v])
        mu_udx = K_residual*math.sin(x[0])  
        mu_udy = K_residual*math.sin(x[1])
        sigma_ud = math.sqrt(var_residual)
        udx = np_rng.normal(mu_udx, sigma_ud, 1)
        udy = np_rng.normal(mu_udy, sigma_ud, 1)
        vd = np.concatenate([udx, udy])
        state = np.concatenate([x, v])
        state_nominal = (A @ state) + (B @ u) #comparison purposes
        state_next = (A @ state) + (B @ u) + (Bd @ vd)
    else:
        state = np.concatenate([x, v])
        state_next = (A @ state) + (B @ u)
    return state_next 

def runControlLoop(plot, x0, v0):
    stl_met = True
    runtimes = []
    sigmax0 = np.array([1e-6, 1e-6])
    sigmav0 = np.array([1e-6, 1e-6])
    obj_cl_sum = 0
    x1_openloop_sol = []
    x2_openloop_sol = []

    _, milp_u1, milp_u2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol = solveMILP(plot, x0, v0)
    _, feasible, nlp_u1, nlp_u2, nlp_x1, nlp_x2 = solveNLP(plot, x0, v0, sigmax0, sigmav0, milp_u1, milp_u2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol)

    for i in range(0, N-1):
        milp_runtime, milp_u1, milp_u2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol = solveMILP(False, x0, v0)
        if len(milp_u1) > 0:
            print("MICP feasible")

            nlp_runtime, feasible, nlp_u1, nlp_u2, nlp_x1, nlp_x2 = solveNLP(False, x0, v0, sigmax0, sigmav0, milp_u1, milp_u2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol)
            runtimes.append(milp_runtime + nlp_runtime)

            if feasible:
                xv_next = simNextStep(True, x0[i], v0[i], np.array([nlp_u1, nlp_u2]))
                x0 = np.concatenate([x0, [xv_next[:2]]], axis=0)
                v0 = np.concatenate([v0, [xv_next[2:]]], axis=0)
                obj_cl_sum += nlp_u1**2 + nlp_u2**2
                x1_openloop_sol.append(nlp_x1)
                x2_openloop_sol.append(nlp_x2)
            else:
                print("NLP infeasible at time step: ", str(i))
                stl_met = False
                break
        else:
            print("MICP infeasible at time step: ", str(i))
            runtimes.append(milp_runtime)
            stl_met = False
            break
        
    if plot == True:
        plotTraj(x0[:,0], x0[:,1], x1_openloop_sol, x2_openloop_sol)
        plt.show()
    
    print(x0)
    print("____")
    print(v0)
    print("Closed loop objective", str(obj_cl_sum))
    avg_runtime = -1
    if len(runtimes) > 0:
        print("Average runtime: ", str(sum(runtimes)/len(runtimes)))

    return avg_runtime, stl_met
    
# x0 = np.array([[1., 1.]])
# v0 = np.array([[0., 0.]])
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

for i in range(numTestIts):
    currIt += 1
    x0_i = np.array([[1., 1.]])
    if singleTest == False:
        x0_i = np.random.rand(2)*10
        while (x0_i[0] >= obstacle_polygon_x1[0] - epsilon and x0_i[0] <= obstacle_polygon_x1[1] + epsilon and x0_i[1] >= obstacle_polygon_x2[0] + - epsilon and x0_i[1] <= obstacle_polygon_x2[1] + epsilon):
            x0_i = np.random.rand(2)*10
        
        x0_i = np.array([x0_i])
    
    v0_i = np.array([[0., 0.]])
    avgRuntime, isFeasible = runControlLoop(singleTest, x0_i, v0_i)
    if isFeasible:
        feas_x0List.append(x0_i)
    else:
        infeas_x0List.append(x0_i)

    plt.show()
    
print("Total feasible: ", len(feas_x0List))
print("Total infeasible: ", len(infeas_x0List))
print("Infeasible list: ", infeas_x0List)