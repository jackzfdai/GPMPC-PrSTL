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

#Control parameters
T = 3 #Time horizon
dt = 0.1
N = int(T/dt)
epsilon = 0.05
M = N

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
# train_g1 = A1*torch.sin(train_inputs[:,0]) + torch.randn(train_inputs[:,0].size()) * math.sqrt(Sigma_w)
# train_g2 = A2*torch.sin(train_inputs[:,1]) + torch.randn(train_inputs[:,1].size()) * math.sqrt(Sigma_w)
var_residual = 0.005
K_residual = 0.05
Bd = np.concatenate([1/2*(dt**2)*np.eye(2), dt*np.eye(2)], axis=0)

#Variable bounds
lbx = [0., 0., -5., -5., -5.]
ubx = [10., 10., 5., 5., 5.]

lbu = [-2., -2.]
ubu = [2., 2.]

lbsigma = [0., 0., 0., 0.]
ubsigma = [10., 10., 10., 10.]

x_lb = lbx[0] #Split into x1, x2 if needed later 
x_ub = ubx[0]
v_lb = lbx[2]
v_ub = ubx[2]
u_lb = lbu[0]
u_ub = ubu[0]

#Reach avoid parameters
goal_A_polygon_x1 = [7, 8]
goal_A_polygon_x2 = [3, 4] 
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

    return solver_runtime, u1_sol, u2_sol, x1_sol, x2_sol, qVarphi1_sol, qVarphi2_sol, qVarphi3_sol, qVarphi4_sol, pPhi1_sol

def plotTraj(x1_sol, x2_sol, x1_openloops = [], x2_openloops = []):
    goal_A_polygon_x1_plot = [goal_A_polygon_x1[0], goal_A_polygon_x1[0], goal_A_polygon_x1[1], goal_A_polygon_x1[1]]
    goal_A_polygon_x2_plot = [goal_A_polygon_x2[1], goal_A_polygon_x2[0], goal_A_polygon_x2[0], goal_A_polygon_x2[1]] 
    obstacle_polygon_x1_plot = [obstacle_polygon_x1[0], obstacle_polygon_x1[0], obstacle_polygon_x1[1], obstacle_polygon_x1[1]]
    obstacle_polygon_x2_plot = [obstacle_polygon_x2[1], obstacle_polygon_x2[0], obstacle_polygon_x2[0], obstacle_polygon_x2[1]]
    plt.figure()
    ax = plt.gca()
    plt.xlim(lbx[0], ubx[0])
    plt.ylim(lbx[1], ubx[1])
    plt.xlabel("x")
    plt.ylabel("y")
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
    runtimes = []
    stl_met = True 
    obj_cl_sum = 0
    x1_openloop_sol = []
    x2_openloop_sol = []

    # _, milp_u1, milp_u2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol = solveMILP(plot, x0, v0)

    for i in range(0, N-1):
        runtime, milp_u1, milp_u2, milp_x1, milp_x2, milp_qVarphi1_sol, milp_qVarphi2_sol, milp_qVarphi3_sol, milp_qVarphi4_sol, milp_pPhi1_sol = solveMILP(False, x0, v0)
        runtimes.append(runtime)
        if len(milp_u1) > 0:
            xv_next = simNextStep(True, x0[i], v0[i], np.array([milp_u1[0], milp_u2[0]]))
            x0 = np.concatenate([x0, [xv_next[:2]]], axis=0)
            v0 = np.concatenate([v0, [xv_next[2:]]], axis=0)
            obj_cl_sum += milp_u1[0]**2 + milp_u2[0]**2
            x1_openloop_sol.append(milp_x1)
            x2_openloop_sol.append(milp_x2)
        else:
            print("Problem infeasible at time step: ", str(i))
            stl_met = False
            break

    if plot == True: 
        plotTraj(x0[:,0], x0[:,1], x1_openloop_sol, x2_openloop_sol)
    print(x0)
    print("____")
    print(v0)
    print("Closed loop objective", str(obj_cl_sum))
    avg_runtime = -1
    if len(runtimes) > 0:
        avg_runtime = sum(runtimes)/len(runtimes)
    print("Average runtime: ", avg_runtime)
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