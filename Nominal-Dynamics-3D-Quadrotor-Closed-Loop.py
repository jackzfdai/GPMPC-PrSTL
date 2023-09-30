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

np_rng = np.random.default_rng(seed=0)

#wind params
K_vent = 0.5
var_vent = 0.005

#Control parameters
T = 6 #Time horizon
surveillance_A_period_t = 3 #s
surveillance_B_period_t = 3 #s
dt = 0.2
N = int(T/dt) # Control intervals 
surveillance_A_period = int(surveillance_A_period_t/dt)
surveillance_B_period = int(surveillance_B_period_t/dt)
epsilon = 0.0
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
    m = gb.Model('quadrotor3DMICP')

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

def solveCP(plot, x0, v0, qVarphi1, qVarphi2, qVarphi3, qVarphi4, pPhi1, pPhi2):
    #Problem at time step k = 0 ____________________________________________
    m = gb.Model('quadrotor3DCP')

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
        
    # qVarphi1 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi1")
    # qVarphi2 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi2")
    # qVarphi3 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi3")
    # qVarphi4 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi4")
    # qVarphi5 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi5")
    # qVarphi6 = m.addVars(N, vtype=GRB.BINARY, name="qVarphi6")
    # pPhi1 = m.addVars(N, vtype=GRB.BINARY, name="pPhi1")
    # pPhi2 = m.addVars(N, vtype=GRB.BINARY, name="pPhi2")


    # pPhi1_sum = gb.LinExpr()
    # pPhi2_sum = gb.LinExpr()

    for i in range(len(x0), N):
        # Avoid predicates
        m.addConstr(0 <= M*(1 - qVarphi1[i]) - epsilon + (obstacle_polygon_x1[0] - x[i, 0]))
        m.addConstr(0 <= M*(1 - qVarphi2[i]) - epsilon + (x[i, 0] - obstacle_polygon_x1[1]))
        m.addConstr(0 <= M*(1 - qVarphi3[i]) - epsilon + (obstacle_polygon_x2[0] - x[i, 1]))
        m.addConstr(0 <= M*(1 - qVarphi4[i]) - epsilon + (x[i, 1] - obstacle_polygon_x2[1]))
        # m.addConstr(0 <= M*(1 - qVarphi5[i]) - epsilon + (obstacle_polygon_x3[0] - x[i, 2]))
        # m.addConstr(0 <= M*(1 - qVarphi6[i]) - epsilon + (x[i, 2] - obstacle_polygon_x3[1]))

        #Avoid disjunctions
        # m.addConstr(1 <= qVarphi1[i] + qVarphi2[i] + qVarphi3[i] + qVarphi4[i]) # + qVarphi5[i] + qVarphi6[i])

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
        pPhi1_i = pPhi1[i] #m.addVars(surveillance_A_period, vtype=GRB.BINARY, name="pPhi1_{:d}".format(i))
        # pPhi1_i_sum = gb.LinExpr()

        for j in range (surveillance_A_period):
            if (i + j >= len(x0)):
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 0] - goal_A_polygon_x1[0]))
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 0] + goal_A_polygon_x1[1]))
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 1] - goal_A_polygon_x2[0]))
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 1] + goal_A_polygon_x2[1]))
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (x[i+j, 2] - goal_A_polygon_x3[0]))
                m.addConstr(0 <= M*(1 - pPhi1_i[j]) - epsilon + (- x[i+j, 2] + goal_A_polygon_x3[1]))

            # pPhi1_i_sum.add(pPhi1_i[j])

        # m.addConstr(1 <= pPhi1_i_sum)

    for i in range(0, N-surveillance_B_period+1):
        pPhi2_i = pPhi2[i] #m.addVars(surveillance_B_period, vtype=GRB.BINARY, name="pPhi2_{:d}".format(i))
        # pPhi2_i_sum = gb.LinExpr()

        for j in range (surveillance_B_period):
            if (i + j >= len(x0)):         
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 0] - goal_B_polygon_x1[0]))
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 0] + goal_B_polygon_x1[1]))
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 1] - goal_B_polygon_x2[0]))
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 1] + goal_B_polygon_x2[1]))
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (x[i+j, 2] - goal_B_polygon_x3[0]))
                m.addConstr(0 <= M*(1 - pPhi2_i[j]) - epsilon + (- x[i+j, 2] + goal_B_polygon_x3[1]))

            # pPhi2_i_sum.add(pPhi2_i[j])

        # m.addConstr(1 <= pPhi2_i_sum)

    # m.setParam("BarHomogenous", 1)
    m.optimize()
    m.printStats()
    m.printQuality()

    x1_sol = []
    x2_sol = []
    x3_sol = []
    v1_sol = []
    v2_sol = []
    v3_sol = []
    theta_sol = []
    phi_sol = []
    thrust_sol = []
    # qVarphi1_sol = []
    # qVarphi2_sol = []
    # qVarphi3_sol = []
    # qVarphi4_sol = []
    # qVarphi5_sol = []
    # qVarphi6_sol = []
    # pPhi1_sol = []
    # pPhi2_sol = []
    solver_runtime = m.getAttr('Runtime')
    solver_status = m.getAttr('Status')
    feasible = False
    if solver_status == GRB.OPTIMAL:
        feasible = True
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

            # qVarphi1_sol_i = m.getVarByName("qVarphi1[{:d}]".format(i))
            # qVarphi1_sol.append(qVarphi1_sol_i.X)
            # qVarphi2_sol_i = m.getVarByName("qVarphi2[{:d}]".format(i))
            # qVarphi2_sol.append(qVarphi2_sol_i.X) 
            # qVarphi3_sol_i = m.getVarByName("qVarphi3[{:d}]".format(i))
            # qVarphi3_sol.append(qVarphi3_sol_i.X)
            # qVarphi4_sol_i = m.getVarByName("qVarphi4[{:d}]".format(i))
            # qVarphi4_sol.append(qVarphi4_sol_i.X)
            # qVarphi5_sol_i = m.getVarByName("qVarphi5[{:d}]".format(i))
            # qVarphi5_sol.append(qVarphi5_sol_i.X)
            # qVarphi6_sol_i = m.getVarByName("qVarphi6[{:d}]".format(i))
            # qVarphi6_sol.append(qVarphi6_sol_i.X)
            
        # for i in range(N-surveillance_A_period+1):
        #     pPhi1_sol_i = [] 
        #     for j in range(surveillance_A_period):
        #         pPhi1_sol_i_j = m.getVarByName("pPhi1_{:d}[{:d}]".format(i, j))
        #         pPhi1_sol_i.append(pPhi1_sol_i_j.X)
        #     pPhi1_sol.append(pPhi1_sol_i)
        
        # for i in range(N-surveillance_B_period+1):
        #     pPhi2_sol_i = [] 
        #     for j in range(surveillance_B_period):
        #         pPhi2_sol_i_j = m.getVarByName("pPhi2_{:d}[{:d}]".format(i, j))
        #         pPhi2_sol_i.append(pPhi2_sol_i_j.X)
        #     pPhi2_sol.append(pPhi2_sol_i)

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

    return feasible, solver_runtime, theta_sol, phi_sol, thrust_sol #qVarphi1_sol, qVarphi2_sol, qVarphi3_sol, qVarphi4_sol, pPhi1_sol, pPhi2_sol

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

def runControlLoop(plot, x0, v0):
    sigma_x0 = [1e-6, 1e-6, 1e-6]
    sigma_v0 = [1e-6, 1e-6, 1e-6]

    runtimes = []
    
    theta_micp, phi_micp, thrust_micp, qVarphi1_micp, qVarphi2_micp, qVarphi3_micp, qVarphi4_micp, pPhi1_micp, pPhi2_micp = solveMICP(plot, x0, v0) #qVarphi5_micp, qVarphi6_micp, 

    isFeasible = False
    satSTL = False
    for i in range(0, N-1):
        isFeasible, runtime, theta_cp, phi_cp, thrust_cp = solveCP(False, x0, v0, qVarphi1_micp, qVarphi2_micp, qVarphi3_micp, qVarphi4_micp, pPhi1_micp, pPhi2_micp)
        runtimes.append(runtime)

        if isFeasible == True:
            xv_next = simNextStep(True, x0[i], v0[i], np.array([theta_cp[0], phi_cp[0], thrust_cp[0]]))
            x0 = np.concatenate([x0, [xv_next[:3]]], axis=0)
            v0 = np.concatenate([v0, [xv_next[3:]]], axis=0)
            print("CP feasible at time step: ", str(i))
        else:
            print("CP infeasible at time step: ", str(i))
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

# plt.show()

singleTest = False
numTestIts = 10

if singleTest == True:
    numTestIts = 1

currIt = 0
# sigmax1Guess = np.zeros(N)
# sigmax2Guess = np.zeros(N)

sat_x0List = []
unsat_x0List = []
infeas_list = []
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
        sat_x0List.append(x0_i)
    else:
        unsat_x0List.append(x0_i)

    if NLPFeasOnTermination == False:
        numNLPInfeas += 1
        infeas_list.append(x0_i)

    plt.show()
    
print("Total sat: ", len(sat_x0List))
print("Total unsat: ", len(unsat_x0List))
print("Total unsat due to infeas: ", str(numNLPInfeas))
print("Unsat list: ", unsat_x0List)
print("Infeas list: ", infeas_list)