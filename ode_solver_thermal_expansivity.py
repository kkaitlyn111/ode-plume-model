# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import math
# %pip install gsw

import pandas as pd


def plot_simple(x, y, desc_x, desc_y, logornot=False):
    plt.figure(figsize=(5, 3))
    plt.plot(x, y)
    plt.xlabel(desc_x)
    plt.ylabel(desc_y)
    plt.title(f'{desc_y} vs. {desc_x}')
    if (logornot):
      plt.xscale('log')
    plt.grid(True)
    plt.show()


def plot_buoyancy_one_figure(buoyancy_linear, buoyancy_gsw):

    x_linear = [item[0] for item in buoyancy_linear]
    y_linear = [item[1] for item in buoyancy_linear]
    x_gsw = [item[0] for item in buoyancy_gsw]
    y_gsw = [item[1] for item in buoyancy_gsw]

    plt.figure(figsize=(5, 3))
    plt.plot(x_linear, y_linear, label='linear', color='g')
    #plt.plot(x_linear, y_gsw, label='gsw', color='r')
    plt.xlabel('iterations (log)')
    plt.ylabel(f'buoyancy values')
    plt.title(f'Plume model: buoyancy Over Time (linear vs gsw)')
    plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_buoyancy_subfigures(buoyancy_linear, buoyancy_gsw):

    x_linear = [item[0] for item in buoyancy_linear]
    y_linear = [item[1] for item in buoyancy_linear]
    x_gsw = [item[0] for item in buoyancy_gsw]
    y_gsw = [item[1] for item in buoyancy_gsw]

    # Creating a figure with two subplots (horizontal layout)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

    if (buoyancy_linear != None):
        # Plotting on the first subplot
        ax1.plot(x_linear, y_linear, 'r-')  # red solid line
        ax1.set_title('buoyancy (linear formula)')
        ax1.set_xlabel('z')
        ax1.set_ylabel('buoyancy')
        ax1.set_xscale('log')
        ax1.set_xlim(0.05,10000)
        ax1.grid(True)


    # Plotting on the second subplot
    ax2.plot(x_gsw, y_gsw, 'b--')  # blue dashed line
    ax2.set_title('buoyancy (gsw non-linear formula)')
    ax2.set_xlabel('z')
    ax2.set_ylabel('buoyancy')
    ax2.set_xscale('log')
    ax2.set_xlim(0.05,10000)
    ax2.grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()

def plot_all(t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol):
    # Plotting the solutions
    plt.figure(figsize=(5, 3))
    plt.plot(t, r_sol, label='r')
    plt.plot(t, w_sol, label='w')
    plt.plot(t, T_sol, label='T')
    plt.plot(t, S_sol, label='S')
    plt.plot(t, b_sol, label='buoyancy')
    plt.plot(t, S_sol, label='alphaT')
    plt.legend()
    plt.yscale('log')

    plt.xlabel('z')
    plt.ylabel('r,w,T,S,buoyancy,alphaT values')
    plt.title('Plume model: r, w,T,S,buoyancy,alphaT Solutions Over Time (y: log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_all_2_subfigures(t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str):

    # Creating a figure with subplots (horizontal layout)
    scol = 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*scol+2, scol))  # 1 row, 2 columns
    fig.suptitle('r, w,T,S,buoyancy,alphaT, rho_plume Solutions: [' + init_cond_str + ']')

    # Plotting on the first subplot
    ax1.plot(t, r_sol, label='r')  # , 'r-')  # red solid line
    ax1.plot(t, w_sol, label='w')  # , 'r-')  # red solid line
    ax1.plot(t, T_sol, label='T')  # , 'r-')  # red solid line
    ax1.plot(t, S_sol, label='S')  # , 'r-')  # red solid line
    ax1.plot(t, rho_plume_sol, label='rho_plume')  #, 'r-')  # red solid line
    ax1.set_title('Radius, Omega, Temperature, Salinity, rho_plume')
    #ax1.set_xscale('log')
    ax1.set_xlabel('z')
    ax1.set_ylabel('r, w, T, S')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Plotting on the second subplot
    ax2.plot(t, b_sol, label='buoyancy')  #, 'r-')  # red solid line
    ax2.plot(t, alphaT_sol, label='alphaT')  #, 'r-')  # red solid line
    ax2.set_title('Buoyancy, AlphaT')
    ax2.set_xscale('log')
    ax2.set_xlabel('z')
    ax2.set_ylabel('Buoyancy, AlphaT')
    #ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()



def plot_all_subfigures(t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str):

    # Creating a figure with subplots (horizontal layout)
    nvar = 7
    scol = 2.5
    fig, axs = plt.subplots(2, 4, figsize=(nvar*scol/2+2, 2*scol))  # 1 row, 2 columns
    fig.suptitle('r, w,T,S,buoyancy,alphaT,rho_plume_sol Solutions: [' + init_cond_str + ']')
    # Flatten the array for easy access and remove the unnecessary last subplot
    axs = axs.flatten()
    fig.delaxes(axs[-1])  # Remove the last subplot (8th subplot in a 2x4 grid)


    # Plotting on the first subplot
    axs[0].plot(t, r_sol)  # , 'r-')  # red solid line
    axs[0].set_title('r (radius)')
    #axs[0].set_xscale('log')
    axs[0].set_xlabel('z')
    axs[0].set_ylabel('radius (log)')
    axs[0].set_yscale('log')
    axs[0].grid(True)

    # Plotting on the second subplot
    axs[1].plot(t, w_sol)  #, 'r-')  # red solid line
    axs[1].set_title('w (Omega)')
    #axs[1].set_xscale('log')
    axs[1].set_xlabel('z')
    axs[1].set_ylabel('Omega (log)')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    # Plotting on the third subplot
    axs[2].plot(t, T_sol)  #, 'r-')  # red solid line
    axs[2].set_title('T (temperature)')
    #axs[2].set_xscale('log')
    axs[2].set_xlabel('z')
    axs[2].set_ylabel('Temperature (log)')
    axs[2].set_yscale('log')
    axs[2].grid(True)

    # Plotting on the fourth subplot
    axs[3].plot(t, S_sol)  #, 'r-')  # red solid line
    axs[3].set_title('S (salinity)')
    #axs[3].set_xscale('log')
    axs[3].set_xlabel('z')
    axs[3].set_ylabel('Salinity')
    #axs[3].set_yscale('log')
    axs[3].grid(True)

    # Plotting on the fifth subplot
    axs[4].plot(t, b_sol, 'b--')  # blue dashed line
    axs[4].set_title('b (buoyancy_gsw)')
    axs[4].set_xscale('log')
    axs[4].set_xlabel('z (log)')
    axs[4].set_ylabel('Buoyancy_gsw')
    axs[4].set_xlim(0.05,10000)
    axs[4].grid(True)

    # Plotting on the sixth subplot
    axs[5].plot(t, alphaT_sol)  #, 'r-')  # red solid line
    axs[5].set_title('alphaT (thermal-expansivity)')
    axs[5].set_xscale('log')
    #axs[5].set_xlim(0.05,10000)
    axs[5].set_xlabel('z (log)')
    axs[5].set_ylabel('alphaT thermal-expansivity')
    #axs[5].set_yscale('log')
    axs[5].grid(True)

    # Plotting on the sixth subplot
    axs[6].plot(t, rho_plume_sol)  #, 'r-')  # red solid line
    axs[6].set_title('rho-plume')
    axs[6].set_xscale('log')
    #axs[6].set_xlim(0.05,10000)
    axs[6].set_xlabel('z (log)')
    axs[6].set_ylabel('rho-plume')
    #axs[6].set_yscale('log')
    axs[6].grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()


def plot_solutions_one(t, xx_sol, desc):
    plt.figure(figsize=(5, 3))
    plt.plot(t, xx_sol, label=desc)
    plt.xlabel('z')
    plt.ylabel(f'{desc} values')
    plt.title(f'Plume model: {desc} Solutions Over Time (log scale)')
    plt.yscale('log')
    #plt.legend()
    plt.grid(True)
    plt.show()


def plot_solutions_rwTS_alphaT(t, r_sol, w_sol, T_sol, S_sol, alphaT_sol):
    # Plotting the solutions
    plt.figure(figsize=(5, 3))
    plt.plot(t, r_sol, label='r')
    plt.plot(t, w_sol, label='w')
    plt.plot(t, T_sol, label='T')
    plt.plot(t, S_sol, label='S')
    plt.plot(t, S_sol, label='alphaT')
    plt.legend()
    plt.yscale('log')

    plt.xlabel('z')
    plt.ylabel('r,w,T,S,alphaT values')
    plt.title('Plume model: r, w,T,S Solutions Over Time (y: log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_solutions(t, r_sol, w_sol, T_sol, S_sol, b_sol=None, alphaT_sol=None):
    plot_solutions_rwTS_alphaT(t, r_sol, w_sol, T_sol, S_sol, alphaT_sol)
    plot_solutions_one(t, b_sol, 'buoyancy')
    plot_solutions_one(t, r_sol, 'r')
    plot_solutions_one(t, w_sol, 'w')
    plot_solutions_one(t, T_sol, 'T')
    plot_solutions_one(t, S_sol, 'S')
    plot_solutions_one(t, S_sol, 'alphaT')







"""# Implicit ODE solver with smooth (non-linear GSW) buoyancy convergence"""

from typing_extensions import TypeVarTuple
# Calculating non-linear buoyancy using GSW functions
# Implicit using ivp solver

import gsw
calc_linear_buoyancy = True
#do_linear_buoyancy = False

longitude = latitude = 0          # example longitude & latitude for the vent position
b_linear = []
b_gsw = []
solve_z = []

# stop condition for integration
def stop_condition(z, state):
    return abs(state[1] - wf) >= 0.01

stop_condition.terminal = True
stop_condition.direction = 0
#stop_condition.direction = -1


# ODE system function
def ODE_system(z, state):
    r, w, T, S, b, alphaT_plume, rho_plume = state

    solve_z.append(z)

    if (calc_linear_buoyancy):  # using simple linear buoyancy formula
        b_lin = g * (alphaT_plume * (T - T_amb) - betaS * (S - S_amb))
        #print('z=', z, ' linear buoyancy b=', b_lin)
        b_linear.append([z, b_lin])

    if (True):
        pressure = pressure_source_of_vent - pressure_factor*z  # Pressure at depth (decibars)
        # In numerical model without external forcing, S and T are SA and CT,
        # and can be directly fit into the gsw.rho function without conversion.

        # Calculate density
        rho_amb = gsw.rho(S_amb, T_amb, pressure)
        old_rho_plume = rho_plume
        rho_plume = gsw.rho(S, T, pressure)

        old_alphaT_plume = alphaT_plume
        alphaT_plume = gsw.alpha(S, T, pressure)

        # Calculate buoyancy
        old_b = b
        b = g * (rho_amb - rho_plume) / rho_amb
        ## b_gsw.append([z, b])

    drdz = (4 * alpha * w**2 - b * r) / (2 * w**2)
    dwdz = (-2 * alpha * w**2 + b * r) / (r * w)
    dTdz = -2 * alpha * (T - T_amb) / r
    dSdz = -2 * alpha * (S - S_amb) / r
    dbdz = (b - old_b) / delta_z
    dalphaTdz = (alphaT_plume - old_alphaT_plume) / delta_z
    drhoPlumdz = (rho_plume - old_rho_plume) / delta_z

    return [drdz, dwdz, dTdz, dSdz, dbdz, dalphaTdz, drhoPlumdz]

# solve the ODE system
#sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='LSODA', events=stop_condition)
#sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='Radau', events=stop_condition)
#sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='RK45', events=stop_condition, t_eval=np.arange(0, max_iter * delta_z, delta_z))

def print_every_n_rows(r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, N):
    if (N==0):
      N = 1
    for i in range(0, len(w_sol), N):
        print(f"{i}: r {r_sol[i]}, w {w_sol[i]}, T {T_sol[i]}, S {S_sol[i]}, b {b_sol[i]}, alpha {alphaT_sol[i]}, rho_plume {rho_plume_sol[i]}")


def solve_ODE(r0, w0, T0, S0, b0, alphaT0, rho_plume0):
    max_iter = 100000
    init_cond_str = f'r0={r0:.6f}, w0={w0:.6f}, T0={T0:.6f}, S0={S0:.5f}, b0={b0:.5f}, alphaT0={alphaT0:.6f}, rho_plume0={rho_plume0}, S_amb={S_amb:.6f}, T_amb={T_amb:.6f}'
    print('solve_ODE: initial state:', init_cond_str)
    sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0, b0, alphaT0, rho_plume0], method='RK45', events=stop_condition)

    # extract solutions
    r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol = sol.y
    iter = len(sol.t)
    N=int(iter/10)+1
    print(f"iter = {iter}; print 10 lines (every {N} iterations)")
    print_every_n_rows(r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, N)
    final_state = f"Final state: iterations = {iter};  r = {r_sol[-1]}, w = {w_sol[-1]}, T = {T_sol[-1]}, S = {S_sol[-1]}, b = {b_sol[-1]}, alphaT = {alphaT_sol[-1]}, rho_plume = {rho_plume_sol[-1]}"
    print(final_state)
    if (len(sol.t_events) > 0):  # stop_events met
        print(' #### stop condition met. (DEBUG: sol.t_events: len =', len(sol.t_events), sol.t_events, ')')
    #print('Z solution = ',  sol.t)     # check Z values

    print("Expected behavior: as the plume slows, \nw approaches 0, stopping at w <= 0.01 , \nradius r increases, \ntemperature T (initially 1E-2) approaches ambient T 1E-6, \nsalinity S (initially 40) approaches ambient S 4.")
    print(final_state)

    #plot_all(sol.t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol)
    plot_all_2_subfigures(sol.t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str)
    plot_all_subfigures(sol.t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str)

def solve_batch(r0, q0, S0, T0, S_amb, T_amb, do_plot=False):
    w0 = q0 / (math.pi * r0 ** 2)
    wf = 0  # end condition
    alphaT0 = gsw.alpha(S0, T0, pressure_source_of_vent)  # thermal_expansivity
    rho_plume0 = gsw.rho(S0, T0, pressure_source_of_vent)
    b0 = g * (alphaT0 * (T0 - T_amb) - betaS * (S0 - S_amb))    # buoyancy

    if(do_plot):
        print(f"Initial plume state:   r0={r0}    w0={w0}    T0={T0}    S0={S0}    alphaT0={alphaT0}    buoyancy0={b0}, rho_plume0={rho_plume0}")
        print(f"Ambient state:         T_amb={T_amb}   S_amb={S_amb}       alphaT_amb={alphaT}       betaS={betaS}       alpha_entrainment={alpha}")

    #solve_ODE(r0, w0, T0, S0, b0, alphaT0, rho_plume0)
    max_iter = 100000
    delta_z = 0.05
    if(do_plot):
        init_cond_str = f'r0={r0:.6f}, w0={w0:.6f}, T0={T0:.6f}, S0={S0:.5f}, b0={b0:.5f}, alphaT0={alphaT0:.6f}, rho_plume0={rho_plume0}, S_amb={S_amb:.6f}, T_amb={T_amb:.6f}'
        print('solve_ODE: initial state:', init_cond_str)
    sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0, b0, alphaT0, rho_plume0], method='RK45', events=stop_condition)
    if (do_plot == True and len(sol.t_events) > 0):  # stop_events met
        print(' #### stop condition met. (DEBUG: sol.t_events: len =', len(sol.t_events), sol.t_events, ')')

    if (do_plot == True):
        r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol = sol.y
        iter = len(sol.t)
        final_state = f"Final state: iterations = {iter};  r = {r_sol[-1]}, w = {w_sol[-1]}, T = {T_sol[-1]}, S = {S_sol[-1]}, b = {b_sol[-1]}, alphaT = {alphaT_sol[-1]}, rho_plume = {rho_plume_sol[-1]}"
        print(final_state)


    # extract solutions
    if (do_plot == False):
        return sol
    else:
        r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol = sol.y
        iter = len(sol.t)
        N=int(iter/10)
        print(f"iter = {iter}; print 10 lines (every {N} iterations)")
        print_every_n_rows(r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, N)
        final_state = f"Final state: iterations = {iter};  r = {r_sol[-1]}, w = {w_sol[-1]}, T = {T_sol[-1]}, S = {S_sol[-1]}, b = {b_sol[-1]}, alphaT = {alphaT_sol[-1]}, rho_plume = {rho_plume_sol[-1]}"
        print(final_state)
        if (len(sol.t_events) > 0):  # stop_events met
            print(' #### stop condition met. (DEBUG: sol.t_events: len =', len(sol.t_events), sol.t_events, ')')
        #print('Z solution = ',  sol.t)     # check Z values

        print("Expected behavior: as the plume slows, \nw approaches 0, stopping at w <= 0.01 , \nradius r increases, \ntemperature T (initially 1E-2) approaches ambient T 1E-6, \nsalinity S (initially 40) approaches ambient S 4.")
        print(final_state)

        plot_all_2_subfigures(sol.t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str)
        plot_all_subfigures(sol.t, r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol, init_cond_str)

    return sol





# Parameters and Initial values

# 4 equations (x1 = mass conservation, x2 = buoyancy, x3 = heat conservation, x4 = salinity conservation
# 4 unknowns r (width), w (upward velocity), T, S

# Ambient parameters
pressure_source_of_vent = 678   #6000 # 678     # example pressure at the source of vent (decibars)
pressure_factor = 0.113/9.78
g = 0.113 # Enceladus grav constant
alpha = 0.072 # entrainment
T_amb = 1E-6
T_amb_low = 1E-6
T_amb_high = 1E-1
S_amb_high = 40
S_amb_low = 4


alphaT = 5E-5 # thermal expansivity
betaS = 7E-4 # haline contractivity

#max_iter = 100000
wf = 0 # end condition

delta_z = 0.05
delta_z_ex = 0.05
print(f"Ambient state:    T_amb_low|high={T_amb_low}|{T_amb_high}      S_amb_low|high={S_amb_low}|{S_amb_high}        alphaT_amb={alphaT}       betaS={betaS}       alpha_entrainment={alpha}")





# Plume initial parameters
# b0 = -0.002967292907970786
#T0 = 1E-2; #S0 = 40

# S0_low = 4; S0_high = 40
# T0_high = 100; T0_mid = 10; T0_low = 1E-2
# Note: S0 = S_amb + 10 is a quite large reasonable difference
T0 = 100  # plume water temperature anomaly at vent is 100'c at Enceladus and Europa.
r0 = 1; q0 = 10;   # r0 is 1 or 10, test both.


def solve_from_initial_condition(r0, q0, S0, T0, S_amb=S_amb_low, T_amb=T_amb_low):
    w0 = q0 / (math.pi * r0 ** 2)
    wf = 0  # end condition
    alphaT0 = gsw.alpha(S0, T0, pressure_source_of_vent)  # thermal_expansivity
    rho_plume0 = gsw.rho(S0, T0, pressure_source_of_vent)
    b0 = g * (alphaT0 * (T0 - T_amb) - betaS * (S0 - S_amb))    # buoyancy

    print(f"Initial plume state:   r0={r0}    w0={w0}    T0={T0}    S0={S0}    alphaT0={alphaT0}    buoyancy0={b0}, rho_plume0={rho_plume0}")
    print(f"Ambient state:         T_amb={T_amb}   S_amb={S_amb}       alphaT_amb={alphaT}       betaS={betaS}       alpha_entrainment={alpha}")

    solve_ODE(r0, w0, T0, S0, b0, alphaT0, rho_plume0)
    #explicit_iter(r0, w0, T0, S0, b0, alphaT0, rho_plume0, S_amb, T_amb)

"""




   




Temperature difference; No salinity difference

Low ambient salinity (4 PSU) = Plume Salinity (4 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-6 Celcius)"""



import numpy as np

# Assuming z_sol is a numpy array with increasing float values. For demonstration, let's create a sample array.
z_sol = np.array([150, 180, 190, 210, 220, 230])

print(z_sol > 200)
# Find the index where z_sol[i] > 200 for the first time
index = np.argmax(z_sol > 200)

print(f"The index where z_sol[i] is > 200 for the first time is: {index}")

num_sampling = 6
z_threshold = 60   # 200m?

S_amb_high = 40
S_amb_low = 4
T_amb_low = 1E-6
#S_amb = S_amb_low

#T0 = 100  # plume water temperature anomaly at vent is 100'c at Enceladus and Europa.
#r0 = 1; q0 = 10;   # r0 is 1 or 10, test both.
#w0 = q0 / (math.pi * r0 ** 2)

T_amb = T_amb_low
S_amb_arr = np.linspace(S_amb_low, S_amb_high, num_sampling)

r0_arr = np.linspace(1,15,num_sampling)
#q0_arr = np.linspace(0.03,30,num_sampling)
q0_arr = np.linspace(10,30,num_sampling)
T0_arr = np.linspace(60,100,num_sampling)
S0_arr = S_amb_arr + 10
w0_arr = q0_arr / (math.pi * r0_arr ** 2)

print(f"T_amb = {T_amb}")
print(f"configs: r0={r0_arr}")
print(f"configs: q0={q0_arr}")
print(f"configs: T0={T0_arr}")
print(f"configs: S_ambient={S_amb_arr}")
print(f"configs: S0={S0_arr}")
print(f"derived config: w0_arr")
for rx in r0_arr:
  for qx in q0_arr:
    wx = qx / (math.pi * rx ** 2)
    print('wx = ', wx, end='')
  print('\n')

#print(f"configs: w0={w0}")

S_amb = S_amb_low
T_amb = T_amb_low
S0 = S_amb + 10
T0 = 100  # plume water temperature anomaly at vent is 100'c at Enceladus and Europa.
r0 = 1; q0 = 10;   # r0 is 1 or 10, test both.
#print('###', r0_arr[0], q0_arr[0], S0_arr[0], T0_arr[0], S_amb_arr[0], T_amb)
#solve_from_initial_condition(r0_arr[0], q0_arr[0], S0_arr[0], T0_arr[0], S_amb_arr[0], T_amb)

#batch_conditions = []
batch_results = []

for r0x in r0_arr:
  print('r0x = ', r0x)
  for q0x in q0_arr:
    for T0x in T0_arr:
      for S0x in S0_arr:
        for S_ambx in S_amb_arr:
          #print(f'configs: r0x={r0x}  q0x={q0x}  T0x={T0x}  S0x={S0x}  S_ambx={S_ambx}')
          sol = solve_batch(r0x, q0x, S0x, T0x, S_ambx, T_amb, do_plot=False)
          z_sol = sol.t
          r_sol, w_sol, T_sol, S_sol, b_sol, alphaT_sol, rho_plume_sol = sol.y

          z_idx = np.argmax(z_sol > z_threshold)
          #print('z_threshold: ', z_threshold, ' idx = ', z_idx, 'r_sol[i] = ', r_sol[z_idx])
          #plot_simple(z_sol, r_sol, 'z_sol', 'r_sol')
          #batch_conditions.append([r0x, q0x, T0x, S0x, S_ambx])
          batch_results.append([r0x, q0x, T0x, S0x, S_ambx, z_sol[z_idx], r_sol[z_idx], w_sol[z_idx], T_sol[z_idx], S_sol[z_idx], b_sol[z_idx], alphaT_sol[z_idx], rho_plume_sol[z_idx]])


# Convert batch_results to a pandas DataFrame
# Adjust the column names as necessary
df = pd.DataFrame(batch_results, columns=['r0', 'q0', 'T0', 'S0', 'S_amb', 'z_sol', 'r_sol', 'w_sol', 'T_sol', 'S_sol', 'b_sol', 'alphaT_sol', 'rho_plume_sol'])
# Specify the CSV file name
filename = 'plume_batch_n' + str(num_sampling) + '_z' + str(z_threshold) + '_new.csv'
print('save results to file: ', filename)
# Save the DataFrame to CSV
df.to_csv(filename, index=False)

# retrieve columns
r0_col = [row[0] for row in batch_results]
q0_col = [row[1] for row in batch_results]
T0_col = [row[1] for row in batch_results]
S0_col = [row[1] for row in batch_results]
S_amb_col = [row[1] for row in batch_results]
z_sol_col = [row[1] for row in batch_results]
r_sol_col = [row[1] for row in batch_results]
w_sol_col = [row[1] for row in batch_results]
T_sol_col = [row[1] for row in batch_results]
S_sol_col = [row[1] for row in batch_results]
b_sol_col = [row[1] for row in batch_results]
alphaT_sol_col = [row[1] for row in batch_results]
rho_plume_sol_col = [row[1] for row in batch_results]

plot_simple(r0_col, r_sol_col, 'r0', 'r_sol')
plot_simple(r0_col, w_sol_col, 'r0', 'w_sol')

import os
import glob

current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")

# List all .csv files in the current directory
csv_files = glob.glob('*.csv')
for file in csv_files:
    # Get the size of the file
    size = os.path.getsize(file)
    # Print the file name and its size
    print(f"File: {file}, Size: {size} bytes")


num_sampling = 1
z_threshold = 60
filename = '/content/plume_batch_n' + str(num_sampling) + '_z' + str(z_threshold) + '.csv'
print(filename)
#df_tmp = pd.read_csv(filename)
#df_tmp

print(len(df))

# Specify the CSV file name

filename = 'plume_batch_n' + str(num_sampling) + '_z' + str(z_threshold) + '.csv'
# Read the CSV into a pandas DataFrame
df_results = pd.read_csv(filename)
# If you need the data in a list of lists format (similar to the original batch_results)
batch_results_list = df.values.tolist()

# Alternatively, you might keep and use the DataFrame directly for data manipulation,
# as it offers a lot of convenient functionalities.
df_results

r02, q02, T02, S02, S_amb2, z_sol2, r_sol2, w_sol2, T_sol2, S_sol2, b_sol2, alphaT_sol2, rho_plume_sol2 = [df[name] for name in df.columns]
r0_v = r02.tolist()
q0_v = q02.tolist()
T0_v = T02.tolist()
S0_v = S02.tolist()
S_amb_v = S_amb2.tolist()
z_sol_v = z_sol2.tolist()
r_sol_v = r_sol2.tolist()
w_sol_v = w_sol2.tolist()
T_sol_v = T_sol2.tolist()
S_sol_v = S_sol2.tolist()
b_sol_v = b_sol2.tolist()
alphaT_sol_v = alphaT_sol2.tolist()
rho_plume_sol_v = rho_plume_sol2.tolist()

plot_simple(r0_v, r_sol_v, 'r0', 'r_sol')
plot_simple(r0_v, w_sol_v, 'r0', 'w_sol')
# plot_2D(r0_v, q0_v, r_sol_v, 'r0', 'q0', 'r_sol')    # DDDD:  Implement 2D suface plots here
# plot_2D(r0_v, q0_v, w_sol_v, 'r0', 'q0', 'w_sol')

print(df_results)



#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import numpy as np

def surface_plot(x1_v, x2_v, y_v, x1_str, x2_str, y_str):
    # Assuming r0_v, q0_v, and r_sol_v are 1D arrays.
    # You need to convert them to a grid for plotting.
    # This requires that r0_v and q0_v form a complete grid.

    # Convert r0_v and q0_v to 2D grids
    x1, x2 = np.meshgrid(x1_v, x2_v)

    # Make sure r_sol_v is reshaped to match the dimensions of the grid.
    # R_sol_v should be reshaped if it's not already in the correct 2D shape.
    y = r_sol_v.reshape(y_v.shape)

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the surface
    surface = ax.plot_surface(x1, x2, y, cmap='viridis')

    # Add a color bar to show the scale of r_sol
    fig.colorbar(surface)

    # Labeling
    ax.set_xlabel(x1_str)
    ax.set_ylabel(x2_str)
    ax.set_zlabel(y_str)
    ax.set_title('3D Surface Plot of ' + y_str)

    plt.show()

surface_plot(r0_v, q0_v, r_sol_v, 'r0', 'q0', 'r_sol')

surface_plot(r0_v, q0_v, r_sol_v, 'r0', 'q0', 'w_sol')

surface_plot(r0_v, q0_v, r_sol_v, 'r0', 'T0', 'r_sol')

surface_plot(r0_v, q0_v, r_sol_v, 'r0', 'T0', 'w_sol')





# No salinity difference
# Low salinity scenario (both plume and ambient)
S0 = S_amb = S_amb_low
T_amb = T_amb_low
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)



"""Temperature difference; No salinity difference


High ambient salinity (40 PSU) = Plume Salinity (40 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-6 Celcius)
"""

# No salinity difference
# High salinity scenario (both plume and ambien)
S0 = S_amb = S_amb_high
T_amb = T_amb_low
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)



"""Temperature difference; Salinity difference

Low ambient salinity (4 PSU), Plume Salinity (S_ambient+10 = 14 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-6 Celcius)
"""

# Add salinity difference
# Low salinity scenario (ambient)
S_amb = S_amb_low
T_amb = T_amb_low
S0 = S_amb + 10   # salinity differnce of 10 is quite large
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)

"""Temperature difference; Salinity difference

High ambient salinity (40 PSU), Plume Salinity (S_ambient+10 = 50 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-6 Celcius)
"""

# Add salinity difference
# High salinity scenario (ambient)
S_amb = S_amb_high
T_amb = T_amb_low
S0 = S_amb + 10   # salinity differnce of 10 is quite large
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)

"""Temperature difference; Salinity difference

High ambient salinity (40 PSU), Plume Salinity (S_ambient+10 = 50 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-1 Celcius)
"""

# Add salinity difference
# High salinity scenario (ambient)
# High ambient temperature
S_amb = S_amb_high
S0 = S_amb + 10   # salinity differnce of 10 is quite large
T_amb = T_amb_high
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)

"""Temperature difference; Salinity difference

High ambient salinity (40 PSU), Plume Salinity (S_ambient+10 = 50 PSU)

Plume temperature (100 Celcius), Ambient temperature (1E-6 Celcius)
"""

# Add salinity difference
# High salinity scenario (ambient)
# Low ambient temperature

S_amb = S_amb_high
S0 = S_amb + 10   # salinity differnce of 10 is quite large
T_amb = T_amb_low
T0 = 100
solve_from_initial_condition(r0, q0, S0, T0, S_amb, T_amb)


