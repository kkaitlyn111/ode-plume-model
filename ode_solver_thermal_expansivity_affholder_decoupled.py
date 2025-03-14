# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
import math
# %pip install gsw

import pandas as pd
from typing_extensions import TypeVarTuple
import gsw
from scipy.interpolate import interp1d



def plot_all_2_subfigures(t, r_sol, w_sol, T_sol, S_sol, bsol, asol, rsol, init_cond_str):

    # Creating a figure with subplots (horizontal layout)
    scol = 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*scol+2, scol))  # 1 row, 2 columns
    fig.suptitle('r,w,T,S,buoyancy,alphaT,plume density: [' + init_cond_str + ']')

    # Plotting on the first subplot
    ax1.plot(t, r_sol, label='r')  # , 'r-')  # red solid line
    ax1.plot(t, w_sol, label='w')  # , 'r-')  # red solid line
    ax1.plot(t, T_sol, label='T')  # , 'r-')  # red solid line
    ax1.plot(t, S_sol, label='S')  # , 'r-')  # red solid line
    ax1.plot(t, rsol, label='rho_plume')  #, 'r-')  # red solid line
    ax1.set_title('Radius, Upward Velocity, Temperature, Salinity, Density of Plume')
    #ax1.set_xscale('log')
    ax1.set_xlabel('z')
    ax1.set_ylabel('r, w, T, S, rho')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Plotting on the second subplot
    ax2.plot(t, bsol, label='buoyancy')  #, 'r-')  # red solid line
    ax2.plot(t, asol, label='alphaT')  #, 'r-')  # red solid line
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

def plot_all_subfigures(t, r_sol, w_sol, T_sol, S_sol, bsol, asol, rsol, init_cond_str):
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
    axs[4].plot(bsol, 'b--')  # blue dashed line
    axs[4].set_title('b (buoyancy_gsw)')
    axs[4].set_xlabel('idx')
    axs[4].set_ylabel('Buoyancy_gsw')
    axs[4].grid(True)

    # Plotting on the sixth subplot
    axs[5].plot(asol)  #, 'r-')  # red solid line
    axs[5].set_title('alphaT (thermal-expansivity)')
    axs[5].set_xlabel('idx')
    axs[5].set_ylabel('alphaT thermal-expansivity')
    # axs[5].set_yscale('log')
    axs[5].grid(True)

    # Plotting on the sixth subplot
    axs[6].plot(rsol)  #, 'r-')  # red solid line
    axs[6].set_title('rho-plume')
    axs[6].set_xlabel('idx')
    axs[6].set_ylabel('rho-plume')
    # axs[6].set_yscale('log')
    axs[6].grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()



"""## Parameters and Initial Values


"""

##### Ambient Parameters #####

pressure_source_of_vent = 678   #6000 # 678     # example pressure at the source of vent (decibars)
pressure_factor = 0.113/9.78
g = 0.113 # Enceladus grav constant
alpha = 0.072 # entrainment

T_amb_low = 1E-6
T_amb_high = 1E-1
S_amb_low = 4
S_amb_high = 40

S_amb = S_amb_high
T_amb = T_amb_low

alphaT = 5E-5 # thermal expansivity
betaS = 7E-4 # haline contractivity

#max_iter = 10000000
wf = 0 # end condition
delta_z = 0.05

calc_linear_buoyancy = True
longitude = latitude = 0          # example longitude & latitude for the vent position

def calculate_K1_K2(T, S):
    # Convert temperature to Kelvin
      T_kelvin = T_amb + 273.15

    # Empirical equations for K1 and K2 (from Millero, 1995)
      K1 = 10**(-1.0 * (3633.86/T_kelvin - 61.2172 + 9.6777 * np.log(T_kelvin) - 0.011555 * S + 0.0001152 * S**2))
      K2 = 10**(-1.0 * (471.78/T_kelvin + 25.9290 - 3.16967 * np.log(T_kelvin) - 0.01781 * S + 0.0001122 * S**2))

      return K1, K2

def calculate_co2(DIC, pH, T, S):
    # Calculate [H+] from pH
    H_plus = 10**-pH

    K1, K2 = calculate_K1_K2(T, S)
    # Calculate [CO2] using the formula derived above
    CO2_concentration = DIC / (1 + (K1 / H_plus) + (K1 * K2 / H_plus**2))

    return CO2_concentration

pH = np.random.uniform(7.95, 9.05)
CH4_amb = np.random.uniform(1e-8, 1e-6)
H2_amb = np.random.uniform(1e-8, 1e-6)
DIC = np.random.uniform(4e-3, 1e-1)
CO2_amb = 1e-6 # calculate_co2(DIC, pH, T_amb, S_amb)

print(f"Ambient state:    T amb low = {T_amb_low}, T amb high = {T_amb_high}     S amb low = {S_amb_low}, S amb high = {S_amb_high}        alphaT amb = {alphaT}       betaS = {betaS}       alpha (entrainment) = {alpha}")

##### Plume Initial Parameters #####


b0 = -0.002967292907970786

# S0_low = 4; S0_high = 40
# T0_high = 100; T0_mid = 10; T0_low = 1E-2
# Note: S0 = S_amb + 10 is a quite large reasonable difference
T0 = 100  # plume water temperature anomaly at vent is 100'C at Enceladus and Europa.
r0 = 1; q0 = 10;   # r0 is 1 or 10, test both.
S0 = S_amb + 10

pH0 = 11 # pH of plume at base, unknown
H20 = np.random.uniform(1e-8, 1e-1)
CH40 = np.random.uniform(1e-8, 1e-4)
DIC0 = np.random.uniform(4e-8, 1e-6)
CO20 = 1e-5 # calculate_co2(DIC0, pH0, T0, S0)
B0 = 0.0001

def solve_from_initial_condition(r0, q0, S0, T0, S_amb=S_amb_low, T_amb=T_amb_low):
    w0 = q0 / (math.pi * r0 ** 2)
    wf = 0  # end condition
    solve_ODE(r0, w0, T0, S0)

#### Stop Condition ####

# stop condition for integration
def stop_condition(z, state):
    return abs(state[1] - wf) - 0.01

stop_condition.terminal = True
stop_condition.direction = 0

#stop_condition.direction = -1

#### Model Parameters ####
max_iter = 10000000

### Non-Prognostic Variables ###

def calculate_vars(z, S, T):
  pressure = pressure_source_of_vent - pressure_factor*z  # Pressure at depth (decibars)
  # In numerical model without external forcing, S and T are SA and CT,
  # and can be directly fit into the gsw.rho function without conversion.

  # Calculate density
  rho_amb = gsw.rho(S_amb, T_amb, pressure)
  rho_plume = gsw.rho(S, T, pressure)

  alphaT_plume = gsw.alpha(S, T, pressure)

  # Calculate buoyancy
  b = g * (rho_amb - rho_plume) / rho_amb

  return [b, alphaT_plume, rho_plume]

### Biochemical Model Constants & Equations ###

Y1 = 0.25 # CH4
Y2 = -0.25 # CO2
Y3 = -1 # H2

R = 0.008314 # kJ/(mol·K)
deltaGcat0 = -32.6 # kJ/mol
deltaGdiss = 1088 # kJ/molC

d = 0.03 # /day, baseline death rate

def calc_qcat(Q, T):
  T_kelvin = T + 273.15
  return calc_deltaGcat(Q, T_kelvin) / deltaGdiss

def calc_deltaGcat(Q, T):
  return deltaGcat0 + R * T * np.log(Q)

def calc_qm(T):
  exponent = np.clip(69400 / R * (1 / 298 - 1 / T), -700, 700)
  em = 84 * np.exp(exponent)
  return em / deltaGdiss

def calc_qana(Q, T):
  T_kelvin = T + 273.15
  lambd = -calc_deltaGcat(Q, T_kelvin) / deltaGdiss
  return lambd * calc_qcat(Q, T) - calc_qm(T_kelvin)

def calc_vars_from_z_r_w_S_T(z, r, w, S, T, old_b):
  pressure = pressure_source_of_vent - pressure_factor*z  # Pressure at depth (decibars)
  # In numerical model without external forcing, S and T are SA and CT,
  # and can be directly fit into the gsw.rho function without conversion.
  # Calculate density
  rho_amb = gsw.rho(S_amb, T_amb, pressure)
  rho_plume = gsw.rho(S, T, pressure)
  alphaT_plume = gsw.alpha(S, T, pressure)
  # Calculate buoyancy
  b = g * (rho_amb - rho_plume) / rho_amb

  drdz = (4 * alpha * w**2 - b * r) / (2 * w**2)
  dwdz = (-2 * alpha * w**2 + b * r) / (r * w)
  dTdz = -2 * alpha * (T - T_amb) / r
  dSdz = -2 * alpha * (S - S_amb) / r
  dbdz = (b - old_b) / delta_z

  #variable_map[z] = [r, w, S, T, alphaT_plume, b, rho_plume]
  #dvariable_map[z] = [dwdz, dbdz]

  return [r, w, S, T, alphaT_plume, b, rho_plume, dwdz, dbdz]

"""## ODE Solver

"""

from sys import dont_write_bytecode
z_sol_iter = []

cnt_ODE1_iter = 0
z_sol, r_sol, w_sol, T_sol, S_sol = [], [], [], [], []
bsol, asol, rhosol = [], [], []
dwdz_sol, dbdz_sol = [], []


b_lin_sol, b_gsw_sol = [], []
#variable_map = { 'z' : ['r', 'w', 'S', 'T', 'alphaT', 'b', 'rho']}
#dvariable_map = { 'z' : ['dwdz', 'dbdz']}
variable_map = {}
dvariable_map = {}

config_use_t_eval = True
#config_use_t_eval = False

old_b_solver1 = 0

# ODE system function
def ODE_system(z, state):
    r, w, T, S = state
    global cnt_ODE1_iter, old_b_solver1

    cnt_ODE1_iter += 1

    print('cnt ', cnt_ODE1_iter, ': step z = ', z, 'r,w,T,S=', r, w, T, S)
    if (True):
        pressure = pressure_source_of_vent - pressure_factor*z  # Pressure at depth (decibars)
        # In numerical model without external forcing, S and T are SA and CT,
        # and can be directly fit into the gsw.rho function without conversion.

        # Calculate density
        rho_amb = gsw.rho(S_amb, T_amb, pressure)
        rho_plume = gsw.rho(S, T, pressure)

        alphaT_plume = gsw.alpha(S, T, pressure)

        # Calculate buoyancy
        b = g * (rho_amb - rho_plume) / rho_amb
        #b_gsw_sol.append(b)

        #if len(b_gsw_sol) == 1:
        #  old_b = 0
        #else:
        #  old_b = b_gsw_sol[-2]

    z_sol_iter.append(z)
    drdz = (4 * alpha * w**2 - b * r) / (2 * w**2)
    dwdz = (-2 * alpha * w**2 + b * r) / (r * w)
    dTdz = -2 * alpha * (T - T_amb) / r
    dSdz = -2 * alpha * (S - S_amb) / r
    dbdz = (b - old_b_solver1) / delta_z

    old_b_solver1 = b

    #variable_map_iter[z] = [r, w, S, T, alphaT_plume, b, rho_plume]
    #dvariable_map_iter[z] = [dwdz, dbdz]

    return [drdz, dwdz, dTdz, dSdz]

# Access and store the solution for r, w, T, S at each t_eval point

def solve_ODE(r0, w0, T0, S0):

    global z_sol, r_sol, w_sol, T_sol, S_sol, bsol, asol, rhosol

    init_cond_str = f'r0 = {r0:.6f} m, w0 = {w0:.6f} m^3/s, T0 = {T0:.6f} C, S0 = {S0:.5f} g/kg, S_amb = {S_amb:.6f} g/kg, T_amb = {T_amb:.6f} C'
    print('### Initial State ###\n\n', init_cond_str)
    print('config_use_t_eval = ', config_use_t_eval, ', max_iter = ', max_iter, ', delta_z = ', delta_z, ', max_iter*delta_z = ', max_iter*delta_z)
    print('len = ', len(t_eval), 't_eval = ', t_eval)

    #sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='RK45', events=stop_condition)
    sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='RK45', t_eval = t_eval, events=stop_condition)
    #sol = solve_ivp(ODE_system, [0, max_iter * delta_z], [r0, w0, T0, S0], method='RK23', t_eval = t_eval, events=stop_condition)

    # extract solutions
    z_sol = sol.t
    iter = len(z_sol)
    N=int(iter/10)+1
    print('config_use_t_eval = ', config_use_t_eval, ', len(t_eval) = ', len(t_eval), ', t_eval = ', t_eval)
    print(f"\n### Solving ODE ###\n\niter (len(sol.t)=len(z_sol))= {iter}; print 10 lines (every {N} iterations)")
    print('len(sol.t) = ', len(sol.t), ', sol.t = ', sol.t)
    print('###len(z_sol_iter) = ', len(z_sol_iter), ', z_sol_iter = ', z_sol_iter)

    r_sol = sol.y[0, :]  # The values of 'r' at each t_eval point
    w_sol = sol.y[1, :]  # The values of 'w' at each t_eval point
    T_sol = sol.y[2, :]  # The values of 'T' at each t_eval point
    S_sol = sol.y[3, :]  # The values of 'S' at each t_eval point

    old_b_solver1 = 0
    for idx in range(len(z_sol)):
        z = z_sol[idx]
        r = r_sol[idx]
        w = w_sol[idx]
        S = S_sol[idx]
        T = T_sol[idx]

        r, w, S, T, alphaT_plume, b, rho_plume, dwdz, dbdz = calc_vars_from_z_r_w_S_T(z, r, w, S, T, old_b_solver1)
        bsol.append(b)
        asol.append(alphaT_plume)
        rhosol.append(rho_plume)
        dwdz_sol.append(dwdz)
        dbdz_sol.append(dbdz)
        old_b_solver1 = b
        #variable_map[z] = [r, w, S, T, alphaT_plume, b, rho_plume]
        #dvariable_map[z] = [dwdz, dbdz]

    #print('len=', len(variable_map), 'variable_map = ', variable_map)
    #print('len=', len(dvariable_map), 'dvariable_map = ', dvariable_map)
    #print('len=', len(variable_map_iter), 'variable_map_iter = ', variable_map_iter)
    #print('len=', len(dvariable_map_iter), 'dvariable_map_iter = ', dvariable_map_iter)


    print_every_n_rows(z_sol, r_sol, w_sol, T_sol, S_sol, bsol, asol, rhosol, N)
    final_state = f"\n### Final State ###\n\niterations = {iter};  r = {r_sol[-1]}, w = {w_sol[-1]}, T = {T_sol[-1]}, S = {S_sol[-1]}, B = {bsol[-1]}, alphaT = {asol[-1]}, rho = {rhosol[-1]}"
    print(final_state)
    print('final sol.status = ', sol.status)
    if sol.status == 1:
      print("Solver stopped due to an event.")
      print(f"Event occurred at time: {sol.t_events}")
    else:
      print(f"Solver stopped at z = {sol.t[-1]} for a different reason.")
      print(f"Solver status: {sol.status}")
      print(f"Solver message: {sol.message}")


    print(f"\n### Expected behavior ###\n\nAs the plume slows, \nw approaches {wf}, stopping at w <= 0.01, \nradius r increases, \ntemperature T (initially {T0}) approaches ambient T {T_amb}, \nsalinity S (initially {S0}) approaches ambient S {S_amb}.")
    print('config_use_t_eval = ', config_use_t_eval, ', t (Z) solution = ',  sol.t)     # check Z values


    plot_all_2_subfigures(z_sol, r_sol, w_sol, T_sol, S_sol, bsol, asol, rhosol, init_cond_str)
    plot_all_subfigures(z_sol, r_sol, w_sol, T_sol, S_sol, bsol, asol, rhosol, init_cond_str)



def print_every_n_rows(z_sol, r_sol, w_sol, T_sol, S_sol, b_sol, a_sol, rho_sol, N=0):

    if (N==0):
      N = 1
    for i in range(0, len(z_sol), N):
        print(f"{i}: z {z_sol[i]}, r {r_sol[i]}, w {w_sol[i]}, T {T_sol[i]}, S {S_sol[i]}, b {b_sol[i]}, alphaT {a_sol[i]}, rho {rho_sol[i]}")





def sequence_timepoints(length):
    a = 1.0
    b = 0.005
    c = 0

    x_data = np.arange(length)
    modeled_values = a * np.exp(b * x_data) + c - 1   # HACK: make it start from 0
    return modeled_values

# Generate time points ensuring they do not exceed delta_z * max_iter
length = max_iter + 1  # Determine the length based on max_iter
modeled_values = sequence_timepoints(length)

# Filtering modeled values to be within the limit delta_z * max_iter
t_eval = [value for value in modeled_values if value <= delta_z * max_iter]

print('sequence_timepoints: len(t_eval) = ', len(t_eval), ', t_eval = ', t_eval)

"""## Run ODE

"""

solve_from_initial_condition(r0, q0, S0, T0)

print('len=', len(t_eval), ', t_eval = ', t_eval)
print('len=', len(z_sol), ', z_sol = ', z_sol)
print('len=', len(z_sol_iter), ', z_sol_iter = ', z_sol_iter)
#print('len= ', len(variable_map), ', variable_map:', variable_map)
#print('len= ', len(dvariable_map), ', dvariable_map:', dvariable_map)
#print(variable_map.keys())
#print(dvariable_map.keys())

"""## Plotting"""



def plot_all_2_subfigures_biochem(t, CH4_sol, CO2_sol, H2_sol, B_sol, init_cond_str):

    # Creating a figure with subplots (horizontal layout)
    scol = 4
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2*scol+2, scol))  # 1 row, 2 columns
    fig.suptitle('CH4, CO2, H2, Buoyancy: [' + init_cond_str + ']')

    # Plotting on the first subplot
    ax1.plot(t, CH4_sol, label='r')  # , 'r-')  # red solid line
    ax1.plot(t, CO2_sol, label='w')  # , 'r-')  # red solid line
    ax1.plot(t, H2_sol, label='T')  # , 'r-')  # red solid line
    ax1.set_title('CH4, CO2, H2')
    #ax1.set_xscale('log')
    ax1.set_xlabel('z')
    ax1.set_ylabel('CH4, CO2, H2')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True)

    # Plotting on the second subplot
    ax2.plot(t, B_sol, label='Buoyancy')  #, 'r-')  # red solid line
    ax2.set_title('Buoyancy')
    ax2.set_xscale('log')
    ax2.set_xlabel('z')
    ax2.set_ylabel('Buoyancy')
    #ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()

def plot_all_subfigures_biochem(t, CH4_sol, CO2_sol, H2_sol, B_sol, init_cond_str):

    # Creating a figure with subplots (horizontal layout)
    nvar = 4
    scol = 2.5
    fig, axs = plt.subplots(2, 4, figsize=(nvar*scol/2+2, 2*scol))  # 1 row, 2 columns
    fig.suptitle('r, w,T,S,buoyancy,alphaT,rho_plume_sol Solutions: [' + init_cond_str + ']')
    # Flatten the array for easy access and remove the unnecessary last subplot
    axs = axs.flatten()
    fig.delaxes(axs[-1])  # Remove the last subplot (8th subplot in a 2x4 grid)


    # Plotting on the first subplot
    axs[0].plot(t, CH4_sol)  # , 'r-')  # red solid line
    axs[0].set_title('CH4')
    #axs[0].set_xscale('log')
    axs[0].set_xlabel('z')
    axs[0].set_ylabel('CH4')
    axs[0].set_yscale('log')
    axs[0].grid(True)

    # Plotting on the second subplot
    axs[1].plot(t, CO2_sol)  #, 'r-')  # red solid line
    axs[1].set_title('CO2')
    #axs[1].set_xscale('log')
    axs[1].set_xlabel('z')
    axs[1].set_ylabel('CO2')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    # Plotting on the third subplot
    axs[2].plot(t, H2_sol)  #, 'r-')  # red solid line
    axs[2].set_title('H2')
    #axs[2].set_xscale('log')
    axs[2].set_xlabel('z')
    axs[2].set_ylabel('H2')
    axs[2].set_yscale('log')
    axs[2].grid(True)

    # Plotting on the fourth subplot
    axs[3].plot(t, B_sol)  #, 'r-')  # red solid line
    axs[3].set_title('Buoyancy')
    #axs[3].set_xscale('log')
    axs[3].set_xlabel('z')
    axs[3].set_ylabel('Buoyancy')
    #axs[3].set_yscale('log')
    axs[3].grid(True)

    # Adjust layout to not overlap
    plt.tight_layout()
    plt.show()

"""# Biochem Model

"""

##biochem_variable_map = { 'z' : ['CH4', 'CO2', 'H2', 'B']}
biochem_variable_map = {}





z_sol2_iter = []
#biochem_variable_map = {}
cnt_ODE2_iter = 0

# Interpolating functions for r, w, T, S from the previous solution
r_interp = interp1d(z_sol, r_sol, kind='linear', fill_value="extrapolate")
w_interp = interp1d(z_sol, w_sol, kind='linear', fill_value="extrapolate")
S_interp = interp1d(z_sol, S_sol, kind='linear', fill_value="extrapolate")
T_interp = interp1d(z_sol, T_sol, kind='linear', fill_value="extrapolate")
b_interp = interp1d(z_sol, bsol, kind='linear', fill_value="extrapolate")
dwdz_interp = interp1d(z_sol, dwdz_sol, kind='linear', fill_value="extrapolate")
dbdz_interp = interp1d(z_sol, dbdz_sol, kind='linear', fill_value="extrapolate")

old_b_biochem = 0

def ODE_system_biochem(z, state):
    CH4, CO2, H2, B  = state

    global old_b_biochem, cnt_ODE2_iter

    z_sol2_iter.append(z)
    #biochem_variable_map[z] = [CH4, CO2, H2, B]

    r = r_interp(z)
    w = w_interp(z)
    S = S_interp(z)
    T = T_interp(z)
    dwdz = dwdz_interp(z)
    dbdz = dbdz_interp(z)
    b = b_interp(z)

    #pressure = pressure_source_of_vent - pressure_factor * z
    #rho_amb = gsw.rho(S_amb, T_amb, pressure)
    #rho_plume = gsw.rho(S, T, pressure)
    #alphaT_plume = gsw.alpha(S, T, pressure)
    #b = g * (rho_amb - rho_plume) / rho_amb  # Calculate buoyancy

    #r, w, S, T, alphaT_plume, b, rho_plume, dwdz, dbdz = calc_vars_from_z_r_w_S_T(z, r, w, S, T, old_b_biochem)
    #dwdz = (-2 * alpha * w**2 + b * r) / (r * w)
    #dbdz = (b - old_b_biochem) / delta_z

    Q = CH4**(0.25)/(H2 * CO2**(0.25))
    qcat = calc_qcat(Q, T)
    qana = calc_qana(Q, T)

    dCH4dz = (2 * alpha * CH4_amb) / b + Y1 * qcat * B / w - (2 * dbdz / b) * CH4 - (dwdz / w) * CH4
    dCO2dz = (2 * alpha * CO2_amb) / b + Y2 * qcat * B / w - (2 * dbdz / b) * CO2 - (dwdz / w) * CO2
    dH2dz = (2 * alpha * H2_amb) / b + Y3 * qcat * B / w - (2 * dbdz / b) * H2 - (dwdz / w) * H2
    dBdz = B * ((qana - d) / w - (2 * dbdz / b) - (dwdz / w))

    old_b_biochem = b

    cnt_ODE2_iter += 1
    if (cnt_ODE2_iter < 10):
      print('cnt_iter ', cnt_ODE2_iter, ': z=', z, ', state (ch4,co2,h2,b_2) = ', state)
      print('(z, r,w,S,T;  b, (old_b);   dwdz,dbdz) =', z, r, w, S, T, ';    ', b, '(', old_b_biochem, ')', ';    ', dwdz, dbdz)
      print(' ')
      print('dCH4dz, dCO2dz, dH2dz, dBdz = ', dCH4dz, dCO2dz, dH2dz, dBdz)

    return [dCH4dz, dCO2dz, dH2dz, dBdz]

def solve_ODE_biochem(CH40, CO20, H20, B0):

    global CH4_sol, CO2_sol, H2_sol, B_sol
    CH4_sol = []
    CO2_sol = []
    H2_sol = []
    B_sol = []

    init_cond_str = f'CH40 = {CH40:.6f}, CO20 = {CO20:.6f}, H20 = {H20:.6f}, B0 = {B0:.5f}'
    print('### Initial State ###\n\n', init_cond_str, ', max_iter = ', max_iter, ', delta_z = ', delta_z)

    #max_iter_solver2 = len(z_sol)
    #print('max_iter_solver2 = ', max_iter_solver2)

    #sol = solve_ivp(ODE_system_biochem, [0, max_iter_solver2 * delta_z], [CH40, CO20, H20, B0], method='RK23')
    #sol = solve_ivp(ODE_system_biochem, [0, max_iter_solver2 * delta_z], [CH40, CO20, H20, B0], method='RK45', t_eval = t_eval, events=stop_condition)
    #sol = solve_ivp(ODE_system_biochem, [0, max_iter_solver2 * delta_z], [CH40, CO20, H20, B0], method='RK45', t_eval = z_sol, first_step = 0.0797873997206577)
    sol = solve_ivp(ODE_system_biochem, [0, max_iter * delta_z], [CH40, CO20, H20, B0], method='RK45', t_eval = z_sol, first_step = 0.0797873997206577)
    #sol = solve_ivp(ODE_system_biochem, [0, max_iter_solver2 * delta_z], [CH40, CO20, H20, B0], method='RK45')

    z_sol2 = sol.t
    print('2: len(sol.t)=len_z_sol2=', len(z_sol2), ', z_sol2 = ', z_sol2)
    print('len_z_sol2_iter = ', len(z_sol2_iter), z_sol2_iter)

    CH4_sol = sol.y[0, :]  # The values of 'CH4' at each t_eval point
    CO2_sol = sol.y[1, :]  # The values of 'CO2' at each t_eval point
    H2_sol = sol.y[2, :]  # The values of 'H2' at each t_eval point
    B_sol = sol.y[3, :]  # The values of 'B' at each t_eval point

    for idx in range(len(z_sol2)):
        z = z_sol2[idx]
        ch4 = CH4_sol[idx]
        co2 = CO2_sol[idx]
        h2 = H2_sol[idx]
        b_2 = B_sol[idx]
        biochem_variable_map[z] = [ch4, co2, h2, b_2]
        print(f"idx {idx}, z={z}: CH4 {ch4}, CO2 {co2}, H2 {h2}, B {b_2}")


    # extract solutions
    iter = len(z_sol2)
    N= int(iter/10)+1
    print(f"\n### Solving ODE ###\n\niter = {iter}; print 10 lines (every {N} iterations)")

    print_every_n_rows_biochem(CH4_sol, CO2_sol, H2_sol, B_sol, N)
    final_state = f"\n### Final State ###\n\niterations = {iter};  CH4 = {CH4_sol[-1]},CO2 = {CO2_sol[-1]},H2 = {H2_sol[-1]},B = {B_sol[-1]}"
    print(final_state)

    print('final sol.status = ', sol.status)
    if sol.status == 1:
      print("Solver stopped due to an event.")
      print(f"Event occurred at time: {sol.t_events}")
    else:
      print(f"Solver stopped at z = {sol.t[-1]} for a different reason.")
      print(f"Solver status: {sol.status}")
      print(f"Solver message: {sol.message}")


    print('biochem: sol.t = ',  sol.t)     # check Z values


    plot_all_2_subfigures_biochem(z_sol2, CH4_sol, CO2_sol, H2_sol, B_sol, init_cond_str)
    plot_all_subfigures_biochem(z_sol2, CH4_sol, CO2_sol, H2_sol, B_sol, init_cond_str)

#def print_every_n_rows_biochem(r_sol, w_sol, T_sol, S_sol, b_gsw_sol, alphaT_sol, rho_sol, N=0):
def print_every_n_rows_biochem(z_sol2, CH4_sol, CO2_sol, H2_sol, B_sol, N=0):

    if (N==0):
      N = 1
    for i in range(0, len(z_sol2), N):
        print(f"{i}: z {z_sol2[i]}, r {CH4_sol[i]}, w {CO2_sol[i]}, T {H2_sol[i]}, S {B_sol[i]}")



print(len(t_eval), t_eval)
print(len(z_sol), z_sol)

##### Plume Initial Parameters #####

b0 = -0.002967292907970786

# S0_low = 4; S0_high = 40
# T0_high = 100; T0_mid = 10; T0_low = 1E-2
# Note: S0 = S_amb + 10 is a quite large reasonable difference
T0 = 100  # plume water temperature anomaly at vent is 100'C at Enceladus and Europa.
r0 = 1; q0 = 10;   # r0 is 1 or 10, test both.
S0 = S_amb + 10

##pH0 = 11 # pH of plume at base, unknown
##H20 = np.random.uniform(1e-8, 1e-1)
##CH40 = np.random.uniform(1e-8, 1e-4)
##DIC0 = np.random.uniform(4e-8, 1e-6)
##CO20 = 1e-5 # calculate_co2(DIC0, pH0, T0, S0)
##B0 = 0.0001

pH0 = 11 # pH of plume at base, unknown
DIC0 = 1e-7
B0 = 0.0001
H20 = 0.073185
CH40 = 0.000067
CO20 = 0.000010

def solve_from_initial_condition_biochem(CH40, CO20, H20, B0):
    solve_ODE_biochem(CH40, CO20, H20, B0)

z_sol2, CH4_sol, CO2_sol, H2_sol, B_sol = [], [], [], [], []

solve_from_initial_condition_biochem(CH40, CO20, H20, B0)

print(dvariable_map.keys())
print(z_sol2)
print(t_eval)

