import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d

import pandas as pd
import scipy as sp
import scipy.integrate
from time import time
# VCSEL Parameters
a = linewidth_enhancement_factor  = 1 # alpha
g_p = linear_birefringence_rate = 20 # rad/ns, gamma_p
g_s = spin_relaxation_rate = 10 # 1/ns, gamma_s #frequency term
g_a = gain_anisotropy = 0 # 1/ns, gamma_a # frequency term
g = electron_lifetime_frequency = 1 # gamma, 1/ns, 1/tau_n
k = half_photon_lifetime_frequency = 125 # kappa = photon_lifetime/2

# Plot the surface.
color_rcp = "#2bfef0"
color_lcp = "#952bfe"

vcsel_parameters = {
	"a": a,
	"g_p": g_p,
	"g_s": g_s,
	"g_a": g_a,
	"g": g,
	"k": k
}
## VCSEL Variations
eta = 2
P = 0
O_x = 0
O_y = 0
O_z = 0

experiment_conditions = pd.DataFrame({
	"eta" : eta,
	"P" : P,
	"O_x" : O_x,
	"O_y" : O_y,
	"O_z" : O_z
}, index=[0])

experiment_conditions.values
initial_solution= np.ones(8) * 1

def SFM(t, y,
		vcsel_parameters,
		experiment_parameters):
	E_pr = y[0]
	E_nr = y[1]
	E_pi = y[2]
	E_ni = y[3]
	N = y[4]
	m_x = y[5]
	m_y = y[6]
	m_z = y[7]

	# VCSEL parameter unpacking:
	alpha = vcsel_parameters["alpha"]
	gamma_a = vcsel_parameters["gamma_a"]
	gamma_p = vcsel_parameters["gamma_p"]
	gamma_s = vcsel_parameters["gamma_s"]
	gamma = vcsel_parameters["gamma"]
	kappa = vcsel_parameters["kappa"]

	# TODO: Verify that the following P timeseries unpacking is performant
	# Experimental parameters unpacking:
	eta_all = experiment_parameters["eta"]  # here P is 2D array with time and value columns
	# get current value index based on simulation time and value step times:
	eta_index = np.searchsorted(eta_all[:, 0], t) - 1
	eta = eta_all[eta_index, 1]

	P = experiment_parameters["P"]
	Omega_x = experiment_parameters["Omega_x"]
	Omega_y = experiment_parameters["Omega_y"]
	Omega_z = experiment_parameters["Omega_z"]

	# Calculate additional parameters from VCSEL parameters:
	I_p = (E_pr ** 2 + E_pi ** 2)
	I_n = (E_nr ** 2 + E_ni ** 2)

	eta_p = eta * (P + 1) / 2
	eta_n = eta * (1 - P) / 2

	# Rate equations:
	dE_pr = kappa * (N + m_z - 1) * (E_pr - alpha * E_pi) - gamma_a * E_pr + gamma_p * E_ni - Omega_z * E_pi
	dE_nr = kappa * (N - m_z - 1) * (E_nr - alpha * E_ni) - gamma_a * E_nr + gamma_p * E_pi + Omega_z * E_ni

	dE_pi = kappa * (N + m_z - 1) * (E_pi + alpha * E_pr) - gamma_a * E_pi - gamma_p * E_nr + Omega_z * E_pr
	dE_ni = kappa * (N - m_z - 1) * (E_ni + alpha * E_nr) - gamma_a * E_ni - gamma_p * E_pr - Omega_z * E_nr

	# Total carrier density N

	dN = gamma * (eta_p + eta_n - (1 + I_p + I_n) * N - (I_p - I_n) * m_z)

	# Normalised electron spin magnetization m_xyz

	dm_x = -1 * (gamma_s + gamma * (I_p + I_n)) * m_x + Omega_y * m_z - Omega_z * m_y
	dm_y = -1 * (gamma_s + gamma * (I_p + I_n)) * m_y + Omega_z * m_x - Omega_x * m_z
	dm_z = gamma * (eta_p - eta_n) - (gamma_s + gamma * (I_p + I_n)) * m_z - gamma * (
			I_p - I_n) * N + Omega_x * m_y - Omega_y * m_x

	# need to return all d/dt things
	dy = [dE_pr, dE_nr, dE_pi, dE_ni, dN, dm_x, dm_y, dm_z]
	return dy

# Stored variable usage
dS_columns = np.array(["dE_pr", "dE_mr", "dE_pi", "dE_mi", "dN", "dm_x", "dm_y", "dm_z"])
results_columns = dS_columns
def further_results(solution = np.zeros(9)):
	# Total Right Circular Emission Radiant Energy and Intensity
	E_p = np.sqrt(solution[0] ** 2 + solution[2] ** 2)
	E_m = np.sqrt(solution[1] ** 2 + solution[3] ** 2)
	N = solution[4]
	I_p = E_p ** 2
	I_m = E_m ** 2
	# Total emission intensity
	I = (I_p + I_m)/2

	# Orthogonal Linear Components
	E_x = (E_p + E_m) / np.sqrt(2)
	E_y = -(E_p - E_m) / np.sqrt(2)
	I_x = E_x ** 2
	I_y = E_y ** 2

	# Total emission elipticity
	I_difference = (I_p - I_m)
	e =  I_difference / (I_p + I_m)

	return np.array([E_p, E_m, I_p, I_m, I_difference, I, E_x, E_y, I_x, I_y, e, N])

further_results_columns = np.array(["E_p", "E_m", "I_p", "I_m", "I_difference", "I", "E_x", "E_y,", "I_x", "I_y", "e", "N"])
results_columns = np.append(results_columns, further_results_columns)
def plot_results(simulation_time, solution, detailed_solution):
	#  +, - Components
	plt.figure()
	plt.plot(simulation_time, solution[0], label="E_pr")
	plt.plot(simulation_time, solution[1], label="E_mr")
	plt.legend()

	# I_n, I_p
	plt.figure()
	plt.plot(simulation_time, detailed_solution[2], label="I_p")
	plt.plot(simulation_time, detailed_solution[3], label="I_n")
	plt.ylim((0,1))
	plt.legend()

	# X-Y Components
	plt.figure()
	plt.plot(simulation_time, detailed_solution[6], label="E_x")
	plt.plot(simulation_time, detailed_solution[7], label="E_y")
	plt.legend()

	# Elipticity
	plt.figure()
	plt.plot(simulation_time, detailed_solution[8], label= "Elipticity Eta")
	plt.legend()

def vcsel_experimental_modelling(vcsel_parameters,
								 experiment_conditions, initial_solution= np.ones(8) * 1):

	# Set up simulation time
	min_simulation_time = 0
	max_simulation_time = 35
	simultion_time_steps= 5
	simulation_time = np.arange(min_simulation_time, max_simulation_time, simultion_time_steps)
	integration_interval = [min_simulation_time, max_simulation_time]


	# Integration solver
	sim_results = sp.integrate.solve_ivp(SFM,
									integration_interval,
									initial_solution,
									method='Radau',
									t_eval=simulation_time,
									dense_output=True,
									args=(vcsel_parameters,
										  experiment_conditions))


	# Selected solution range
	solution = sim_results.sol(simulation_time)
	detailed_solution = further_results(solution)
	simulation_data = np.concatenate((solution, detailed_solution), 0)


	# Get derived parameters
	experiment_conditions_dataframe = pd.DataFrame(experiment_conditions,
												   index=[0])
	vcsel_parameters_dataframe = pd.DataFrame(vcsel_parameters,
												   index=[0])
	repeated_experiment_conditions = pd.concat(
		[experiment_conditions_dataframe]*simulation_time.size, ignore_index=True)
	repeated_vcsel_parameters = pd.concat(
		[vcsel_parameters_dataframe]*simulation_time.size, ignore_index=True)

	# Simulation dataframe
	simulation_dataframe = pd.DataFrame(simulation_data.T,
										columns = results_columns)
	simulation_dataframe["simulation_time"] = simulation_time

	for experimental_parameter in repeated_experiment_conditions.keys():
		simulation_dataframe[
			experimental_parameter] =  repeated_experiment_conditions[
											experimental_parameter]

	for vcsel_parameter in repeated_vcsel_parameters.keys():
		simulation_dataframe[
			vcsel_parameter] =  repeated_vcsel_parameters[
									vcsel_parameter]

	return simulation_dataframe
## VCSEL Variations
eta = 2
P = 0
O_x = 0
O_y = 0
O_z = 0

experiment_conditions = pd.DataFrame({
	"eta" : eta,
	"P" : P,
	"O_x" : O_x,
	"O_y" : O_y,
	"O_z" : O_z
}, index=[0])
#Varying Linewidth Enhancement Factor $\alpha$
a = linewidth_enhancement_factor  = 1 # alpha
g_p = linear_birefringence_rate = 20 # rad/ns, gamma_p
g_s = spin_relaxation_rate = 10 # 1/ns, gamma_s #frequency term
g_a = gain_anisotropy = 0 # 1/ns, gamma_a # frequency term
g = electron_lifetime_frequency = 1 # gamma, 1/ns, 1/tau_n
k = half_photon_lifetime_frequency = 125 # kappa = photon_lifetime/2
initial_time = int(time())

# Varing VCSEL parameters
VCSEL_variations_DataFrame = pd.DataFrame()
for a_i in np.linspace(0, 8, 50):
	# print("g_p_i: " + str(g_p_i))
	# print("time: " + str(int(time()-initial_time)))
	vcsel_parameters = {
		"a": a_i,
		"g_p": g_p,
		"g_s": g_s,
		"g_a": g_a,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 2,
		"P" : 0,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	VCSEL_variations_DataFrame = VCSEL_variations_DataFrame.append(iteration_dataframe, ignore_index=True)

VCSEL_variations_DataFrame.to_csv(
	"simulation_results/linewidth_enhancement_factor_variations_"
	 + str(int(time()))
	 + ".csv")
linewidth_enhancement_factor_variations_dataframe = pd.read_csv("simulation_results/linewidth_enhancement_factor_variations_1589052786.csv")
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1, projection = '3d')

dataframe_selection = linewidth_enhancement_factor_variations_dataframe[(linewidth_enhancement_factor_variations_dataframe.simulation_time > 15) &
															 (linewidth_enhancement_factor_variations_dataframe.index % 10 == 0)]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.a
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component

# Plot the surface.
color_rcp = "#2bfef0"
color_lcp = "#952bfe"
# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission polarization - $\alpha$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r' Linewidth Enhancement Factor $\alpha$', fontsize =12)#, fontweight="bold")
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  ['Y Linearly Polarized (YLP)', 'X Linearly Polarized (XLP)'],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)

plt.savefig("graph/linewidth_enhancement_factor_variations_"
			 + str(int(time()))
			 + ".png", bbox_inches="tight")
#Varying Photon Lifetime Frequency
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')
half_photon_lifetime_frequency_variations = pd.read_csv("simulation_results/half_photon_lifetime_frequency_variations_1588206418.csv")
half_photon_lifetime_frequency_variations
dataframe_selection = half_photon_lifetime_frequency_variations[(half_photon_lifetime_frequency_variations.simulation_time > 10) &
															   (half_photon_lifetime_frequency_variations.a > 1)]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.k # z axial field component
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component

# Plot the surface.
color_rcp = "#2bfef0"
color_lcp = "#952bfe"
# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission polarization - $\kappa$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r'Variations $\frac{rad}{ns}$', fontsize =12)#, fontweight="bold") # TODO more applicable definition
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  ['Y Linearly Polarized (YLP)', 'X Linearly Polarized (XLP)', ],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)

# plt.savefig("graphs/half_photon_lifetime_frequency_variations_"
#              + str(int(time()))
#              + ".png", bbox_inches="tight")
### Varying Linear Birefringence Rate $\gamma_p$
a = linewidth_enhancement_factor  = 5 # alpha
g_p = linear_birefringence_rate = 34.5 # rad/ns, gamma_p
g_s = spin_relaxation_rate = 105 # 1/ns, gamma_s #frequency term
g_a = gain_anisotropy = 0 # 1/ns, gamma_a # frequency term
g = electron_lifetime_frequency = 1 # gamma, 1/ns, 1/tau_n
k = half_photon_lifetime_frequency = 250 # kappa = photon_lifetime/2
initial_time = int(time())

# Varing VCSEL parameters
VCSEL_variations_DataFrame = pd.DataFrame()
for g_p_i in np.linspace(0, 10, 100):
	vcsel_parameters = {
		"a": a,
		"g_p": g_p_i,
		"g_s": g_s,
		"g_a": g_a,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 2,
		"P" : 0,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	VCSEL_variations_DataFrame = VCSEL_variations_DataFrame.append(iteration_dataframe, ignore_index=True)

VCSEL_variations_DataFrame.to_csv(
	"simulation_results/linear_birefringence_rate_variations_"
	 + str(int(time()))
	 + ".csv")
linear_birefringence_variations_dataframe = pd.read_csv("simulation_results/linear_birefringence_rate_variations_1621435012.csv")
linear_birefringence_variations_dataframe
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')

dataframe_selection = linear_birefringence_variations_dataframe[(linear_birefringence_variations_dataframe.simulation_time > 15)]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.g_p # z axial field component
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component

# Plot the surface.
color_rcp = "#2bfef0"
color_lcp = "#952bfe"
# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission polarization - $\gamma_p$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r' $\gamma_p$ Linear Birefringence Variations $\frac{rad}{ns}$', fontsize =12)#, fontweight="bold") # TODO more applicable definition
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  [ 'Y Linearly Polarized (YLP)', 'X Linearly Polarized (XLP)'],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)

# Varying Gain Anisotropy (Dichroism)  𝛾𝑎
# Varing VCSEL parameters
VCSEL_variations_DataFrame = pd.DataFrame()
for g_a_i in np.linspace(0, 60, 100):
	vcsel_parameters = {
		"a": a,
		"g_p": g_p,
		"g_s": g_s,
		"g_a": g_a_i,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 2,
		"P" : 0,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	VCSEL_variations_DataFrame = VCSEL_variations_DataFrame.append(iteration_dataframe, ignore_index=True)

VCSEL_variations_DataFrame.to_csv(
	"simulation_results/gain_anisotropy_variations_"
	 + str(int(time()))
	 + ".csv")
gain_anisotropy_variations_dataframe = pd.read_csv("simulation_results/gain_anisotropy_variations_1621435614.csv")
gain_anisotropy_variations_dataframe
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')

dataframe_selection = gain_anisotropy_variations_dataframe[gain_anisotropy_variations_dataframe.simulation_time > 10]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.g_a # z axial field component
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component
# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission polarization - $\gamma_a$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r' $\gamma_a$ Gain Anisotropy $ns^{-1}$', fontsize =12)#, fontweight="bold") # TODO more applicable definition
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  ['Y Linearly Polarized (YLP)','X Linearly Polarized (XLP)'],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)
###################################################################################################
# Varying Spin Relaxation Rate  𝛾𝑠

# Varing VCSEL parameters
spin_relaxation_rate_variations_dataframe = pd.DataFrame()
for g_s_i in np.linspace(90, 100, 50):
	vcsel_parameters = {
		"a": a,
		"g_p": g_p,
		"g_s": g_s_i,
		"g_a": g_a,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 2,
		"P" : 0,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	spin_relaxation_rate_variations_dataframe = spin_relaxation_rate_variations_dataframe.append(iteration_dataframe, ignore_index=True)

spin_relaxation_rate_variations_dataframe.to_csv(
	"simulation_results/spin_relaxation_rate_variations_"
	 + str(int(time()))
	 + ".csv")
spin_relaxation_rate_variations_dataframe = pd.read_csv("simulation_results/spin_relaxation_rate_variations_1590484216.csv")
spin_relaxation_rate_variations_dataframe
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')

dataframe_selection = spin_relaxation_rate_variations_dataframe[spin_relaxation_rate_variations_dataframe.simulation_time > 10]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.g_s # z axial field component
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component

# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission magneto-polarization - $\gamma_p$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r' $\gamma_s$ spin relaxation rate $\frac{rad}{ns}$', fontsize =12)#, fontweight="bold") # TODO more applicable definition
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  ['X Linearly Polarized (XLP)', 'Y Linearly Polarized (YLP)'],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)
###################################################################################################
#Varying Electron Lifetime Frequency  𝛾
electron_lifetime_frequency_rate_variations_dataframe = pd.DataFrame()
for g_i in np.linspace(0.9, 1.1, 50):
	vcsel_parameters = {
		"a": a,
		"g_p": g_p,
		"g_s": g_i,
		"g_a": g_a,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 2,
		"P" : 0,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	electron_lifetime_frequency_rate_variations_dataframe = electron_lifetime_frequency_rate_variations_dataframe.append(iteration_dataframe, ignore_index=True)

electron_lifetime_frequency_rate_variations_dataframe.to_csv(
	"simulation_results/electron_lifetime_frequency_rate_variations_"
	 + str(int(time()))
	 + ".csv")
electron_lifetime_frequency_rate_variations_dataframe = pd.read_csv("simulation_results/electron_lifetime_frequency_rate_variations_1589222357.csv")
electron_lifetime_frequency_rate_variations_dataframe
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection = '3d')

dataframe_selection = electron_lifetime_frequency_rate_variations_dataframe[electron_lifetime_frequency_rate_variations_dataframe.simulation_time > 10]
# Make data.
X = dataframe_selection.simulation_time
Y = dataframe_selection.g_s # z axial field component
Z_1 = dataframe_selection.I_x # X-Component
Z_2 = dataframe_selection.I_y # Y-Component

# ax.plot_trisurf(X, Y, Z_1, edgecolor='none', label=r'$I_{RCP}$');
# ax.plot_trisurf(X, Y, Z_2, edgecolor='none', label=r'$I_{LCP}$');
ax.plot_trisurf(X, Y, Z_1, color=color_rcp, alpha=1, edgecolor='none', label=r'$I_{RCP}$');
ax.plot_trisurf(X, Y, Z_2, color=color_lcp, alpha=1, edgecolor='none', label=r'$I_{LCP}$');

# Customize the axes
ax.xaxis.set_major_locator(LinearLocator(4))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.f'))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# ax.set_title(r'Linear emission magneto-polarization - $\gamma$ variations', fontweight="bold", fontsize=16)
ax.set_xlabel(r'$time$ (ns)', fontsize =12)#, fontweight="bold")
ax.set_ylabel(r' $\gamma_p$ Photon Lifetime Frequency $\frac{rad}{ns}$', fontsize =12)#, fontweight="bold") # TODO more applicable definition
ax.set_zlabel(r'Emission Intensity (%)', fontsize =12)#, fontweight="bold")

# Pad Axes Labels
ax.xaxis.labelpad=10
ax.yaxis.labelpad=5
ax.zaxis.labelpad=10

# ax.view_init(elev=0., azim=-210)
ax.view_init(elev=15., azim=-25)
rcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_rcp, marker = 'o')
lcp_surface_buffer_2d_line = mpl.lines.Line2D([0],[0], linestyle="none", c=color_lcp, marker = 'o')
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend([rcp_surface_buffer_2d_line, lcp_surface_buffer_2d_line],
		  ['X Linearly Polarized (XLP)', 'Y Linearly Polarized (YLP)'],
		  numpoints = 1,
		  loc="lower left",
		  bbox_to_anchor=f(0,-0.3,0.95),
		  bbox_transform=ax.transData)

plt.savefig("electron_lifetime_frequency_rate_variations_"
			 + str(int(time()))
			 + ".png", bbox_inches="tight")
#################################################
#Varying Pump Elipticity
# Varing VCSEL parameters
pump_elipticity_variations_dataframe = pd.DataFrame()
for eta_i in np.linspace(0, 10, 50):
	# print("g_p_i: " + str(g_p_i))
	# print("time: " + str(int(time()-initial_time)))
	vcsel_parameters = {
		"a": a,
		"g_p": g_p,
		"g_s": g_s,
		"g_a": g_a,
		"g": g,
		"k": k
	}
	experiment_conditions = {
		"eta" : 1,
		"P" : eta_i,
		"O_x" : 0,
		"O_y" : 0,
		"O_z" : 0
	}

	iteration_dataframe = vcsel_experimental_modelling(vcsel_parameters, experiment_conditions)
	pump_elipticity_variations_dataframe = pump_elipticity_variations_dataframe.append(iteration_dataframe, ignore_index=True)

pump_elipticity_variations_dataframe.to_csv(
	"simulation_results/pump_elipticity_variations_"
	 + str(int(time()))
	 + ".csv")
