#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test_read_simulationdata"""
import sys
import math
sys.path.append('../../../')
import warnings
import glob
import numpy as np
import numba as nb
from plasmapy import atomic
import nei as nei
from scipy.io import readsav
import pytest
from sys import stdout


def test_nei_multisteps(natom=8):
    """
        Read temprature and density history file and perfrom NEI calculations.
        Starting the equilibrium states, and set any time-step in here
        (adaptive time-step is required in practice).
    """
    #
    # Read Te and rho profiles
    #
    file_path = '/Users/chshen/Works/Project/REU/REU_2018/2018_SummerProject/cr1913_traj_180601/'
    files = sorted(glob.glob(file_path + '*.sav'))

    for file in files:
        data = readsav(file, python_dict=True)
        time = data['time']  # unit: [s]
        te = data['te']  # unit: [K]
        rho = data['rho']  # unit: [cm^-3]
        v = data['v']  # unit: [km/s]
        r = data['r']  # r position in sphericalcoordinate: [Rolar radio]
        t = data['t']  # t position in sphericalcoordinate: [0 ~ PI]
        p = data['p']  # p position in sphericalcoordinate: [0 ~ 2PI]

        #
        # NEI calculations parametes: dt and element
        #
        dt = 0.1  # 0.1s

        # Start from equilibrium ionizaiont(EI) states
        time_current = 0
        table = nei.EigenData2(element=natom)
        f_ini = table.equilibrium_state(T_e=te[0])

        print('START:')
        print(f'time_sta = ', time_current, te[0])
        print(f_ini)

        # Get adaptive time-step
        newprofile = adaptive_time_step(te, rho, time, table)
        print(f"Origin points = ", len(te))
        dt_min = 10000.0
        for i in range(1, newprofile["ntime"]):
            dt_c = newprofile["time"][i] - newprofile["time"][i - 1]
            if (dt_c <= dt_min):
                dt_min = dt_c

        print(f"Adaptive time-step =", newprofile["ntime"], "min_dt=", dt_min)

        # Enter the time-advance using the adaptive time-step:
        f0 = np.copy(f_ini)
        for i in range(1, newprofile["ntime"]):
            im = i - 1
            ic = i
            dt_mc = math.fabs(newprofile["time"][ic] - newprofile["time"][im])
            te_mc = newprofile["te"][im]
            ne_mc = 0.5 * (newprofile["ne"][im] + newprofile["ne"][ic])
            ft = func_solver_eigenval(natom, te_mc, ne_mc, dt_mc, f0, table)
            f0 = np.copy(ft)
            stdout.write("\r%f" % time_current)
            stdout.flush()
        stdout.write("\n")
        f_nei_ada_time = ft
        print(f"NEI(adt)={f_nei_ada_time}")

        # Enter the time loop using constant time-step = 0.1s:
        f0 = np.copy(f_ini)
        while time_current < time[-1]:
            # The original time interval is 1.4458s in this test
            te_current = np.interp(time_current, time, te)
            ne_current = 0.5 * (np.interp(time_current, time, rho) + np.interp(
                time_current + dt, time, rho))
            ft = func_solver_eigenval(natom, te_current, ne_current, dt, f0,
                                      table)
            f0 = np.copy(ft)
            time_current = time_current + dt
            stdout.write("\r%f" % time_current)
            stdout.flush()
        stdout.write("\n")

        # The last step
        dt = time[-1] - time_current
        te_current = np.interp(time_current, time, te)
        ne_current = np.interp(time_current, time, rho)
        ft = func_solver_eigenval(natom, te_current, ne_current, dt, f0, table)

        # final results
        f_ei_end = table.equilibrium_state(T_e=te[-1])

        print(f'time_end = ', time_current, te[-1])
        print(f"EI :", f_ei_end)
        print(f"NEI(cdt)={ft}")

        #
        # Output results
        #
    return 1


def func_solver_eigenval(natom, te, ne, dt, f0, table):
    """
        The function for one step time_advance.
    """

    common_index = table._get_temperature_index(te)
    evals = table.eigenvalues(
        T_e_index=common_index)  # find eigenvalues on the chosen Te node
    evect = table.eigenvectors(T_e_index=common_index)
    evect_invers = table.eigenvector_inverses(T_e_index=common_index)

    # define the temperary diagonal matrix
    diagona_evals = np.zeros((natom + 1, natom + 1))
    for ii in range(0, natom + 1):
        diagona_evals[ii, ii] = np.exp(evals[ii] * dt * ne)

    # matirx operation
    matrix_1 = np.dot(diagona_evals, evect)
    matrix_2 = np.dot(evect_invers, matrix_1)

    # get ions fraction at (time+dt)
    ft = np.dot(f0, matrix_2)

    # re-check the smallest value
    minconce = 1.0e-15
    for ii in np.arange(0, natom + 1, dtype=np.int):
        if (abs(ft[ii]) <= minconce):
            ft[ii] = 0.0
    return ft


def adaptive_time_step(tein, nein, timein, table, accuracy_factot=1.0e-7):
    """
        A method to get adaptive time steps along the Temperature(or density)
        profile. The new time nodes will be created according to
        adaptive time-steps either through interpolating or skipping.

        Keyword arguments:
        tein -- one dimensional array for temperature (unit: K)
        nein -- one dimensional array for density (unit: cm^-3)
        timein -- one dimensional array for time (unit: s)
        table -- 'Eigentable2' class for a perticular element
        accuracy_factot -- accuracy factor to estimate time-step. It will be 
            used to set the allowed maximum changes on charge states.

        Return/Output:
        outprofile -- dictionary including new time, te, and ne profiles.
    """
    # Check the size of input profiles.
    ntime = len(timein)
    if (ntime <= 1):
        raise NameError

    # Te grid interval from Eigentables.
    dte_grid = math.log10(table.temperature_grid[1]) \
    - math.log10(table.temperature_grid[0])

    # Step (1): Check how many points can be skipped based on the dte_grid?
    # Initialize array to save results.
    timenew = [timein[0]]
    timenew_index = [0]
    te_last_point = tein[0]

    for i in range(1, ntime):
        # Set time index: m means 'minus', and c means 'current'.
        im = i - 1
        ic = i

        # Get temperatures.
        tem = tein[im]
        tec = tein[ic]
        dte_current = math.fabs(math.log10(tec) \
                    - math.fabs(math.log10(te_last_point)))

        # Compare it with dte_grid.
        if (dte_current >= dte_grid):
            # Keep this points
            timenew.append(timein[ic])
            timenew_index.append(ic)
            te_last_point = tein[ic]

    # Step (2): Re-check the survivors and see if they can be skipped based on
    # Eigenvalues
    # Initialize array to save results.
    timenew2 = [timenew[0]]
    timenew2_index = [timenew_index[0]]

    for i2 in range(1, len(timenew_index)):
        # Set index:
        im = timenew_index[i2 - 1]
        ic = timenew_index[i2]

        # Get temperature and density at the survivors
        tem = tein[im]
        tec = tein[ic]
        nem = nein[im]
        nec = nein[ic]

        # Check time-step using density and eigenvalue and find
        # the minimum time-step during im->ic.
        evalues_m = table.eigenvalues(T_e=tem)
        evalues_c = table.eigenvalues(T_e=tec)

        dt_ne_m = accuracy_factot / (np.amax(evalues_m) * nem)
        dt_ne_c = accuracy_factot / (np.amax(evalues_c) * nec)
        dt_ne = np.amin([dt_ne_m, dt_ne_c])

        # Compare it with the survivor's time interval.
        dt_current = math.fabs(timenew[i2] - timenew[i2 - 1])

        if (dt_ne < dt_current):
            # Keep this point
            timenew2.append(timenew[i2])
            timenew2_index.append(ic)

    # Step (3): Check it again and see if it should be interpolated based on
    # dte_grid?
    # Initialize array to save results.
    timenew3 = [timenew2[0]]
    tenew3 = [tein[timenew2_index[0]]]
    nenew3 = [nein[timenew2_index[0]]]

    for i3 in range(1, len(timenew2_index)):
        # Set index:
        im = timenew2_index[i3 - 1]
        ic = timenew2_index[i3]

        # Get temperature and density
        tem = tein[im]
        tec = tein[ic]
        nem = nein[im]
        nec = nein[ic]
        timem = timein[im]
        timec = timein[ic]
        dte_current = math.fabs(math.log10(tec) \
                    - math.fabs(math.log10(tem)))

        # Compare it with dte_grid
        if (dte_current < dte_grid):
            # Keep the same point
            timenew3_index.append(ic)
            tenew3.append(tec)
            nenew3.append(nec)
        else:
            # Linear interpolation and insert new points
            n_interp = int(dte_current / dte_grid) + 1
            dd_time = (timec - timem) / n_interp
            dd_te = (tec - tem) / n_interp
            dd_ne = (nec - nem) / n_interp

            # Insert new points
            for j in range(1, n_interp):
                timenew3.append(timem + j * dd_time)
                tenew3.append(tem + j * dd_te)
                nenew3.append(nem + j * dd_ne)

            # Attach the final (original) point
            timenew3.append(timec)
            tenew3.append(tec)
            nenew3.append(nec)

    # TO DO:
    # Compute average density between two chosen points
    # ...

    # Set a dictionary to save/return results
    outprofile = {
        "time": timenew3,
        "te": tenew3,
        "ne": nenew3,
        "ne_avg": neavg,
        "ntime": len(timenew3)
    }
    return outprofile
