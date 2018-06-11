#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test_adptivetimestep_method"""
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
from scipy import integrate
import pytest
from sys import stdout


def create_te_ne_profile(icase=0):
    """
        Temperature and density profiles creator.
    """
    if (icase == 0):
        ntime = 100000
        te_sta = 1.0e5  # Unit: [K]
        te_end = 1.0e7
        ne_max = 1.0e12  # Unit: [cm^-3]
        ne_min = 1.0e7
        time_end = 3600.0  # Unit: [s]

        time_arr = np.linspace(0.0, time_end, ntime)
        te_arr = np.linspace(te_sta, te_end, ntime)
        ne_arr = np.linspace(ne_max, ne_min, ntime)
        return {"te_arr": te_arr, "ne_arr": ne_arr, "time_arr": time_arr}


def test_adapt_time_step(natom=2):
    """
        Set a set of Te/ne profiles. Perfrom NEI calculations using adaptive
        time-step method and compare it with constant dt results.
    """
    #
    # Set Te and rho profiles
    #
    ncase = 1
    for icase in range(ncase):
        data = create_te_ne_profile(icase=icase)
        time_arr = data["time_arr"]  # unit: [s]
        te_arr = data["te_arr"]  # unit: [K]
        rho_arr = data["ne_arr"]  # unit: [cm^-3]

        # Start from equilibrium ionizaiont(EI) states
        table = nei.EigenData2(element=natom)
        f_ini = table.equilibrium_state(T_e=te_arr[0])

        # Case 1: Adaptive time-step:
        newprofile = adaptive_time_step(
            te_arr, rho_arr, time_arr, table, accuracy_factor=1.0e-2)
        print(f"Origin sampling points = ", len(te_arr))
        dt_min = 10000.0
        for i in range(1, newprofile["ntime"]):
            dt_c = newprofile["time"][i] - newprofile["time"][i - 1]
            if (dt_c <= dt_min):
                dt_min = dt_c

        print(f"Adaptive samplinge points =", newprofile["ntime"], "Min_dt=",
              dt_min)
        print(
            f"Time = {time_arr[-1]}, Te_sta={te_arr[0]}, Te_end={te_arr[-1]}")
        print(f"EI_start={f_ini}")

        f0 = np.copy(f_ini)
        for i in range(1, newprofile["ntime"]):
            im = i - 1
            ic = i
            time_im = newprofile["time"][im]
            time_ic = newprofile["time"][ic]

            # Time step between im and ic
            dt_mc = math.fabs(time_ic - time_im)
            # Constant Temperature
            te_mc = newprofile["te"][im]
            # Average Densisty
            ne_mc = newprofile["neavg"][im]

            # NEI one-step
            ft = func_solver_eigenval(natom, te_mc, ne_mc, dt_mc, f0, table)
            f0 = np.copy(ft)

        f_nei_ada_time = ft
        print(f"NEI(adt)={f_nei_ada_time}")

        # Case 2: Constant time-step:
        f0 = np.copy(f_ini)
        time_current = 0.0
        dt = 1.0
        while time_current < time_arr[-1]:
            # The original time interval is 1.4458s in this test
            te_current = np.interp(time_current, time_arr, te_arr)
            ne_current = 0.5 * (
                np.interp(time_current, time_arr, rho_arr) + np.interp(
                    time_current + dt, time_arr, rho_arr))
            ft = func_solver_eigenval(natom, te_current, ne_current, dt, f0,
                                      table)
            f0 = np.copy(ft)
            time_current = time_current + dt
            stdout.write("\r%f" % time_current)
            stdout.flush()
        stdout.write("\n")
        dt = time_arr[-1] - time_current
        te_current = np.interp(time_current, time_arr, te_arr)
        ne_current = np.interp(time_current, time_arr, rho_arr)
        ft = func_solver_eigenval(natom, te_current, ne_current, dt, f0, table)
        ft_nei = ft

        # final results
        f_ei_end = table.equilibrium_state(T_e=te_arr[-1])
        print(f"NEI(cdt)={ft_nei}")
        diff_adp_cdt = (ft_nei - f_nei_ada_time) / np.amax(
            [ft_nei, f_nei_ada_time])
        print(f"Dif_a&c ={diff_adp_cdt}")
        print(f"EI_end  ={f_ei_end}")

        #
        # Output results
        #
    return 1


def adaptive_time_step(tein, nein, timein, table, accuracy_factor=1.0e-4):
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
        outprofile -- dictionary including new time nodes.
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

        dt_ne_m = accuracy_factor / (np.amax(evalues_m) * nem)
        dt_ne_c = accuracy_factor / (np.amax(evalues_c) * nec)
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

    # Step 4: Compute average density
    ntime = len(timenew3)
    neavg = np.zeros(ntime)
    for i in range(1, ntime):
        im = i - 1
        ic = i
        time_im = timenew3[im]
        time_ic = timenew3[ic]

        res = np.where(timein >= time_im)
        itime_sta = res[0][0]
        res = np.where(timein >= time_ic)
        itime_end = res[0][0]
        ditime = itime_end - itime_sta
        if (ditime >= 1):
            # Include density contributed by skipped sampling points
            rho_piece = nein[itime_sta:itime_end + 1]
            time_piece = timein[itime_sta:itime_end + 1]
            dt_time = time_ic - time_im
            ne_avgc = integrate.simps(rho_piece, time_piece) / dt_time
        else:
            # On inserted nodes, linear interplote ne_mc
            ne_avgc = 0.5 * (nenew3[im] + nenew3[ic])

        # Save into array: rhoavg[im]
        neavg[im] = ne_avgc

    # Set a dictionary to save/return results
    outprofile = {
        "time": timenew3,
        "ntime": ntime,
        "te": tenew3,
        "neavg": neavg
    }
    return outprofile


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