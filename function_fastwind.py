import os
import numpy as np
import collections

def gauss(x,a,x0,sigma):
    '''
    Gaussian with continuum at 1.
    '''
    if sigma == 0:
        return x*0.0
    else:
        y = a * np.exp(-(x-x0)**2 / (2*sigma**2))
        return y

def mock_function_x(x, teff, logg, mdot, He, vrot, N, Si, micro):
    # y = d*np.sin(x*a) + x*b + x**(c*0.5) - x*c - d - x**(g*0.5)
    # y = a + b*x + 0.1*c*x**2 - 0.1*c*(x-d)**2 + 0.1*c*(x-dg)**2#+ d*x**3 - g*x**3
    # y = np.cos(teff*x) + np.sin(logg*x) + np.cos(mdot*x**He)- np.sin(vrot*(x-N)) + np.sin(micro*(x-Si))#+ d*x**3 - g*x**3

    # line1 = gauss(x, teff*0.3*(1-He)*(1-mdot), 5. , 0.2*logg*vrot)
    # line2 = gauss(x, teff*0.3*He, 19., 0.4*logg*vrot)
    # line3 = gauss(x, teff*0.7*Si*(micro*0.5), 22., 0.1*logg*vrot)
    # line4 = gauss(x, teff*0.3*He, 31., 0.2*logg*vrot)
    # line5 = gauss(x, teff*0.4*Si*micro, 37., 0.4*logg*vrot)
    # line6 = gauss(x, teff*0.3*N*(1-mdot)*(1-micro), 43., 0.4*logg*vrot)
    # line7 = gauss(x, teff*0.4*N*micro, 47., 0.4*logg*vrot)
    # y = 1 - (line1 + line2 + line3 + line4 + line5 + line6 + line7)


    line1 = gauss(x, (1-teff)*0.3*(1-He), 5. , 0.5*(2+logg)*vrot)
    line2 = gauss(x, teff*0.3*He, 19., 0.5*(2+logg)*vrot) - gauss(x, mdot*0.2, 5. , 0.5*(2+logg)*vrot)
    line3 = gauss(x, (0.8-teff)**0.5*0.7*Si*(micro*0.5), 22., 0.5*(2+logg*3)*vrot)
    line4 = gauss(x, (0.8-teff)*0.3*He, 31., 0.5*(2+logg)*vrot)
    line5 = gauss(x, teff*0.5*Si*micro, 37., 0.5*(2+logg*5)*vrot) - gauss(x, mdot*0.3, 37. , 0.5*(2+logg)*vrot)
    line6 = gauss(x, (1-teff)**0.5*0.3*N*(1-micro), 43., 0.5*(2+logg*3)*vrot)
    line7 = gauss(x, teff*0.4*N*micro, 47., 0.5*(2+logg)*vrot)
    y = 1 - (line1 + line2 + line3 + line4 + line5 + line6 + line7)

    return y

def mock_function(teff, logg, mdot, He, vrot, N, Si, micro):
    x = np.linspace(0, 50, 1000)
    y = mock_function_x(x, teff, logg, mdot, He, vrot, N, Si, micro)
    return y

def eval_fitness_ind(model, obs_flux, obs_err):

    chi2 = np.sum(((obs_flux - model)/obs_err)**2)

    return chi2

def calculate_models(parameter_input):

    output_array = []
    for individual in parameter_input:
        modelout = mock_function(*individual)
        output_array.append(modelout)
    return output_array

def evaluate_fitness(output_array, data):

    data_y, data_err = data[1], data[2]

    chi2list = []
    for individual in output_array:
        chi2 = eval_fitness_ind(individual, data_y, data_err)
        chi2list.append(chi2)

    fitness = chi2list

    return fitness

def read_paramspace(param_source):
    """ Read the parameter space from text file

    #FIXME account for parameters that don't vary
    """

    dtypes = (str, float, float, float)
    pspace = np.genfromtxt(param_source, dtype=str).T

    ndecimals = []
    for x in pspace[3]:
        if '.' in x:
            ndecimals.append(len(x.split('.')[-1]))
        else:
            ndecimals.append(0)
    ndecimals = np.array(ndecimals)

    names = pspace[0]
    values = pspace[1:].astype(float).T

    variable_names = []
    variable_vals = []
    fixed_names = []
    fixed_vals = []

    for apname, pval, ndec in zip(names, values, ndecimals):
        if pval[0] == pval[1]:
            fixed_names.append(apname)
            fixed_vals.append(pval)
        else:
            new_pval = np.concatenate((np.array(pval), np.array([int(ndec)])))
            variable_names.append(apname)
            variable_vals.append(new_pval)

    return variable_names, variable_vals, fixed_names, fixed_vals
