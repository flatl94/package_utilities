import os
import sys
import numpy as np
import numdifftools as nd
import copy
from numdifftools import Jacobian
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import scipy
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

import pandas as pd
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LogNorm
import matplotlib.animation as animation

import matplotlib.patches as patches
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D

class UncertaintyPropagation:
    """
    This class performs uncertainty propagation of various quantity.
    
    Args:
    - _sigma_vect: np.array((n_ele))
        Vector storing the variances of a data set
    - _correlation_factor: float
        Correlation factor between quantities. Must be a float in the range [-1.0, 1.0]

    """
    def __init__(self, _sigma_vect, _correlation_factor):
    
        # --- intialisation of uncertainty matrix
        
        n_ele = np.size(_sigma_vect)
        self.__sigma_matrix = np.multiply(_correlation_factor*np.ones((n_ele,n_ele))
                                          +(1-_correlation_factor)*scipy.sparse.eye((n_ele)),
                                _sigma_vect.reshape((n_ele,1)) @ _sigma_vect.reshape((1,n_ele)))

    def uncertainty_propagation(self, _function, _data_vect, opt='vect'):
        """
        Returns the uncertainty (standard deviation) of the
        function _function propagating the uncertainties of set of data.
        Input:
        - _function: lambda function
            Function over which is applied the Jacobian operator, of the type f = lambda x: f(x)
            
        - _data_vect: np.array((n_ele))
            Vector storing the output data for which we want to evaluate the Jacobian matrix
        - opt: str
            returning option: if opt == "matrix" the uncertainty propagated matrix is returned;
            if opt == "vect" the uncertainty vector of the different quantities is returned;
            in any other case, both quantities are returned

        """
        
        # --- Evaluate the Jacobian matrix
        JAC = Jacobian(_function)(_data_vect)
        # --- propagate according to sigma_F^2 = J * sigma_x^2 * J.T
        
        """
        
        To be adapted for MPI processes for faster calculations.
        If possible ALSO perform GPU acceleration.
        
        Add methods to distinguish the two operations.
        
        To be used for matrix multiplication, mostly
                
        """
        _res_sigma_matrix = JAC @ self.__sigma_matrix @ JAC.T
        
        if opt == "vect":
            _res_sigma_vect = np.sqrt(np.diag(_res_sigma_matrix))
            return _res_sigma_vect
        elif opt == "matrix":
            _res_sigma_mat = np.sqrt(_res_sigma_matrix)
            return _res_sigma_mat
        else:
            _res_sigma_vect = np.sqrt(np.diag(_res_sigma_matrix))
            _res_sigma_mat = np.sqrt(_res_sigma_matrix)
            return _res_sigma_vect, _res_sigma_mat

    def sigma_matrix_copy(self):
        sigma_matrix = self.__sigma_matrix
        return sigma_matrix

class FitFunctionConstructor:
    def __init__(self, data_array, var_dict):
        """
        This class will construct a fitting correlation between quantities. In particular:
        Y = F(var_dict['v_1'],...,var_dict['v_N'])
        Args:
            data_array: np.array((dim_v_1,dim_v_2, ..., dim_v_N))
                Data for which a fitting function must be evaluated.
            var_dict: dict
                var_dict.keys() == 'v_{i}'
                    {i} is an integer
                var_dict['v_{i}'].items(): np.array((dim_v_{i}))
                    Elements to be processed
                Array of data over which the fitting function must be constructed.  
        """
        dim_array = np.shape(data_array)
        len_dim_array = len(dim_array)
        for i in range(0,len_dim_array):
            if dim_array[i] != np.size(var_dict['v_'+str(i+1)]):
                raise ValueError('Error! Incompatible dimensions between data.')
        self.__data = data_array
        self.__vars = copy.deepcopy(var_dict)
        self.__dim_data = dim_array
    
    def logistic_fitting(self, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        """
        This function will perform logistic function fitting of data with repsect a single variable.
        F(x) = A*B*exp(C*x)/(A+B*(exp(C*x)-1))+ D
        order: int
            Order of the polynomial fitting function.
        """
        def func(x, A,B,C,D):
            return A*B*np.exp(C*x)/(A+B*(np.exp(C*x)-1))+D

        p_0 = [10.0, 10.0, 0.0001, 10.0]
        par_lb = [-np.inf, -np.inf, -10,-np.inf, -np.inf]
        par_ub = [np.inf, np.inf, 10, np.inf, np.inf]
        popt, pcov = curve_fit(func, self.__vars['v_1'], self.__data, p0=p_0,bounds=(par_lb, par_ub), maxfev=10000)
        fit_data = func(self.__vars['v_1'],*popt)
        figure_of_merit = sum([abs(self.__data[i]-fit_data[i]) for i in range(np.size(self.__data))])/sum([abs(self.__data[i]) for i in range(np.size(self.__data))])
        self.fit_visualizer(fit_data, self.__vars['v_1'], figure_of_merit,_path, _label, option)
        return popt, figure_of_merit

    def explog_fitting(self, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        """
        This function will perform exp-log fitting fitting of data with repsect a single variable.
        F(x) = A*exp(B*x)*(C*x - D ) + E
        order: int
            Order of the polynomial fitting function.
        """
        def func(x, A,B,C,D,E):
            return A*np.exp(B*x)*(C*x-D)+E

        p_0 = [10.0, 0.0001, 10.0, 10.0, 10.0]
        par_lb = [-np.inf, -10, -np.inf, -np.inf, -np.inf]
        par_ub = [np.inf, 10, np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(func, self.__vars['v_1'], self.__data, p0=p_0,bounds=(par_lb, par_ub), maxfev=100000)
        fit_data = func(self.__vars['v_1'],*popt)
        figure_of_merit = sum([abs(self.__data[i]-fit_data[i]) for i in range(np.size(self.__data))])/sum([abs(self.__data[i]) for i in range(np.size(self.__data))])
        self.fit_visualizer(fit_data, self.__vars['v_1'], figure_of_merit,_path, _label, option)
        return popt, figure_of_merit

    def poly_fitting(self,order, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        """
        This function will perform polynomial fitting of data with repsect a single variable.
        F(x) = a_0 + a_1*x + a_2*x**2 + ... + a_N*x**N
        order: int
            Order of the polynomial fitting function.
        """
        def func(x, *params):
            return sum([p*(x**i) for i, p in enumerate(params)])

        popt, pcov = curve_fit(func, self.__vars['v_1'], self.__data, p0=[1]*(order+1))
        fit_data = func(self.__vars['v_1'],*popt)
        figure_of_merit = sum([abs(self.__data[i]-fit_data[i]) for i in range(np.size(self.__data))])/sum([abs(self.__data[i]) for i in range(np.size(self.__data))])
        self.fit_visualizer(fit_data, self.__vars['v_1'], figure_of_merit,_path, _label, option)
        return popt, figure_of_merit
        
    def nexp_fitting(self, order, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        """
        This function will perform polynomial fitting in the form of
        F(x) = a_1*exp(-b_1*x)+a_2*exp(-b_2*x)+...+a_N+1*exp(-b_N*x)
        """
        def func(x, *params):
            return sum([params[p]*np.exp(-params[p+1]*x) for p in range(0,len(params),2)])
        par = []
        par_lb = []
        par_ub = []
        for i1 in range(order):
            par.append(1.0)
            par.append(1.0)
            par_lb.append(0.0)
            par_lb.append(0.000000001)
            par_ub.append(np.inf)
            par_ub.append(10.0)
        
        popt, pcon = curve_fit(func, self.__vars['v_1'],self.__data, bounds=(par_lb, par_ub),p0=par,maxfev=100000)
        fit_data = func(self.__vars['v_1'],*popt)
        figure_of_merit = sum([abs(self.__data[i]-fit_data[i]) for i in range(np.size(self.__data))])/sum([abs(self.__data[i]) for i in range(np.size(self.__data))])
        self.fit_visualizer(fit_data, self.__vars['v_1'], figure_of_merit, _path, _label, option)
        return popt, figure_of_merit

    def pexp_fitting(self, order, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        """
        This function will perform polynomial fitting in the form of
        F(x) = a_1*exp(b_1*x)+a_2*exp(b_2*x)+...+a_N+1*exp(b_N*x)
        """
        def func(x, *params):
            return sum([params[p]*np.exp(params[p+1]*x) for p in range(0,len(params),2)])
        par = []
        par_lb = []
        par_ub = []
        for i1 in range(order):
            par.append(1.0)
            par.append(1.0)
            par_lb.append(0.0)
            par_lb.append(0.000000001)
            par_ub.append(np.inf)
            par_ub.append(10.0)
        
        popt, pcon = curve_fit(func, self.__vars['v_1'],self.__data, bounds=(par_lb, par_ub),p0=par,maxfev=10000)
        fit_data = func(self.__vars['v_1'],*popt)
        figure_of_merit = sum([abs(self.__data[i]-fit_data[i]) for i in range(np.size(self.__data))])/sum([abs(self.__data[i]) for i in range(np.size(self.__data))])
        self.fit_visualizer(fit_data, self.__vars['v_1'], figure_of_merit, _path, _label, option)
        return popt, figure_of_merit

    def fit_visualizer(self, fit_trend, fit_axis,fom, _path, _label, option = {'xaxis':{'label':""},'yaxis':{'label':""}}):
        fig, ax = plt.subplots(figsize= (10,6))
        list_scale = ['asinh', 'function', 'functionlog', 'linear', 'log', 'logit', 'symlog']
        ax.set_title('Comparison of between fitting function and data')
        if 'label' in option['xaxis'].keys():
            ax.set_xlabel(option['xaxis']['label'])
        if 'scale' in option['xaxis'].keys() and option['xaxis']['scale'] in list_scale:
            ax.set_xscale(option['xaxis']['scale'])
        if 'label' in option['yaxis'].keys():
            ax.set_ylabel(option['yaxis']['label'])
        if 'scale' in option['yaxis'].keys() and option['yaxis']['scale'] in list_scale:
            ax.set_yscale(option['yaxis']['scale'])
        ax.plot(fit_axis, fit_trend,linestyle='-.', color = 'b', label = 'fit')
        ax.plot(self.__vars['v_1'], self.__data, marker='*', linestyle='--', color = 'r', label = 'data', markersize = 5)
        name_fig = _path+'/'+_label+'_fit_function.png'
        text_annotation = 'ID calculation: '+_label+'\nFOM = {0:.4f}'.format(fom)
        at = AnchoredText(text_annotation, prop=dict(size=9), frameon=True, loc='lower left')
        
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        at.patch.set_edgecolor('k')
        ax.add_artist(at)
        ax.legend(loc = 'upper right',fontsize=10)
        plt.savefig(name_fig, format = 'png',dpi = 600, bbox_inches = 'tight')
        plt.close()



