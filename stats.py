import numpy as np
from scipy.optimize import curve_fit

def gauss(x, A0, mu0, sigma0, y0):
    return A0*np.exp(-1/2*((x-mu0)/sigma0)**2) + y0

def double_gauss(x, A0, mu0, sigma0, y0, A1, mu1, sigma1, y1):
    return A0*np.exp(-1/2*((x-mu0)/sigma0)**2) + y0 + A1*np.exp(-1/2*((x-mu1)/sigma1)**2)+y1

def super_gauss(x, A, mu, sigma, n, y0):
    return A * np.exp( -1/2*(np.abs(x-mu)/sigma)**n ) + y0

def double_super_gauss(x, A0, A1, mu0, mu1, sigma0, sigma1, n0, n1, y0):
    return A0 * np.exp( -1/2*(np.abs(x-mu0)/sigma0)**n0 ) + A1 * np.exp( -1/2*(np.abs(x-mu1)/sigma1)**n1 ) + y0

def fit_data(x, y, method="gauss", bound_at_peak=False, double_gauss_iter=1):
    """
    Fit data with one of three methods
    """
    if method.lower() == "gauss":
        A0 = np.max(y)
        mu0 = x[np.argmax(y)]
        sigma0 = (x[-1]-x[0])/4
        y0 = np.median(y)
        if y0<0:
            y0=1e-10
        
        p0 = [A0, mu0, sigma0, y0]
        if bound_at_peak:
            bounds = ([0, mu0-1e-12, (x[1]-x[0])*2, 0], [np.max(y)*2, mu0+1e-12, x[-1]-x[0], np.max(y)])
        else:
            bounds = ([0, x[0], (x[1]-x[0])*2, 0], [np.max(y)*2, x[-1], x[-1]-x[0], np.max(y)])
        
        params, cov = curve_fit(gauss, x, y, p0=p0, bounds = bounds)
        
    elif method.lower() == "double_gauss":
        if double_gauss_iter==1:
            A0 = [np.max(y), np.max(y)]
            mu0 = x[np.argmax(y)]
            mu0 = [mu0-(x[-1]-x[0])/50, mu0+(x[-1]-x[0])/50]
            for i,mu in enumerate(mu0):
                if mu<x[0]:
                    mu0[i]=x[0]
                elif mu>x[-1]:
                    mu0[i]=x[-1]
                    
            sigma0 = [(x[-1]-x[0])/4, (x[-1]-x[0])/4]
            y0 = [np.min(y), np.min(y)]
            
            for i, y_ in enumerate(y0):
                if y_<0:
                    y0[i]=0
            
            p0 = [A0[0], mu0[0], sigma0[0], y0[0], A0[1], mu0[1], sigma0[1], y0[1]]
            
            bounds = ([0, x[0], (x[1]-x[0])*2, 0]*2,
                     [np.max(y), x[-1], (x[-1]-x[0])*2, np.max(y)]*2)
            
            params, cov = curve_fit(double_gauss, x, y, p0=p0, bounds =bounds)
        else:
            params = []
            y = y.astype(np.float64)
            for _ in range(2):
                params_ = fit_data(x, y, method='gauss', bound_at_peak=True)
                params.extend(params_)
                y_gauss = gauss(x, *params_) 
                y -= y_gauss
        
        return params

    elif method.lower() == "super_gauss":
        A0 = np.max(y)
        mu0 = x[np.argmax(y)]
        sigma0 = (x[-1]-x[0])/4
        y0 = np.min(y)
        if y0<0:
            y0=0
        n0=2
        
        p0 = [A0, mu0, sigma0, n0, y0]
        bounds = ([0, x[0], (x[1]-x[0])*5, .5, 0], [np.max(y)*2, x[-1], x[-1]-x[0], 10, np.max(y)])
        
        params, cov = curve_fit(super_gauss, x, y, p0=p0, bounds=bounds)

    elif method.lower() == "double_super_gauss":
        A0 = np.max(y)
        mu0 = x[np.argmax(y)]-(x[-1]-x[0])/50
        if mu0<x[0]:
            mu0=x[0]
        elif mu0>x[-1]:
            mu0=x[-1]
        sigma0 = (x[-1]-x[0])/4
        y0 = np.mean(y)/10
        if y0<0:
            y0=0
        n0=2

        A1 = np.max(y)
        mu1 = x[np.argmax(y)]+(x[-1]-x[0])/50
        if mu1<x[0]:
            mu1=x[0]
        elif mu1>x[-1]:
            mu1=x[-1]
        sigma1 = (x[-1]-x[0])/4
        n1=2
        
        p0 = [A0, A1, mu0, mu1, sigma0, sigma1, n0, n1, y0]
        bounds = ([0, 0, x[0], x[0], (x[1]-x[0])*5, (x[1]-x[0])*5, .5, .5, 0],
                 [np.max(y)*2, np.max(y)*2, x[-1], x[-1], (x[-1]-x[0])*2, (x[-1]-x[0])*2, 10, 10, np.max(y)])
        
        params, cov = curve_fit(double_super_gauss, x, y, p0=p0, bounds=bounds)
    else:
        print("No valid selections")
    return params