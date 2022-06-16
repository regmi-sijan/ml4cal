import  numpy as np

def landau(x, *par):
    '''
    par[0] - origin
    par[1] - scale
    par[2] - scaling for the exponent
    par[3] - pedestal    
    par[4] - decay in the second term
    '''

    w = x - par[0]

    divider = par[2]
    if divider < 0.01: divider = 0.01

    scaler = par[4]
    if scaler<0.01:    scaler=0.01
    if scaler>20.0:     scaler=20.0

    my_exp = np.exp(-(w+np.exp(-float(par[4])*w))/divider)

    scaled = par[1]*my_exp
    return scaled+par[3]

###

