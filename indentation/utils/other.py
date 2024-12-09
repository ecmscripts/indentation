import numpy as np 

def get_pdms_stiffness(ratio):
    '''
    Gives PDMS stiffness in [kPa] when the ratio is entered as ratio:1 mixing ratio.
    '''
    return 1000*5.6*np.exp(-5.4*(ratio/40))
