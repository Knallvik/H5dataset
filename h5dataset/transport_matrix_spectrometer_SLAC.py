import numpy as np

def calc_TransportMatrix(zob,zim, B_Q0D, B_Q1D, B_Q2D, BeamEnergy=10):
        
    OO = np.zeros((2,2))
    def M_drift(d):
        m = np.array([[1, d], [0, 1]])
        M = np.block([[m, OO], [OO, m]])
        return M
    L_eff = 1 # Effective quad length
    
    def M_quad(B):
        if B==0:
            M = M_drift(L_eff)
        else:
            k = 0.299792458*abs(B*0.1)/E
    
            phi = L_eff*np.sqrt(k)
            m_F = np.array([ [np.cos(phi),           (1/np.sqrt(k))*np.sin(phi)],
                   [-np.sqrt(k)*np.sin(phi),   np.cos(phi)]])
            m_D = np.array([ [np.cosh(phi),          (1/np.sqrt(k))*np.sinh(phi)],
                    [np.sqrt(k)*np.sinh(phi),  np.cosh(phi)]])
            if B>0:
                M = np.block([[m_F, OO], [OO, m_D]])
            else:
                M = np.block([[m_D, OO], [OO, m_F]])
        return M
        
    # Define magnet positions
    zQ0D = 1996.98244
    zQ1D = 1999.20656
    zQ2D = 2001.43099
    
    # Calculate the drift distances
    
    d1 = (zQ0D-L_eff/2) - zob 
    d2 = (zQ1D-L_eff/2) - (zQ0D+L_eff/2)
    d3 = (zQ2D-L_eff/2) - (zQ1D+L_eff/2)
    d4 = zim - (zQ2D+L_eff/2)
    
    # Calculate the transport matrix
    E = BeamEnergy
    Mi = M_drift(d4)@M_quad(B_Q2D)@M_drift(d3)@M_quad(B_Q1D)@M_drift(d2)@M_quad(B_Q0D)@M_drift(d1)
    
    return Mi

        

            

    


