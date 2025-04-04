import numpy as np
from itertools import product
import numpy as np

from contraction import  UJ_contract

def UJAFclock(q,J):
    T_up1 = np.zeros((q,q,q),dtype=np.float64)
    delta1 = np.zeros((q,q,q,q),dtype=np.float64)
    delta2 = np.zeros((q,q,q),dtype=np.float64)
    delta2 = np.zeros((q,q,q,q,q,q,q,q),dtype=np.float64)
    for i1,i2,i3 in product(range(q),repeat=3):
        T_up1[i1,i2,i3]  = np.exp(-0.5*J*(np.cos(2*np.pi*(i1-i2)/q)+np.cos(2*np.pi*(i2-i3)/q)+np.cos(2*np.pi*(i3-i1)/q)))
    for i in range(q):
        delta1[i,i,i,i] = 1
        delta2[i,i,i,i,i,i,i,i] = 1
    
    TA = UJ_contract(T_up1,T_up1,delta1)
    delta2 = delta2.reshape(q**2,q**2,q**2,q**2,order="F")
    TA = TA.reshape(q**2,q**2,q**2,q**2,order="F")

    Ts = []

    Ts.append(TA)
    Ts.append(delta2)
    Ts.append(TA.transpose(2,3,0,1))
    Ts.append(delta2)

    return Ts

def UJAFclock_chiral(q,J):
    T_up1 = np.zeros((q,q,q),dtype=np.float64)
    T_chiral = np.zeros((q,q,q),dtype=np.float64)
    delta1 = np.zeros((q,q,q,q),dtype=np.float64)
    delta2 = np.zeros((q,q,q),dtype=np.float64)
    delta2 = np.zeros((q,q,q,q,q,q,q,q),dtype=np.float64)
    for i1,i2,i3 in product(range(q),repeat=3):
        T_up1[i1,i2,i3]  = np.exp(-0.5*J*(np.cos(2*np.pi*(i1-i2)/q)+np.cos(2*np.pi*(i2-i3)/q)+np.cos(2*np.pi*(i3-i1)/q)))
        T_chiral[i1,i2,i3] = (np.sin(2*np.pi*(i1-i2)/q)+np.sin(2*np.pi*(i2-i3)/q)+np.sin(2*np.pi*(i3-i1)/q))*T_up1[i1,i2,i3]
    for i in range(q):
        delta1[i,i,i,i] = 1
        delta2[i,i,i,i,i,i,i,i] = 1
    delta2 = delta2.reshape(q**2,q**2,q**2,q**2,order="F") 
    TA = UJ_contract(T_up1,T_up1,delta1)
    TA = TA.reshape(q**2,q**2,q**2,q**2,order="F")

    TB = UJ_contract(T_up1,T_chiral,delta1)
    TB = TB.reshape(q**2,q**2,q**2,q**2,order="F")

    Ts = []

    Ts.append(TA)
    Ts.append(delta2)
    Ts.append(TB.transpose(2,3,0,1))
    Ts.append(delta2)

    return Ts

