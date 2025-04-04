import numpy as np
from ncon import ncon
import scipy.linalg as scl
from LOBPCG_transfer_L4 import LOBPCG_transfer_L4

from NNR_loop_optimization_modified import optimize_aug

def Ising_tensor(temp,h):
    beta = 1/temp
    H_local = np.array([[-1.-h/2,1.],[1.,-1.+h/2]])
    Q = np.exp(-beta*H_local)
    g = np.zeros((2,2,2,2))
    g[0,0,0,0] = 1.
    g[1,1,1,1] = 1.
    Qsr = scl.sqrtm(Q)
    T = ncon([g,Qsr,Qsr,Qsr,Qsr],[[1,2,3,4],[-1,1],[-2,2],[3,-3],[4,-4]])
    Ts = []
    for i in range(4):
        Ts.append(T)
    return Ts

def normalize_T(Ts,g):
    for i in range(4):
        Ts[i] = Ts[i]/(g**(1/4))
    return Ts
def LN_renormalization(Ts):
    res_Ts= []

    T1 = ncon([Ts[7],Ts[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[3],Ts[0]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3]))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3]))
    res_T0 = (T1@T2)
    res_T0 = res_T0.reshape(Ts[0].shape[2],Ts[0].shape[2],Ts[0].shape[2],Ts[0].shape[2])


    T1 = ncon([Ts[1],Ts[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[5],Ts[2]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3]))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3]))
    res_T1 = (T1@T2)
    res_T1 = res_T1.reshape(Ts[0].shape[2],Ts[0].shape[2],Ts[0].shape[2],Ts[0].shape[2])


    res_T2 = res_T0.transpose(2,3,0,1)
    res_T3 = res_T1.transpose(2,3,0,1)

    res_Ts.append(res_T0)
    res_Ts.append(res_T1)
    res_Ts.append(res_T2)
    res_Ts.append(res_T3)
    return res_Ts

def transfer_matrix(Ts):

    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])
    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2

    eig , _ =  np.linalg.eig(M)
    eig = -np.sort(-eig)

    g = np.trace(M)

    central_charge = (6/(np.pi))*np.log(eig[0]/(g*g))
    scaling_dims = -(0.5/np.pi)*np.log(eig[1:41]/eig[0])

    
    return central_charge,scaling_dims
def Norm_Ts(Ts):

    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])



    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2
    g = np.trace(M)
    return g

def LN_TRG_decomp(Ts,chi):
    chi =chi
    LN_decomp= []
    for i in range(4):
        size1 = np.shape(Ts[i])

        if (Ts[0].shape[0]*Ts[0].shape[1]) >  chi :
            u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False, hermitian=False)

            u = np.real(u[:,:chi])
            s = s[:chi]
            v = np.real(v[:chi,:])

        else :
            u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False, hermitian=False)

        s1 = u@np.sqrt(np.diag(s))
        s2 = np.sqrt(np.diag(s))@v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)]))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]]))
    return LN_decomp

def LN_TRG_decomp_prime(Ts,chi):
    chi =chi*chi
    LN_decomp= []
    for i in range(4):
        size1 = np.shape(Ts[i])
        u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)
        if len(s) >  chi :
            u = u[:,:chi]
            s = s[:chi]
            v = v[:chi,:]
        size2 = np.shape(np.diag(s))
        s1 = u@np.sqrt(np.diag(s))
        s2 = np.sqrt(np.diag(s))@v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)]))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]]))
    return LN_decomp
def gauge_invariant(Ts):
    X = ncon(Ts[0],[1,2,1,2])
    X1 = ncon([Ts[0],Ts[3]],[[1,2,3,2],[4,3,4,1]])
    X2 = ncon([Ts[0],Ts[3]],[[1,2,3,4],[2,3,4,1]])
    X1 = (X**2)/X1
    X2 = (X**2)/X2
    return X1,X2



def conformal_data_L4(Ts,Ss,g):
    """
    This function compute the conformal data including central charge, scaling dimension and OPE coeffcients from the fixed-point tensor.
    Here, we use the LOBCG algorithm for eigenvalue decomposition.

    For the detailed discussion of OPE coefficients, reader are refered to PhysRevB.108.024413.
    
    """
    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])
    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3],order="F")
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3],order="F")
    M = M1@M2

    eig_l2 , eig_vecL2 = np.linalg.eig(M)

    idx =  np.argsort(-eig_l2)  
    eig_l2 = eig_l2[idx]
    eig_vecL2 = eig_vecL2[:,idx]

    k = 20
    eig_vecL4 ,eig = LOBPCG_transfer_L4(Ss, k, 40, 1e-4, debug=True)
    eig = eig[:k]
    idx = np.argsort(-eig)  
    eig = eig[idx]
    eig_vecL4 = eig_vecL4[:,idx]

    central_charge = 2*(6/(np.pi))*(np.log(eig[0]/(g**2)))
    scaling_dims = -2*(0.5/np.pi)*np.log(eig[1:k]/eig[0])

    def compute_OPE(table, sd, di, dj, dk):
        AIII = np.abs(table[0,0,0])
        Aabc= np.abs(table[di, dj, dk])
        c = Aabc/(AIII*(2**(sd[di]-2*sd[dj]-2*sd[dk])))
        return c

    fusion_table = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                temp = int(np.sqrt(len(eig_vecL4[:,i])))
                fusion_table[i, j, k]  =  np.real(ncon([eig_vecL4[:,i].reshape(temp,temp,order="F"),eig_vecL2[:,j], eig_vecL2[:,k]], [[2,1], [1], [2]]))
  
    scaling_dimension = np.array([0, scaling_dims[0], scaling_dims[1]])


    counter=0
    OPE = np.zeros(12)
    for i,j,k in [[0,1,1],[1,0,1],[1,1,0],[0,2,2],[2,0,2],[2,2,0],[2,1,1],[1,2,1],[1,1,2],[1,2,2],[2,1,2],[2,2,1]]:
        
        a,b,c=(i,j,k)
        Cabc=compute_OPE(fusion_table, scaling_dimension,a,b,c)
        OPE[counter] =  np.real(np.round(Cabc,5))
        counter += 1


    return central_charge,scaling_dims,OPE


def NNR_TNR(Ts,FILT_EPS, FILT_MAX_I, OPT_EPS, loop_iter,RG_I,chi,temp,K,rho,solver_eps):

    G = 1

    C = 0
    N = 1
    Nplus = 2


    central_c = 0
    scaling_dims=[0]
    for i in range(RG_I):
        print("\n//----------- Renormalization step:   "+ str(i)+ " -----------\n")

        eight_tensors = LN_TRG_decomp(Ts,chi)
        eight_tensors_p= LN_TRG_decomp(Ts,chi**2)
        eight_tensors = optimize_aug(eight_tensors,eight_tensors_p,loop_iter,K,rho,solver_eps)


        if i > 1:
           
            central_c ,scaling_dims,OPE = conformal_data_L4(Ts,eight_tensors,G)
 
            print("\n * central charge L4 :       " +str(central_c)+"\n")
            print("\n * scaling dims L4 :       " +str(scaling_dims[:20])+"\n")

        Ts = LN_renormalization(eight_tensors)

        G0 = G
        G = Norm_Ts(Ts)

        central_c_l2,scaling_dims_l2 = transfer_matrix(Ts)
        print("\n * central charge  L2:       " +str(central_c_l2)+"\n")

        Ts = normalize_T(Ts,G)

        C = np.log(G**(1/4))+Nplus*C
        N *= Nplus
        Z= G


    print("\n * NNR_TNR :      \n")


    return Ts,central_c,np.real(scaling_dims)

import argparse

"""
    A sample implementation of Appendix 2 in arXiv:2403.17309.  

    This code attemps to extract the conformal data of central charge, scaling dimension and OPE coefficients of Ising CFT via the LOBCG algorithm.

    Below, we list several hyper-parameters for users to tune. 

    Parameters:
    chi :  Bond dimension
    temp_ratio : Temperature ratio T/Tc, where Tc refers to the exact transition temperature of 2D classical Ising model.
    RG_step : Number of  RG step.
    OPT_EPS: Stopping threshold for NNR loop optimization.

    Hyper-parameters:
    xi_hyper: the penalty parameter introduced in our paper. If one increases xi, the lower-rank solutions are induced.
    rho_hyper: the penalty schedule parameter.

    OPT_MAX_I: Number of maximum (sweep) iterations for NNR loop optimization. 
    solver_eps: Cut-off ratio for small singular values in the linear-matrix solver. Generally, the smaller the better.

"""

parser = argparse.ArgumentParser(
        description="arXiv:2403.17309",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("chi", type=int,nargs="?", help="Bond dimension",default=6)
parser.add_argument("temp_ratio", type=float,nargs="?",help="temp ratio",default=1)
parser.add_argument("q_state", type=int,nargs="?",help="q",default=6)
parser.add_argument("RG_step", type=int,nargs="?",help="RG_step",default=31)
parser.add_argument("K_hyper", type=float,nargs="?",help="K_hyper",default= 2E-6)
parser.add_argument("rho_hyper", type=float,nargs="?",help="rho_hyper",default= 0.9)

parser.add_argument("OPT_EPS", type=float,nargs="?",help="OPT_EPS ",default= 1E-15)
parser.add_argument("OPT_MAX_I", type=int,nargs="?",help="OPT_MAX_I",default=30)
parser.add_argument("solver_eps", type=int,nargs="?",help="solver_eps",default=1E-15)

parser.add_argument("FILT_EPS", type=float,nargs="?",help="FILT_EPS",default=1E-15)
parser.add_argument("FILT_MAX_I", type=int,nargs="?",help="FILT_MAX_I",default=100)

args = parser.parse_args()
chi = args.chi
q = args.q_state
temp_ratio = args.temp_ratio
RG_step = args.RG_step
FILT_EPS =  args.FILT_EPS
FILT_MAX_I = args.FILT_MAX_I
OPT_EPS = args.OPT_EPS
OPT_MAX_I = args.OPT_MAX_I
solver_eps= args.solver_eps

K = args.K_hyper
rho = args.rho_hyper

temp = 2/np.log(1+np.sqrt(2))
Ts = Ising_tensor(temp,0)


#from hamiltonian import UJAFclock,UJAFclock_chiral
#Ts = UJAFclock(q,(1/temp) )

_,_,_  = NNR_TNR(Ts,FILT_EPS, FILT_MAX_I, OPT_EPS, OPT_MAX_I,RG_step,chi,temp,K,rho,solver_eps)
