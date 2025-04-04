import numpy as np
from ncon import ncon
from NNR_loop_optimization_modified import optimize_aug
from hamiltonian import UJAFclock,UJAFclock_chiral

def normalize_T_max(Ts,Ts_inp):

    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])

    PA = ncon([M1,M2],[[-1,3,1,2],[1,2,-2,3]])

    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2
    g = np.trace(M)

    for i in range(4):
        Ts[i] = Ts[i]/(g**(1/4))
        Ts_inp[i] = Ts_inp[i]/(g**(1/4))

    return Ts,Ts_inp,g

def LN_renormalization(Ts):
    res_Ts= []
    for i in range(8):
        print(np.shape(Ts[i]))

    T1 = ncon([Ts[7],Ts[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[3],Ts[0]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T0 = (T1@T2)
    res_T0 = res_T0.reshape(Ts[2].shape[2],Ts[0].shape[2],Ts[2].shape[2],Ts[0].shape[2],order="F")
   

    T1 = ncon([Ts[1],Ts[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ts[5],Ts[2]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T1 = (T1@T2)
    res_T1 = res_T1.reshape(Ts[0].shape[2],Ts[2].shape[2],Ts[0].shape[2],Ts[2].shape[2],order="F")
    

    res_T2 = res_T0.transpose(2,3,0,1)
    res_T3 = res_T1.transpose(2,3,0,1)

    res_Ts.append(res_T0)
    res_Ts.append(res_T1)
    res_Ts.append(res_T2)
    res_Ts.append(res_T3)

    return res_Ts

def LN_renormalization_inp(Tp,Ti):

    res_Ts= []

    T1 = ncon([Tp[7],Tp[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Tp[3],Tp[0]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T0 = (T1@T2)
    res_T0 = res_T0.reshape(Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],order="F")
   
    T1 = ncon([Tp[1],Tp[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Tp[5],Tp[2]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T1 = (T1@T2)
    res_T1 = res_T1.reshape(Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],order="F")
    

    res_T2 = res_T0.transpose(2,3,0,1)
    res_T3 = res_T1.transpose(2,3,0,1)

    res_Ts.append(res_T0)
    res_Ts.append(res_T1)
    res_Ts.append(res_T2)
    res_Ts.append(res_T3)

    res_Ts_inp= []

    T1 = ncon([Ti[7],Tp[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Tp[3],Ti[0]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T0 = (T1@T2)
    res_T0 = res_T0.reshape(Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],order="F")
   

    T1 = ncon([Ti[1],Tp[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Tp[5],Ti[2]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T1 = (T1@T2)
    res_T1 = res_T1.reshape(Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],order="F")

    T1 = ncon([Tp[7],Ti[4]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ti[3],Tp[0]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T2 = (T1@T2)
    res_T2 = res_T2.reshape(Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],order="F")
   
    T1 = ncon([Tp[1],Ti[6]],[[-1,1,-3],[-4,1,-2]])
    T2 = ncon([Ti[5],Tp[2]],[[-3,1,-2],[-1,1,-4]])
    T1 = (T1.reshape(T1.shape[0]*T1.shape[1],T1.shape[2]*T1.shape[3],order="F"))
    T2 = (T2.reshape(T2.shape[0]*T2.shape[1],T2.shape[2]*T2.shape[3],order="F"))
    res_T3 = (T1@T2)
    res_T3 = res_T3.reshape(Tp[0].shape[2],Tp[2].shape[2],Tp[0].shape[2],Tp[2].shape[2],order="F")

    res_T2 = res_T2.transpose(2,3,0,1)
    res_T3 = res_T3.transpose(2,3,0,1)

    res_Ts_inp.append(res_T0)
    res_Ts_inp.append(res_T1)
    res_Ts_inp.append(res_T2)
    res_Ts_inp.append(res_T3)
    return res_Ts,res_Ts_inp



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
    scaling_dims = -(0.5/np.pi)*np.log(eig[1:12]/eig[0])
    return central_charge,scaling_dims



def LN_TRG_decomp(Ts,chi):
    chi =chi
    LN_decomp= []
    for i in range(4):
        size1 = np.shape(Ts[i])

        if (Ts[0].shape[0]*Ts[0].shape[1]) >  chi :
            u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)

            u = u[:,:chi]
            s = s[:chi]
            v = v[:chi,:]
        else :
            u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)

        s1 = u@(np.diag(s**(1/2)))
        s2 = (np.diag(s**(1/2)))@v
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
        s1 = u@(np.diag(s))
        s2 =v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)]))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]]))
    return LN_decomp

def one_inp_quantity(Ts,Ts_inp):
    M1 = ncon([Ts[0],Ts[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts[1],Ts[2]],[[-1,2,-3,1],[1,-4,2,-2]])

    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M = M1@M2


    M1 = ncon([Ts_inp[0],Ts_inp[3]],[[1,-1,2,-3],[-4,2,-2,1]])
    M2 = ncon([Ts_inp[1],Ts_inp[2]],[[-1,2,-3,1],[1,-4,2,-2]])
    M1 =  M1.reshape(M1.shape[0]*M1.shape[1], M1.shape[2]*M1.shape[3])
    M2 =  M2.reshape(M2.shape[0]*M2.shape[1], M2.shape[2]*M2.shape[3])
    M12 = (M1@M2)
    g = ((np.trace(M12)))
    return g

def Normalize_Ss(Ts,X_tr,X_tr_new):
    
    size = len(X_tr)
    XX_dagger = [0 for x in range(4)]
    for i in range(4):
        M = ncon([np.conj(Ts[i]),Ts[i]],[[-1,2,3,-3],[-2,2,3,-4]])
        XX_dagger[i]= M.reshape(M.shape[0]*M.shape[1],M.shape[2]*M.shape[3])
    gamma = XX_dagger[0]
    for i in range(1,4):
        gamma = gamma@XX_dagger[i]
    gamma = np.trace(gamma)

    for i in range(size):
        X_tr[i] = X_tr[i]/(gamma**(1/(2*size)))
        X_tr_new[i]= X_tr_new[i]/(gamma**(1/(2*size)))
    return X_tr,X_tr_new,gamma

def Normalize_Back_Ss(X_tr,gamma):
    
    size = len(X_tr)
    for i in range(size):
        X_tr[i] = X_tr[i]*(gamma**(1/(2*size)))
    return X_tr

def LN_TRG_decomp_eps(Ts,chis):
    LN_decomp= []
    for i in range(4):
        u,s,v = np.linalg.svd(np.reshape(Ts[i],[Ts[i].shape[0]*Ts[i].shape[1],Ts[i].shape[2]*Ts[i].shape[3]]),full_matrices=False)
        u = u[:,:chis[i]]
        s = s[:chis[i]]
        v = v[:chis[i],:]
        s1 = u@(np.diag(s**(1/2)))
        s2 = (np.diag(s**(1/2)))@v
        LN_decomp.append(np.reshape(s1,[Ts[i].shape[0],Ts[i].shape[1],len(s)],order="F"))
        LN_decomp.append(np.reshape(s2,[len(s),Ts[i].shape[2],Ts[i].shape[3]],order="F"))
    return LN_decomp

def normalize_first(Ts,Ts_inp):
    from contraction import norm_ts
    g = norm_ts(Ts)

    for i in range(4):
        Ts[i] = Ts[i]/(g**(1/4))
        Ts_inp[i] = Ts_inp[i]/(g**(1/4))
    return Ts,Ts_inp,g

def NNR_TNR(Ts,Ts_inp,FILT_EPS, FILT_MAX_I, OPT_EPS, loop_iter,RG_I,chi,temp,K,rho,n,q,IND):

    G = 1

    Ts,Ts_inp,G = normalize_T_max(Ts,Ts_inp)

    C = 0
    N = 1
    Nplus = 2


    print("\n ===============hyperparameter  ===============\n")
    print("K:", K)
    print("rho:", rho)
    print("chi:", chi)
    print("\n =============== NNR-TNR starts ===============\n")

    mag = 0
    iteration = 0


    while iteration  < RG_I :
        print("\n//----------- Renormalization step:   "+ str(iteration)+ " -----------\n");
        if iteration == 0:
            
            eight_tensors = LN_TRG_decomp_eps(Ts,[q**3,q,q**3,q])

            eight_tensors_inp = LN_TRG_decomp_eps(Ts_inp,[q**3,q,q**3,q])
            Ts,Ts_inp = LN_renormalization_inp(eight_tensors,eight_tensors_inp)
            Ts,Ts_inp,G= normalize_first(Ts,Ts_inp)

        eight_tensors =     LN_TRG_decomp(Ts,chi)
        eight_tensors_inp = LN_TRG_decomp(Ts_inp,chi)



        if iteration >0:
            eight_tensors_p = LN_TRG_decomp(Ts,chi**2)
            eight_tensors_inp_p = LN_TRG_decomp(Ts_inp,chi**2)

            eight_tensors_p,eight_tensors, G_norm= Normalize_Ss(Ts,eight_tensors_p,eight_tensors)
            eight_tensors = optimize_aug(eight_tensors,eight_tensors_p,loop_iter,K,rho,solver_eps)                           
            eight_tensors= Normalize_Back_Ss(eight_tensors,G_norm)

            eight_tensors_inp_p, eight_tensors_inp, G_norm= Normalize_Ss(Ts_inp,eight_tensors_inp_p, eight_tensors_inp)
            eight_tensors_inp =  optimize_aug(eight_tensors_inp,eight_tensors_inp_p,loop_iter,K,rho,solver_eps)
            eight_tensors_inp= Normalize_Back_Ss(eight_tensors_inp,G_norm)


        Ts,Ts_inp= LN_renormalization_inp(eight_tensors,eight_tensors_inp)

        central_c,scaling_dims = transfer_matrix(Ts)

        G0 = G
        Ts,Ts_inp,G= normalize_T_max(Ts,Ts_inp)

        Z =G
        C = np.log(G)+Nplus*C
        N *= Nplus

        f = temp*(np.log(Z)+2*C)/(2*N)

        iteration += 1
        if iteration > 2:
            mag =  (2/(3*np.sqrt(3)))*abs(one_inp_quantity(Ts,Ts_inp))
            print("mag:",mag)


    print("\n * NNR_TNR :      \n")

    return Ts,f,mag

"""
    A sample implementation of Appendix 1 in arXiv:2403.17309.  

    This code calculates the one-point function for a single spin variable using NNR-TNR algorithm.
    In this example, it gives chilarity of the antiferromagnetic q=6 clock model on the Union Jack lattice at T=0.3.

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

import argparse

parser = argparse.ArgumentParser(
        description="arXiv:2403.17309",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("chi", type=int,nargs="?", help="Bond dimension",default=12)
parser.add_argument("temp_ratio", type=float,nargs="?",help="temp ratio",default=1)
parser.add_argument("q_state", type=int,nargs="?",help="q",default=6)
parser.add_argument("RG_step", type=int,nargs="?",help="RG_step",default=51)
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


temp = 0.3
Ts = UJAFclock(q,(1/temp) )

n =1
IND = [1,1,1]
Ts_inp = UJAFclock_chiral(q,(1/temp) )
_,f,mag = NNR_TNR(Ts,Ts_inp,FILT_EPS, FILT_MAX_I, OPT_EPS, OPT_MAX_I,RG_step,chi,temp,K,rho,n,q,IND)

