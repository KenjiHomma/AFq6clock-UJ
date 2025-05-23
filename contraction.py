# contra.py
import numpy as np
from ncon import ncon

def ncon(tensor_list, connect_list_in, cont_order=None, check_network=True):
    """
------------------------
by Glen Evenbly (c) for www.tensors.net, (v1.31) - last modified 30/8/2019
------------------------
Network CONtractor. Input is an array of tensors 'tensor_list' and an array \
of vectors 'connect_list_in', with each vector labelling the indices of the \
corresponding tensor. Labels should be  positive integers for contracted \
indices and negative integers for free indices. Optional input 'cont_order' \
can be used to specify order of index contractions (otherwise defaults to \
ascending order of the positive indices). Checking of the consistancy of the \
input network can be disabled for slightly faster operation.

Further information can be found at: https://arxiv.org/abs/1402.0939
"""

    # put inputs into a list if necessary
    if type(tensor_list) is not list:
        tensor_list = [tensor_list]
    if type(connect_list_in[0]) is not list:
        connect_list_in = [connect_list_in]
    connect_list = [0 for x in range(len(connect_list_in))]
    for ele in range(len(connect_list_in)):
        connect_list[ele] = np.array(connect_list_in[ele])

    # generate contraction order if necessary
    flat_connect = np.array([item for sublist in connect_list for item in sublist])
    if cont_order == None:
        cont_order = np.unique(flat_connect[flat_connect > 0])
    else:
        cont_order = np.array(cont_order)

    # check inputs if enabled
    if check_network:
        dims_list = [list(tensor.shape) for tensor in tensor_list]
        check_inputs(connect_list, flat_connect, dims_list, cont_order)

    # do all partial traces
    for ele in range(len(tensor_list)):
        num_cont = len(connect_list[ele]) - len(np.unique(connect_list[ele]))
        if num_cont > 0:
            tensor_list[ele], connect_list[ele], cont_ind = partial_trace(tensor_list[ele], connect_list[ele])
            cont_order = np.delete(cont_order, np.intersect1d(cont_order,cont_ind,return_indices=True)[1])

    # do all binary contractions
    while len(cont_order) > 0:
        # identify tensors to be contracted
        cont_ind = cont_order[0]
        locs = [ele for ele in range(len(connect_list)) if sum(connect_list[ele] == cont_ind) > 0]

        # do binary contraction
        cont_many, A_cont, B_cont = np.intersect1d(connect_list[locs[0]], connect_list[locs[1]], assume_unique=True, return_indices=True)
        tensor_list.append(np.tensordot(tensor_list[locs[0]], tensor_list[locs[1]], axes=(A_cont, B_cont)))
        connect_list.append(np.append(np.delete(connect_list[locs[0]], A_cont), np.delete(connect_list[locs[1]], B_cont)))

        # remove contracted tensors from list and update cont_order
        del tensor_list[locs[1]]
        del tensor_list[locs[0]]
        del connect_list[locs[1]]
        del connect_list[locs[0]]
        cont_order = np.delete(cont_order,np.intersect1d(cont_order,cont_many, assume_unique=True, return_indices=True)[1])

    # do all outer products
    while len(tensor_list) > 1:
        s1 = tensor_list[-2].shape
        s2 = tensor_list[-1].shape
        tensor_list[-2] = np.outer(tensor_list[-2].reshape(np.prod(s1)),
                   tensor_list[-1].reshape(np.prod(s2))).reshape(np.append(s1,s2))
        connect_list[-2] = np.append(connect_list[-2],connect_list[-1])
        del tensor_list[-1]
        del connect_list[-1]

    # do final permutation
    if len(connect_list[0]) > 0:
        return np.transpose(tensor_list[0],np.argsort(-connect_list[0]))
    else:
        return np.asscalar(tensor_list[0])


#-----------------------------------------------------------------------------
def partial_trace(A, A_label):
    """ Partial trace on tensor A over repeated labels in A_label """

    num_cont = len(A_label) - len(np.unique(A_label))
    if num_cont > 0:
        dup_list = []
        for ele in np.unique(A_label):
            if sum(A_label == ele) > 1:
                dup_list.append([np.where(A_label == ele)[0]])

        cont_ind = np.array(dup_list).reshape(2*num_cont,order='F')
        free_ind = np.delete(np.arange(len(A_label)),cont_ind)

        cont_dim = np.prod(np.array(A.shape)[cont_ind[:num_cont]])
        free_dim = np.array(A.shape)[free_ind]

        B_label = np.delete(A_label, cont_ind)
        cont_label = np.unique(A_label[cont_ind])
        B = np.zeros(np.prod(free_dim))
        A = A.transpose(np.append(free_ind, cont_ind)).reshape(np.prod(free_dim),cont_dim,cont_dim)
        for ip in range(cont_dim):
            B = B + A[:,ip,ip]

        return B.reshape(free_dim), B_label, cont_label

    else:
        return A, A_label, []

#-----------------------------------------------------------------------------
def check_inputs(connect_list, flat_connect, dims_list, cont_order):
    """ Check consistancy of NCON inputs"""

    pos_ind = flat_connect[flat_connect > 0]
    neg_ind = flat_connect[flat_connect < 0]

    # check that lengths of lists match
    if len(dims_list) != len(connect_list):
        raise ValueError(('NCON error: %i tensors given but %i index sublists given')
            %(len(dims_list), len(connect_list)))

    # check that tensors have the right number of indices
    for ele in range(len(dims_list)):
        if len(dims_list[ele]) != len(connect_list[ele]):
            raise ValueError(('NCON error: number of indices does not match number of labels on tensor %i: '
                              '%i-indices versus %i-labels')%(ele,len(dims_list[ele]),len(connect_list[ele])))

    # check that contraction order is valid
    if not np.array_equal(np.sort(cont_order),np.unique(pos_ind)):
        raise ValueError(('NCON error: invalid contraction order'))

    # check that negative indices are valid
    for ind in np.arange(-1,-len(neg_ind)-1,-1):
        if sum(neg_ind == ind) == 0:
            raise ValueError(('NCON error: no index labelled %i') %(ind))
        elif sum(neg_ind == ind) > 1:
            raise ValueError(('NCON error: more than one index labelled %i')%(ind))

    # check that positive indices are valid and contracted tensor dimensions match
    flat_dims = np.array([item for sublist in dims_list for item in sublist])
    for ind in np.unique(pos_ind):
        if sum(pos_ind == ind) == 1:
            raise ValueError(('NCON error: only one index labelled %i')%(ind))
        elif sum(pos_ind == ind) > 2:
            raise ValueError(('NCON error: more than two indices labelled %i')%(ind))

        cont_dims = flat_dims[flat_connect == ind]
        if cont_dims[0] != cont_dims[1]:
            raise ValueError(('NCON error: tensor dimension mismatch on index labelled %i: '
                              'dim-%i versus dim-%i')%(ind,cont_dims[0],cont_dims[1]))

    return True
#-----------------------------------------------------------------------------

def dot_HLv_L4(tensors_in,v):

    v = v.reshape(tensors_in[0].shape[1],tensors_in[0].shape[1],tensors_in[0].shape[1],tensors_in[0].shape[1],order ="F")
    connects = [[8,-1,1],[1,20,3],[3,19,2],[2,21,5],[5,22,4],[4,18,7],[7,17,6],[6,-2,8],[16,-3,9],[9,17,11],[11,18,10],[10,23,13],[13,24,12],[12,19,15],[15,20,14],[14,-4,16],[21,22,23,24]] 
    cont_order = [7, 3, 11, 13, 5, 15, 23, 24, 21, 22, 18, 17, 4, 10, 20, 19, 2, 12, 8, 1, 6, 16, 14, 9,] 
    w = ncon([tensors_in[0],tensors_in[1],tensors_in[2],tensors_in[3],tensors_in[4],tensors_in[5],tensors_in[6],tensors_in[7],tensors_in[0],tensors_in[1],tensors_in[2],tensors_in[3],tensors_in[4],tensors_in[5],tensors_in[6],tensors_in[7],v],connects,cont_order=cont_order,check_network=False) 
    w = np.ravel(w,order ="F")
    return w

def UJ_contract(A,B,delta1):
    connects = [[-2,-3,1],[-4,-5,2],[-6,-7,3],[-8,-1,4],[4,1,2,3]] 
    cont_order = [3, 2, 1, 4,] 
    return ncon([A,B,A,A,delta1],connects,cont_order=cont_order,check_network=False) 

def norm_ts(tensors_in):
    connects = [[1,7,2,5],[5,3,7,4],[4,8,3,6],[6,2,8,1]] 
    cont_order = [8, 6, 7, 5, 1, 2, 3, 4,] 
    return ncon([tensors_in[0],tensors_in[1],tensors_in[2],tensors_in[3]],connects,cont_order=cont_order,check_network=False) 