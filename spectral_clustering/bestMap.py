import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment


def bestMap(L1, L2):
    """
    permute labels of L2 to match L1 as good as possible
    :param L1:
    :param L2:
    :return:
    """
    if L1.shape != L2.shape:
        raise Exception('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    n_class1 = Label1.shape[0]
    Label2 = np.unique(L2)
    n_class2 = Label2.shape[0]

    G = np.zeros((n_class1, n_class2))
    for i in range(n_class1):
        for j in range(n_class2):
            G[i, j] = np.sum((L1 == Label1[i]) & (L2 == Label2[j]))

    c = linear_assignment(-G)
    newL2 = np.zeros(L2.shape[0])
    for i in range(n_class2):
        newL2[L2 == Label2[c[i][1]]] = Label1[i]
    return newL2
