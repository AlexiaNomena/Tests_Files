"""
import sys, scipy, numpy; print(scipy.__version__, numpy.__version__, sys.version_info)
#1.7.3 1.21.5 sys.version_info(major=3, minor=10, micro=4, releaselevel='final', serial=0)
"""
import pdb
import numpy as np
import scipy.linalg as splinalg
import pickle

M = np.loadtxt("test_array.txt")

print("M is symmetric:", np.all(np.isclose(M, M.T))) ### test symmetry of M
M[np.isclose(M, np.zeros(M.shape))] = 0 # remove numerical zeros

E, U = splinalg.eigh(M) ### eigendecomposition for symmetric matrices
E = np.real(E) # remove tiny imaginary numbers
U = np.real(U)
sort = None
#sort = np.argsort(E)[::-1] # descending order
#E = E[sort]
#U = U[:, sort]

Ec, Uc = splinalg.eig(M) ### eigendecomposition for general matrices
Ec = np.real(Ec) # remove tiny imaginary numbers
Uc = np.real(Uc) 
sortc = None   
#sortc = np.argsort(Ec)[::-1] # descending order
#Ec = Ec[sortc]
#Uc = Uc[:, sortc]

if sort is not None:
	print("All eigenvalues are equal:", np.all(np.isclose(E, Ec))) # True
if sortc is not None:
	print("All eigenvectors are equal:", np.all(np.isclose(U, Uc))) # False

print("eigh eigenvalues", E)

### Recovering M from Eigendecomposition ####
SS = np.sqrt(np.diag(E)) 
tX0 = np.real(U.dot(SS))
Gram = tX0.dot(tX0.T)
        
SSc = np.sqrt(np.diag(Ec)) 
tX0c = np.real(Uc.dot(SSc))
Gramc = tX0c.dot(tX0c.T)


print("Gram matrices are the same:", np.all(np.isclose(Gram,Gramc))) # False
print("sp.linalg.eigh recovers M:", np.all(np.isclose(Gram, M))) # True
print("sp.linalg.eig recovers M:", np.all(np.isclose(M, Gramc))) # False



