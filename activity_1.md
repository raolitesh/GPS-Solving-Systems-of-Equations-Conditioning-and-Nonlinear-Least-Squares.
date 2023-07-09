## Refer to activity 1
```python
# importing the necessary libraries
import time
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from sympy import Matrix, init_printing
from sympy import Symbol 
from simple_colors import *
```
We employed the Multivariate Newton's Method to solve for the position and time correction using four satellites.

We will test the Multivariate Newton's Method for 4 satellites $(Ai,Bi,Ci,ti)$ with a known answer to see if we can obtain the same solution.

System 4.37 is algebraically manipulated into the form of System below.

$$f_1 = (x - A_1)^2 + (y - B_1)^2 + (z - C_1)^2 - (c(t_1-d))^2 = 0 $$

$$f_2 = (x - A_2)^2 + (y - B_2)^2 + (z - C_2)^2 - (c(t_2-d))^2 = 0 $$

$$f_3 = (x - A_3)^2 + (y - B_3)^2 + (z - C_3)^2 - (c(t_3-d))^2 = 0 $$

$$f_4 = (x - A_4)^2 + (y - B_4)^2 + (z - C_4)^2 - (c(t_4-d))^2 = 0 $$

To solve for $α=(x,y,z,d)$ the the Multivariate Newton Method takes the Jacobian matrix of System as $F(α)=(f_1,f_2,f_3,f_4)$ as seen below.

$$DF(\alpha) = \begin{bmatrix} \partial f_1/ \partial x & \partial f_1/ \partial y & \partial f_1/ \partial z & \partial f_1/ \partial t \\ \partial f_2/ \partial x & \partial f_2/ \partial y & \partial f_2/ \partial z & \partial f_2/ \partial t \\ \partial f_3/ \partial x & \partial f_3/ \partial y & \partial f_3/ \partial z & \partial f_3/ \partial t \\ \partial f_4/ \partial x & \partial f_4/ \partial y & \partial f_4/ \partial z & \partial f_4/ \partial t \end{bmatrix} = \begin{bmatrix} 2(x - A_1) & 2(y - B_1) & 2(z - C_1) & 2c^2(t_1 - d) \\ 2(x - A_2) & 2(y - B_2) & 2(z - C_2) & 2c^2(t_2 - d) \\ 2(x - A_3) & 2(y - B_3) & 2(z - C_3) & 2c^2(t_3 - d) \\ 2(x - A_4) & 2(y - B_4) & 2(z - C_4) & 2c^2(t_4 - d) \end{bmatrix}$$


Beginning with an initial vector $α_0$ we iterate through n∈N steps of the process of solving:

$$ \left\{
\begin{array}{ll}
      DF(\alpha_k)s = F(\alpha_k) \\
      \alpha_{k+1} = \alpha_k - s\\
\end{array} 
\right.$$

$$ for \, k = 0,1,2, \ldots, n $$ 

```python
#calculating the time involved in these steps
start = time. time()


# setting up the given or known values

A1 = 15600
A2 = 18760
A3 = 17610
A4 = 19170
B1 = 7540
B2 = 2750
B3 = 14630
B4 = 610
C1 = 20140
C2 = 18610
C3 = 13480
C4 = 18390
t1 = 0.07074
t2 = 0.07220
t3 = 0.07690
t4 = 0.07242
c = 299792.458

end = time. time()
print(magenta('The execution time in seconds=',['bold']),end - start)

```
```python
#calculating the time involved in these steps
start = time. time()

# writing multivariate Newton-Raphson to solve for x,y,z,d


def mulnew(x0, f, jac, tol, maxit = 100):
    for it in range(maxit):
        fvec = f(x0)
        j = jac(x0)
        QR = np.linalg.qr(j, mode='complete')
        a = QR[1]
        b = QR[0].transpose().dot(fvec)
        xnew = x0 - np.linalg.lstsq(a, b, rcond=None)[0]
        if np.sqrt(np.linalg.norm(xnew-x0)/len(fvec)) <= tol:
            return xnew, it
        print('F(x)=',f(x0).reshape(4,1),'||', blue('F(x) infinity norm=',['bold','reverse']), np.linalg.norm(f(x0).reshape(4,1),np.inf),
              '||', magenta('iteration no.=',['bold','reverse']),it,'||', 'x=', xnew,'||')
        x0 = xnew
        if it == maxit-1:
            print ('warning, no convergence in', maxit, 'iterations')
            return xnew, it+1
        


# writing the vector of functions
def fnd(x):
    return np.array([(x[0]-A1)**2 + (x[1]-B1)**2 + (x[2]-C1)**2 - (c*(t1-x[3]))**2,
                    (x[0]-A2)**2 + (x[1]-B2)**2 + (x[2]-C2)**2 - (c*(t2-x[3]))**2,
                    (x[0]-A3)**2 + (x[1]-B3)**2 + (x[2]-C3)**2 - (c*(t3-x[3]))**2,
                    (x[0]-A4)**2 + (x[1]-B4)**2 + (x[2]-C4)**2 - (c*(t4-x[3]))**2])

# writing the vector of differentiated functions
def jac(x):
    return np.array([[2*(x[0]-A1), 2*(x[1]-B1), 2*(x[2]-C1), 2*c**2*(t1-x[3])],
                    [2*(x[0]-A2), 2*(x[1]-B2), 2*(x[2]-C2), 2*c**2*(t2-x[3])],
                    [2*(x[0]-A3), 2*(x[1]-B3), 2*(x[2]-C3), 2*c**2*(t3-x[3])],
                    [2*(x[0]-A4), 2*(x[1]-B4), 2*(x[2]-C4), 2*c**2*(t4-x[3])]])

# Initial solution in vector form and iterating the Newton-Raphson to solve for x,y,z, and d
x0 = np.array([0,0,6370,0])
x_sol, nit = mulnew(x0, fnd, jac, 1e-6,)
print(x_sol)


print()
print()



end = time. time()
print(magenta('The execution time in seconds=',['bold']),end - start)
```

