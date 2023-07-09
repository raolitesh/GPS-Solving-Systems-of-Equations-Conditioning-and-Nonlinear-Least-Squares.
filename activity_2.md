## Refer to activity 2
```python
#importing the necessary libraries
import time
import numpy as np
import sympy as smp
import matplotlib.pyplot as plt
from sympy import Matrix, init_printing
from sympy import Symbol 
from simple_colors import *
```
In order to solve System as a closed form solution each component x,y, and z were isolated in terms of d. By subtracting the last three equations from the first, three linear equations can be obtained as shown below:

$$2(A_4-A_1)x + 2(B_4-B_1)y + 2(C_4-C_1)z - 2c^2(t_4-t_1)d - c^2(t_1^2-t_4^2) + A_1^2 - A_4^2 + B_1^2 - B_4^2 + C_1^2 - C_4^2 = 0$$

$$2(A_3-A_1)x + 2(B_3-B_1)y + 2(C_3-C_1)z - 2c^2(t_3-t_1)d - c^2(t_1^2-t_3^2) + A_1^2 - A_3^2 + B_1^2 - B_3^2 + C_1^2 - C_3^2 = 0$$

$$2(A_2-A_1)x + 2(B_2-B_1)y + 2(C_2-C_1)z - 2c^2(t_2-t_1)d - c^2(t_1^2-t_2^2) + A_1^2 - A_2^2 + B_1^2 - B_2^2 + C_1^2 - C_2^2 = 0$$

Note that System can be broken into matrix-vector notation by taking the coefficient of $x,y,z,d$ and denoting them by matrix $A$, unknown vector $x,y,z,d$ as a column vector $u$ and all constant terms by a column vector $b$ . This can be rewritten as:
$$Au = b$$

$$\begin{bmatrix} 2(A_4-A_1) & 2(B_4-B_1) & 2(C_4-C_1) & -2c^2(t_4-t_1)\\ 2(A_3-A_1) & 2(B_3-B_1) & 2(C_3-C_1) & -2c^2(t_3-t_1)\\ 2(A_2-A_1) & 2(B_2-B_1) & 2(C_2-C_1) & -2c^2(t_2-t_1)\end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ d \end{bmatrix} = \begin{bmatrix} c^2(t_1^2-t_4^2) - A_1^2 + A_4^2 - B_1^2 + B_4^2 - C_1^2 + C_4^2 \\  c^2(t_1^2-t_3^2) - A_1^2 + A_3^2 - B_1^2 + B_3^2 - C_1^2 + C_3^2 \\ c^2(t_1^2-t_2^2) - A_1^2 + A_2^2 - B_1^2 + B_2^2 - C_1^2 + C_2^2 \end{bmatrix}$$  

We know $A_1, A_2, A_3, A_4, B_1, B_2, B_3, B_4, C_1, C_2, C_3, C_4, t_1, t_2, t_3, t_4, c$ from question 1.

Since we have 4 unknowns and 3 equations, the system is inconsistent. We express $x,y,z$ in terms of d after solving the system using augmented matrix.

Substituting these $x,y,z$ values in the following equation to get quadratic equation in one variable $d$.

$$(x - A_1)^2 + (y - B_1)^2 + (z - C_1)^2 - (c(t_1-d))^2 = 0 $$

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

# creating augmented matrix
M = Matrix([[2*(A4-A1), 2*(B4 - B1), 2*(C4 - C1), -2*(c**2)*(t4-t1), (c**2)*(t1**2 - t4**2) - A1**2 + A4**2 - B1**2 + B4**2 - C1**2 + C4**2],
           [2*(A3-A1), 2*(B3 - B1), 2*(C3 - C1), -2*(c**2)*(t3-t1), (c**2)*(t1**2 - t3**2) - A1**2 + A3**2 - B1**2 + B3**2 - C1**2 + C3**2],
           [2*(A2-A1), 2*(B2 - B1), 2*(C2 - C1), -2*(c**2)*(t2-t1), (c**2)*(t1**2 - t2**2) - A1**2 + A2**2 - B1**2 + B2**2 - C1**2 + C2**2]])

# using Gauss-Jordan rref to solve for x,y,z, and d

M_rref,q = M.rref()
print(yellow('The matrix after rref operations are as below',['bold','reverse']))
print(M_rref)
print()
# representing x,y,z in terms of d. We are using p,q,r to represent x,y,z respectively in order to avoid repeated usage of same
# variables from question 1

d = smp.symbols('d')
p = -M_rref[0,3]*d + M_rref[0,4]
q = -M_rref[1,3]*d + M_rref[1,4]
r = -M_rref[2,3]*d + M_rref[2,4]
print(blue('x in terms of d=',['bold','bright']), p)
print(blue('y in terms of d=',['bold','bright']), q)
print(blue('z in terms of d=',['bold','bright']), r)

# substituting x,y,z to get one quadratic in terms of d 

eq = (p - A1)**2 + (q - B1)**2 + (r - C1)**2 - (c*(t1-d))**2

# simplifying it 
eq1 = smp.simplify(eq)
print(blue('The quadratic equation is:',['bold','bright']), eq1)
print()
print()
end = time. time()
print(magenta('The execution time in seconds=',['bold']),end - start)
```
```python
#calculating the time involved in these steps
start = time. time()

# collecting the coefficients into array
sol = np.zeros((1,2))
coeff = np.array([[-82854564582.5318, 15077167846.7619, 49119806.6409449]])

# calculating the solutions to quadratic equation
for i in range(0,1):
    a = coeff[i,0]
    b = coeff[i,1]
    c = coeff[i,2]
    disc = (b**2)-(4*a*c)
    sol[i,0] = ((-b) + np.sqrt(disc))/(2*a)
    sol[i,1] = ((-b) - np.sqrt(disc))/(2*a)
print(blue('The two solutions are:',['bold','bright']),sol)


#finding the coordinates

print()
print()

print(green('Since the norm with first solution is approximately equal to the radius of the earth, we are considering this first solution',['bold', 'reverse']))    

# Now, using first solution to get coordinates x, y,z.
print(blue('Coordinates and norm with d =',['bold', 'bright']), sol[0,0])
coor1 = np.array([-M_rref[0,3]*sol[0,0] + M_rref[0,4], -M_rref[1,3]*sol[0,0] + M_rref[1,4], -M_rref[2,3]*sol[0,0] + M_rref[2,4]])
print(blue('The x coordinate is=',['bold']), coor1[0])
print(blue('The y coordinate is=',['bold']), coor1[1])
print(blue('The z coordinate is=',['bold']), coor1[2])

#calculating infinity norm of the vector ||x,y,z||
norm2a = np.linalg.norm(coor1.reshape(3,1), np.inf) 
print(blue('The norm is=',['bold']), norm2a)


print()
print()

# Now, using second solution to get coordinates x, y,z.
print(red('Since the norm with second solution is larger than the radius of the earth, we are not considering this second solution',['bold', 'reverse']))    

print(blue('Coordinates and norm with d =',['bold', 'bright']), sol[0,1])
coor2 = np.array([-M_rref[0,3]*sol[0,1] + M_rref[0,4], -M_rref[1,3]*sol[0,1] + M_rref[1,4], -M_rref[2,3]*sol[0,1] + M_rref[2,4]])
print(blue('The x coordinate is=',['bold']), coor2[0])
print(blue('The y coordinate is=',['bold']), coor2[1])
print(blue('The z coordinate is=',['bold']), coor2[2])

#calculating infinity norm of the vector ||x,y,z||
norm2b = np.linalg.norm(coor2.reshape(3,1), np.inf) 
print(blue('The norm is=',['bold']), norm2b)

print()
print()

end = time. time()
print(magenta('The execution time is=',['bold']),end - start)
```

