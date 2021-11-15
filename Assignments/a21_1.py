
import numpy as np

#define equation coefficient
A=[[2,1,1],[2,1,0],[0,2,-3]]
B=[[1],[1],[1]]

#solve variables
x = np.linalg.solve(A,B)


#for i in range(len(x)):
#    print(f"x{i+1}:{x[i+1]}\n")

print(x)