
import numpy as np

#define equation coefficient
A=[[2,1,1],[1,1,0],[0,1,-3]]
B=[[2],[2],[1]]

#solve variables
x = np.linalg.solve(A,B)


#for i in range(len(x)):
#    print(f"x{i+1}:{x[i+1]}\n")

print(x)