#!/usr/bin/env python
# coding: utf-8

# In[10]:


import matplotlib.pyplot as plt
import numpy as np

k = 8.617*10**(-5)   # in eV/K

# Plot 1
# Partition Function
def Z(g,E,T):
    N = len(g)
    
    partition = []
    
    for j in range(len(T)):
        Z = 0
        for i in range(N):
            Z += g[i]*np.exp(-E[i]/(k*T[j]))
        partition.append(Z)
    return partition

def frac(g,E,T):
    N = len(g)
    frac1 = []
    frac2 = []
    frac3=[]
    z = Z(g,E,T)
      
    for i in range(len(T)):
        f1 = (g[0]*np.exp(-E[0]/(k*T[i])))/z[i]
        f2 = (g[1]*np.exp(-E[1]/(k*T[i])))/z[i]
        f3 = (g[2]*np.exp(-E[2]/(k*T[i])))/z[i]
        frac1.append(f1)
        frac2.append(f2)
        frac3.append(f3)
    return[frac1,frac2,frac3]

T1 = np.linspace(0,5000,100)
T2 = np.linspace(5000,10**6,100)

# for 3 level system
E = [0,1,2]
g = [1,1,1]
plt.figure(figsize = (10,7))
plt.subplot(1,2,1)
plt.plot(T1,Z(g,E,T1))
plt.xlabel("Temperature")
plt.ylabel("Z")
plt.title("Partition Function vs Low Temperature")
plt.grid()


plt.subplot(1,2,2)
plt.plot(T2,Z(g,E,T2))
plt.xlabel("Temperature")
plt.ylabel("Z")
plt.title("Partition Function vs High Temperature")
plt.grid()
plt.show()

#Fractional plot
plt.plot(T1,frac(g,E,T1)[0],label = "for energy ="+str(E[0]))
plt.plot(T1,frac(g,E,T1)[1],label = "for energy ="+str(E[1]))
plt.plot(T1,frac(g,E,T1)[2],label = "for energy ="+str(E[2]))
plt.xlabel("Temperature")
plt.ylabel("N j/ N")
plt.legend()
plt.title("Plot N j/ N vs Low Temperature")
plt.grid()
plt.show()

plt.plot(T2,frac(g,E,T2)[0],label = "for energy ="+str(E[0]))
plt.plot(T2,frac(g,E,T2)[1],label = "for energy ="+str(E[1]))
plt.plot(T2,frac(g,E,T2)[2],label = "for energy ="+str(E[2]))
plt.xlabel("Temperature")
plt.ylabel("N j/ N")
plt.legend()
plt.title("Plot N j/ N vs High Temperature")
plt.grid()
plt.show()

# Internal Energy

population_1 = frac(g,E,T1)
population_2 = frac(g,E,T2)
U1 = 0
U2 = 0
for i in range(len(population_1)):
    U1 +=np.array(population_1[i])*E[i]

for i in range(len(population_2)):
    U2 +=np.array(population_2[i])*E[i]
    
plt.figure(figsize = (10,7))
plt.subplot(1,2,1)    
plt.plot(T1,U1)
plt.xlabel("Temperature")
plt.ylabel("U / N")
plt.title("U / N vs Low Temperature")
plt.grid()



plt.subplot(1,2,2)        
plt.plot(T2,U2)
plt.xlabel("Temperature")
plt.ylabel("U / N")
plt.title("U / N vs High Temperature")
plt.grid()
plt.show()

# Entropy

z1 = Z(g,E,T1)
z2 = Z(g,E,T2)

N = 6.022e23
S1 = N*k*np.log(np.array(z1)/N)+U1/T1 + N*k
S2 = N*k*np.log(np.array(z2)/N)+U2/T2 + N*k
plt.plot(T1,S1)
plt.xlabel("Temperature")
plt.ylabel("Entropy")
plt.title("Entropy vs Low Temperature")
plt.grid()
plt.show()

plt.plot(T2,S2)
plt.xlabel("Temperature")
plt.ylabel("Entropy")
plt.title("Entropy vs High Temperature")
plt.grid()
plt.show()





    

    










        
        


# In[ ]:




