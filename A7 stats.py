#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate
from scipy.stats import linregress


def derivative(y, x):
    dy_dx = []
    
    for i in range(len(x) - 1):
        dy = y[i+1] - y[i]
        dx = x[i+1] - x[i]
        
        dy_dx.append(dy/dx)
        
    return np.array(dy_dx)
    
    
k = 1.38*10**(-23)  # J/K
h = 6.626*10**(-34) # Js
N_a = 6.023*10**(23)
m = 1.6*10**(-27)

V = np.linspace(20*10**(-3),50*10**(-3),50)   # m3
T = np.linspace(150,450,50)

matrix = np.zeros(len(V)*len(T)).reshape(len(T),len(V))
#print(matrix)
for i in range(len(V)):
    for j in range(len(T)):
        v = V[i]
        t = T[j]
        #print(v,t)
        def Z(n):
            Z = (np.pi/2)* (n**2) * np.exp(-h**2 * n**2/(8 * m * (v)**(2/3) * k * t) )
            return Z
    
        I = integrate.quad(Z,0,10**(11))[0]
        #print(I)
        matrix[i][j] = I
    
#print(matrix)

log_z = np.log(matrix)

# plot of log z vs temp
plt.subplot(1,2,1)
plt.title("plot of log(z) vs T ")
plt.plot(T,log_z[:,0],label = "for V= "+str(np.round(V[0],3)))
plt.plot(T,log_z[:,1],label = "for V= "+str(np.round(V[1],3)))
plt.plot(T,log_z[:,2],label = "for V= "+str(np.round(V[2],3)))
plt.xlabel("T") 
plt.ylabel("log(Z)")
plt.grid()
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.title("plot of log(z) vs log(T) ")
plt.plot(np.log(T),log_z[:,0],label = "for V= "+str(np.round(V[0],3)))
plt.plot(np.log(T),log_z[:,1],label = "for V= "+str(np.round(V[1],3)))
plt.plot(np.log(T),log_z[:,2],label = "for V= "+str(np.round(V[2],3)))
plt.xlabel("log(T)") 
#plt.ylabel("log(Z)")
plt.grid()
plt.legend(loc='best')
plt.show()


# plot of log z vs volume
plt.subplot(1,2,1)
plt.title("plot of log(z) vs V ")
plt.plot(V,log_z[0],label = "for T= "+str(np.round(T[0],3)))
plt.plot(V,log_z[1],label = "for T= "+str(np.round(T[1],3)))
plt.plot(V,log_z[2],label = "for T= "+str(np.round(T[2],3)))
plt.xlabel("V") 
plt.ylabel("log(Z)")
plt.grid()
plt.legend(loc='best')

plt.subplot(1,2,2)
plt.title("plot of log(z) vs log(V) ")
plt.plot(np.log(V),log_z[0],label = "for T= "+str(np.round(T[0],3)))
plt.plot(np.log(V),log_z[1],label = "for T= "+str(np.round(T[1],3)))
plt.plot(np.log(V),log_z[2],label = "for T= "+str(np.round(T[2],3)))
plt.xlabel("log(V)") 
#plt.ylabel("log(Z)")
plt.grid()
plt.legend(loc='best')
plt.show()


# pressure matrix
pressure = []
for i in range(len(T)):
    der = derivative(log_z[i],V)
    P = N_a*k*T[i] * der
    pressure.append(P)
pressure = np.array(pressure).reshape(len(T),len(V)-1)
#print(pressure)

plt.title("plot of Pressure vs Volume ")
plt.plot(V[:len(V)-1],pressure[0],label = "for T= "+str(np.round(T[0],3)))
plt.plot(V[:len(V)-1],pressure[1],label = "for T= "+str(np.round(T[1],3)))
plt.plot(V[:len(V)-1],pressure[2],label = "for T= "+str(np.round(T[2],3)))
plt.xlabel("V") 
plt.ylabel("Pressure")
plt.grid()
plt.legend(loc='best')
plt.show()

plt.title("plot of Pressure vs Temperature ")
plt.plot(T,pressure[:,0],label = "for V= "+str(np.round(V[0],3)))
plt.plot(T,pressure[:,1],label = "for V= "+str(np.round(V[1],3)))
plt.plot(T,pressure[:,2],label = "for V= "+str(np.round(V[2],3)))
plt.xlabel("Temperature") 
plt.ylabel("Pressure")
plt.grid()
plt.legend(loc='best')
plt.show()

# energy matrix
energy = []
for i in range(len(V)):
    der = derivative( log_z[:,i], T)
    
    energy.append(der)
    
energy = np.array(energy).reshape(len(T)-1,len(V))


for i in range(len(T)-1):
    energy[i] = (k*T[i]**2 * energy[i])
    
e = []
for i in range(len(T)-1):
    e.append(derivative(log_z[:,i], T))
    
e = k*(T[:len(T)-1]**2)*np.array(e)
    
plt.plot(T[:len(T)-1], e[7])
plt.show()
    
    
slope = linregress(T[:len(T)-1], e[-1])[0]

print(slope)
    
cv = derivative(e[-1], T[:len(T)-1])
print(cv)
'''
plt.title("plot of Energy vs Temperature ")
plt.plot(T[:len(V)-1],e[:,0],label = "for V= "+str(np.round(V[0],3)))
plt.plot(T[:len(V)-1],e[:,1],label = "for V= "+str(np.round(V[1],3)))
plt.plot(T[:len(V)-1],e[:,2],label = "for V= "+str(np.round(V[2],3)))
plt.xlabel("Temperature") 
plt.ylabel("energy")
plt.grid()
plt.legend(loc='best')
plt.show()


'''

# Entropy Plot



    














            
    
    
    









# In[ ]:





# In[ ]:




