#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate



#Dulong and Petit Law
#Plot 1
T = np.linspace(0,10,100)

Cv_3R = np.ones(len(T))   # makes an array containing 1
plt.xlabel("Temperature")
plt.ylabel(r"$C_{v}/3R$")
plt.plot(T,Cv_3R)
plt.title("Dulong and Petit Law")
plt.grid()
plt.show()

# Einstein Plot
#Plot 2
## Plot(a)
x = np.linspace(0,2,100)  #Frequency
y = np.zeros(len(x))     #Number Density
y[int(len(y)/2)] = x[int(len(y)/2)]
plt.plot(x,y)
plt.xlabel(r"$V/V_{E}$")
plt.ylabel(r"$G(V)dV/3N_{A}$")
plt.grid()
plt.show()

## Plot(b)
def Einstein(x):
    x1 = []
    y = []
    for i in x:
        if i != 0 :          # At zero our y becomes undefined
            x1.append(i)
            y.append(((1/i)**2* np.exp(1/i))/(np.exp(1/i)-1)**2)
        else:
            continue
    return(x1,y)

x = np.linspace(0,10,1000)
plt.plot(Einstein(x)[0],Einstein(x)[1])
plt.xlabel("T/THETA_E")
plt.ylabel(r"$C_{v}/3R$")
plt.grid()
plt.show()

# Debye Plot
# Plot 3
def y(x):
    return x**4 * np.exp(x) / (np.exp(x) - 1)**2

y_d = [3*(i**3) * integrate.quad(y, 0, 1/i)[0] for i in x]

plt.plot(x, y_d, label = 'Debye')
plt.xlabel(r'$T/\theta$')
plt.ylabel(r'$C_v/3 R$')
plt.title("Theories of Specific Heat")
plt.grid()

plt.show()

#density of states for debye's theory

yd = x**2

plt.plot(x,yd)
plt.title("Debye's Theory: Density of States")
plt.xlabel(r'$v/v_D$')
plt.ylabel(r'$G(v)dv/3N_A$')
plt.grid()
plt.show()




# In[ ]:




