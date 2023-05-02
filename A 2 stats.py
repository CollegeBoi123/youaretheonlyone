#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

k = 8.61733 * 10**(-5)    #eV K^-1

#Maxwell Boltzmann Distribution
x = np.linspace(-4,4,100)
alpha = 0
x2= np.linspace(alpha+0.0001,4,100)

def f_MB(x):
    
    y = np.exp(-x)
    return y

def f_BE(x,alpha):
    
    y = 1/(np.exp(alpha)*np.exp(x) - 1)
    return y

def f_D(x,alpha):
    
    y = 1/(np.exp(alpha)*np.exp(x) + 1)
    return y

   
# Plot 1  #All the three distributions in one plot
plt.plot(x,f_MB(x),marker = 'o',markersize = 2.5, label = "Maxwell Boltzmann Distribution")
plt.plot(x2,f_BE(x2,alpha),marker = 'o',markersize = 2.5, label = "Bose Einstein Distribution")
plt.plot(x,f_D(x,alpha),marker = 'o',markersize = 2.5, label = "Fermi Dirac Distribution")
plt.xlabel("E/kT")
plt.ylim([0,10])
plt.ylabel("F((E/kT))")
plt.legend()
plt.grid()
plt.show()

#plot 2
#Fermi-Dirac Distribution
T = [10,100,1000,5000]
E = np.linspace(-4,4,100)
E_f = 1


for i in T:
    x = E/(k*i)
    alpha = -E_f/(k*i)
    plt.plot(E,f_D(x,alpha),marker = 'o',markersize = 2.5, label = "Temperature = "+str(i) +" K")
    
plt.xlabel("E(in eV)")
plt.ylabel(r'$F_{D}(E)$') 

plt.title("figure 2")
plt.legend()
plt.grid()
plt.show()

#plot 3
#Bose-Einstein Distribution
U = 1  #eV
T = [500,1000,5000,10000]
E = np.linspace(U+0.00001,4,100)


for i in T:
    x = E/(k*i)
    alpha = -U/(k*i)
    
    plt.plot(E,f_BE(x,alpha),marker = 'o',markersize = 2.5 ,label = 'Temp ='+str(i)+'  K')

plt.xlabel("E (in eV)")
plt.xlim([0,2])
plt.ylabel("F")
plt.ylim([0,10])
plt.legend()
plt.title("PROBABILITY PLOT OF BOSE EINSTEIN FUNCTION FOR DIFFERENT TEMPERATURE")
plt.grid()
plt.show()


#plot 4 

T = [500,1000,5000,10000]
e = np.linspace(-0.1,0.1,100)

for i in T:
    X = e/(k*i)
    plt.plot(e,f_MB(X),marker = 'o',markersize=2.5,label = 'Temp ='+str(i)+'  K')

plt.xlabel("E (in eV)")
plt.ylabel("F")
plt.legend()
plt.title("PROBABILITY PLOT OF MAXWELL BOLTZMANN FUNCTION FOR DIFFERENT TEMPERATURE")
plt.grid()
plt.show()






    



# In[ ]:




