#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

k =8.617*10**(-5)

def z_fermions(T):
    E=1
    Z_par=[]
    for i in range(len(T)):
        beta=1/(kb*T[i])
        z=np.exp((-1)*beta*E)+np.exp((-2)*beta*E)+np.exp((-3)*beta*E)
        Z_par.append(z)
    return Z_par


T_low_fer=np.linspace(1,5000,100)
T_high_fer=np.linspace(5000,500000,100)

Z_low_fer=z_fermions(T_low_fer)
Z_high_fer=z_fermions(T_high_fer)

plt.plot(T_low_fer,Z_low_fer)
plt.xlabel("low temp")
plt.ylabel("partition function")
plt.title('For Fermions')
plt.grid()
plt.show()

plt.plot(T_high_fer,Z_high_fer)
plt.xlabel("high temp")
plt.ylabel("partition function")
plt.title('For Fermions')
plt.grid()
plt.show()

def z_bosons(T):
    E=1
    Z_par1=[]
    for i in range(len(T)):
        beta=1/(kb*T[i])
        z=1+np.exp((-1)*beta*E)+2*np.exp((-2)*beta*E)+np.exp((-3)*beta*E)+np.exp((-4)*beta*E)
        Z_par1.append(z)
    return Z_par1

T_low_bos=np.linspace(1,5000,100)
T_high_bos=np.linspace(5000,500000,100)

Z_low_bos=z_bosons(T_low_bos)
Z_high_bos=z_bosons(T_high_bos)

plt.plot(T_low_bos,Z_low_bos)
plt.xlabel("low temp")
plt.ylabel("partition function")
plt.title('For Bosons')
plt.grid()
plt.show()

plt.plot(T_high_bos,Z_high_bos)
plt.xlabel("high temp")
plt.ylabel("partition function")
plt.title('For Bosons')
plt.grid()
plt.show()


# In[ ]:




