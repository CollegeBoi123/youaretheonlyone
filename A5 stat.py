#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import stats

# Density of States
#Wein's Displacement Law
# Question 1


# Part(a)
x = np.linspace(0.01,12,100)
def f_p(a):
    return (a**3)/(np.exp(a)-1)

plt.plot(x,f_p(x))


plt.xlabel("x")
plt.ylabel("f_p(x)")
plt.title("Wein's Displacement law, Energy of States")
plt.grid()
plt.show()

# Part (b)
i = list(f_p(x)).index(max(f_p(x)))
xp = x[i]
print("The peak value is xp" , xp)
c = 2.99* 10**(8)   # m/s
k = 8.61733 * 10**(-5)*1.6*10**(-19)
h = 6.626 * 10**(-34)

b = (h*c)/(k*xp)
print("The vale of Wein's constant is", b)

#Stefan Boltzmann Law
# Question 2
# part (a) 

I_p = integrate.quad(f_p,0.01,1000)[0]
print("The value of integration using python" , I_p)
cal_value = np.pi**4/15
print("The value of integration using numerical method is" , cal_value)

def C(T):
    l = h*c/(k*T)
    C = 8*np.pi*(k*T)/l**3
    return C

T = np.arange(100,10000,500)
C_T = C(T)
print(C_T)

u = cal_value*(C_T)
F = c*u/4

plt.plot(T,F,"o-")
plt.xlabel("Temperature")
plt.ylabel("F")
plt.title("Radiant Flux vs Temperature")
plt.grid()
plt.show()

#part(b)
plt.plot(np.log(T),np.log(F),"o-")
plt.xlabel("Temperature")
plt.ylabel("F")
#plt.xlim(-1,9)
#plt.ylim(-20,20)
plt.title("Radiant Flux vs Temperature")
plt.grid()
plt.show()

result = stats.linregress(np.log(T),np.log(F))
print("The Slope is : ",result.slope, "and the Intercept is : ",result.intercept)

#part (c)

sigma_cal = c*8*np.pi**5*k**4/(4*15*c**3*h**3)
sigma = np.exp(result.intercept)

print("The value of sigma from Stefan-Boltzmann Law is: " , sigma_cal,"W *m**-2*K**-4")
print("The value of sigma from the plot is: ", sigma, "W *m**-2*K**-4")
if np.round(sigma_cal,10) == np.round(sigma,10):
    print("The value of sigma obtained from Stefan Boltzmann Law and plot are equal hence the law is proved.")

    
#Area:
a = 0.001
b = 12
x = np.linspace(a,b,100)
for i in x:
    int_l = integrate.quad(f_p,a,i)[0]
    int_r = integrate.quad(f_p,i,b)[0]
    
    if abs(int_l - int_r)/int_r <= 0.1:
        x_mean = i
        print("The X value which divides the plot in equal parts is: ", x_mean)
        break
                           
b_mean = (h*c)/(k*x_mean)
print("The mean b value is" , b_mean)



    
    


# In[ ]:





# In[ ]:





# In[ ]:




