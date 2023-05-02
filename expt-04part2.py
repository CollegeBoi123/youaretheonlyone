import numpy as np
import matplotlib.pyplot as plt


nu = np.logspace(10, 30, 50)
nu_vis = np.linspace(430*(10**12), 750*(10**12), 20)

def G(x):
    return np.pi*(x**2)

#2(a)-------------------------------------------------------------

plt.plot(nu, G(nu), marker='.', label='complete range')
plt.plot(nu_vis, G(nu_vis), marker='o', label='visible range')
plt.xscale('log')
plt.xlabel('frequency')
plt.ylabel('Density of states')
plt.title("Density of states: Rayleigh Jean's and planck's Law")
plt.legend()
plt.grid()
plt.show()

 
#2(b)-------------------------------------------------------------

x = np.linspace(0, 12, 50)

f_RJ = x**2

plt.plot(x, f_RJ, marker='.')
plt.grid()
plt.xlabel(r'x (dimensionless variable=h$\nu$/KT)')
plt.ylabel(r'$f_{RJ}(x)$')
plt.title("Rayleigh Jean's Law")
plt.show()

h=6.62e-34      #Js
k=1.38e-23      #J/K
c=3e8           #m/s2

def Uv(T):
    e_ = k*T
    l_ = h*c/e_
    constant = 8*np.pi*e_/(l_**3)
    
    return constant

T = [4000, 6000, 10000]

for t in T:
    
    x_vis = h*nu_vis/(k*t)
    
    plt.plot(x, Uv(t)*f_RJ, label = f'T={t}K complete range')
    plt.plot(x_vis, Uv(t)*(x_vis**2), marker='o', label=f'T={t}K visible range')
    
plt.legend()
plt.xlabel(r'x (dimensionless variable = $h\nu/KT$)')
plt.ylabel('Energy density U(x)')
plt.title("Rayleigh Jean's Law")
plt.grid()
plt.show()
    
#2(c)-----------------------------------------------------------
e = 1.6e-19

for t in T:
    avg_e = h*nu_vis/(np.exp(h*nu_vis/(k*t)) - 1)

    plt.plot(nu_vis, avg_e/e) #avg energy in eV

#plt.xscale('log')
plt.grid()
plt.xlabel('frequency (visible range Hz)')
plt.ylabel('Average energy (in eV)')
plt.title("Planck's Radiation Law")
plt.show()
    

x = np.linspace(0.1, 12, 100)

def f_P(x):
    return (x**3)/(np.exp(x) - 1)
    
for t in T:
    
    x_vis = h*nu_vis/(k*t)
    plt.plot(x, Uv(t)*f_P(x), label = f'T={t}K complete range')
    plt.plot(x_vis, Uv(t)*f_P(x_vis), marker='o', label=f'T={t}K visible range')
    
plt.grid()
plt.legend()
plt.xlabel(r'x (dimensionless variable = $h\nu/KT$)')
plt.ylabel('Energy of states')
plt.title("Planck's radiation Law")
plt.show()
    
     

    
    
    
    
    
    
    
    
    
    