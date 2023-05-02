import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import random


def Trials(N_c, N_t):

    outcomes_array = []

    for i in range(N_t):
        outcomes = []

        for j in range(N_c):
            
            outcomes.append(random.randint(0,1))

        outcomes_array.append(outcomes)

    n_heads = []  # counting the number of heads and tails

    for n in outcomes_array:
        n_heads.append(sum(n))

    n_heads = np.array(n_heads)
    n_tails = N_c - n_heads

    freq_count = []  # frequency count of macro-states

    for i in range(N_c + 1):
        freq_count.append(list(n_heads).count(i))

    freq_count = np.array(freq_count)

    probability = freq_count/N_t

    bd_prob = [] # data from bionomial distribtion
    Nc_arr = np.arange(0, N_c+1)

    for i in range(len(Nc_arr)):
        bd_prob.append(math.comb(N_c, i)/(2**N_c))

    p = []   # calculating p and q
    for i in range(len(n_heads)): 
        p.append(np.sum(n_heads[:i+1])/((i+1)*N_c))
        
    p = np.array(p)
    q = 1-p

    table = pd.DataFrame({'Trials': np.arange(
        1, N_t + 1), 'Outcomes': outcomes_array, 'No. of Heads': n_heads, 'No. of Tails': n_tails, 'p': p, 'q': q})

    table.set_index('Trials', inplace=True)

    return table, probability, bd_prob

out = Trials(3, 10)

print(out[0])
print('Probability:', out[1])
print('from bionomial distribution', out[2])

# plot1~

N_c = 5
N_t = 10
while N_t <= 10000:

    y = Trials(N_c, N_t)[1]
    x = np.arange(0, N_c+1)

    plt.plot(x, y, label=f'N_t = {N_t}', marker='o')
    N_t *= 10

y_bd = Trials(N_c, N_t)[2]
plt.plot(x, y_bd, label='bionomial distribution', marker='*')
plt.legend()
plt.grid()
plt.xlabel('No. of Heads')
plt.ylabel('Probability')
plt.savefig('01-1', dpi=800)
plt.title('Trials Variation Plot')

plt.show()

# plot2~

N_t = 10000
for coins in range(3, 10, 3):

    x = np.arange(0, coins+1)
    y = Trials(coins, N_t)[1]

    plt.plot(x, y, label=f'N_c = {coins}', marker='o')

plt.legend()
plt.grid()
plt.xlabel('No. of Coins')
plt.ylabel('Probability')
plt.title('Coin Variation Plot')
plt.savefig('01-2', dpi=800)

plt.show()

# plot3~

N_c = 3
N_t = 10000

data = Trials(N_c, N_t)[0]
y1 = data['p'].to_numpy()
y2 = data['q'].to_numpy()

x = np.arange(0, N_t)

plt.plot(x, y1, label='p')
plt.plot(x, y2, label='q')
plt.legend()
plt.grid()
plt.xlabel('No. of Trials')
plt.ylabel('p, q')
plt.title('Cumulative Plot')
plt.savefig('01-3', dpi=800)

plt.show()


    

        
    
    
    
        
        
  

    