import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt 
#import tensorflow as tf
import seaborn as sns

#bernoulli distribution
n = 1
N = 1000


p2  = 0.5
p1 = 0.8

s1=np.random.binomial(n,p1,N)
s2=np.random.binomial(n,p2,N)

num_ones, num_zeros = np.bincount(s1)
prob_ones = np.sum(num_ones) / len(s1)
prob_zeros = np.sum(num_zeros) / len(s1)

num_ones2, num_zeros2 = np.bincount(s2)
prob_ones2 = num_ones2 / len(s2)
prob_zeros2 = num_zeros2 / len(s2)

fig = plt.figure()
ax = fig.subplots(1,1)
ax.bar( 'p1', prob_ones, color='b')
ax.bar( '1-p1', prob_ones2, color='b' )
ax.bar( 'p2', prob_zeros, color='r' )
ax.bar( '1-p2', prob_zeros2, color='r' )

# ax.bar(X_axis - 0.2, num_ones2, 0.4, label='p1')
# ax.bar(X_axis + 0.2, num_zeros2, 0.4, label='p2')

ax.set_ylabel('Probability')
ax.set_title('Two Bernoulli Distribution')	
plt.tight_layout()

plt.show()


