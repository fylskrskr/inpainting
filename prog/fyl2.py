import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 101)
y = np.sin(2*np.pi*x)

plt.figure()
plt.plot(x, y, label="$\sin(x)$", c='b')
plt.xlabel('x', fontdict=dict(fontsize=14))
plt.ylabel('y', fontdict=dict(fontsize=14))
plt.legend()
plt.show()
plt.savefig('curves.jpg')