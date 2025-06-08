import matplotlib.pyplot as plt 
import numpy as np 
iters = [700,1000,1500,2000]
costs = [1412.4489918614565,1410.3569118128148, 1372.7953501803677,1370.4019330203898]
times = [1.1930241584777832,2.062098264694214,5.495332956314087, 8.296157121658325]
# Plot costs vs max_itr
plt.figure(figsize=(8, 4))
plt.plot(iters, costs, marker='o')
plt.title('Costs vs Max Iterations')
plt.xlabel('Max Iterations')
plt.ylabel('Costs')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot times vs max_itr
plt.figure(figsize=(8, 4))
plt.plot(iters, times, marker='o', color='orange')
plt.title('Computation Time vs Max Iterations')
plt.xlabel('Max Iterations')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.tight_layout()
plt.show()