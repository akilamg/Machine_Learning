from utils import *
from run_knn import run_knn
import matplotlib.pyplot as plt

t_data = load_train()
v_data = load_test()

t_input = t_data[0]
v_input = v_data[0]

t_feature = t_data[1]

k_arr = [1,3,5,7,9]
results = {}
plot_r_k = {}

# Get the results
for k in k_arr:
    results[k] = run_knn(k, t_input, t_feature, v_input)

# Evaluate the results
for key,value in results.iteritems():
    valid = 0
    length = v_data[1].shape[0]
    for d, v in zip(value, v_data[1]):
        if d[0] == v[0]:
            valid += 1

    plot_r_k[key] = valid/float(length)

# plot the evaluation
x = plot_r_k.keys()
y = plot_r_k.values()
plt.bar(x,y, color="blue")
plt.yticks(np.arange(0,1.1,0.1))

plt.show()