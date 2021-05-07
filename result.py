import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# mix_data = np.load("mix_data.npy")
# mix_data_fix = np.load("mix_data_fix.npy")
# high_density = np.load("high_density.npy")
# high_density_fix = np.load("high_density_fix.npy")
# low_density_fix = np.load("low_density_fix.npy")
# low_density = np.load("low_density.npy")
change_weight1 = np.load("change_weight_2.npy")
change_weight2 = np.load("change_weight_3.npy")
change_weight3 = np.load("change_weight_4.npy")
weight_test1 = np.load("weight_test1.npy")
weight_test2 = np.load("weight_test2.npy")
weight_1_4 = np.load("weight_1_4.npy")
weight_2_4 = np.load("weight_2_4.npy")
weight_1_4_2 = np.load("weight_1_4_2.npy")
weight_2_4_2 = np.load("weight_2_4_2.npy")

single_gate_1 = np.load("single_att1.npy")
single_gate_2 = np.load("single_att2.npy")
single_gate_3 = np.load("single_att3.npy")

x = np.array(range(20, 50))
plt.figure()

# plt.plot(x, np.mean(mix_data, axis=1), label="mix_data")
# plt.plot(x, np.mean(mix_data_fix, axis=1), label="mid_data_fix")
# plt.plot(x, np.mean(low_density, axis=1), label="low_density")
# plt.plot(x, np.mean(low_density_fix, axis=1), label="low_density_fix")
# plt.plot(x, np.mean(change_weight1, axis=1), label="weight_2_4")
# plt.plot(x, np.mean(change_weight2, axis=1), label="weight_2_4_2")
# plt.plot(x, np.mean(change_weight3, axis=1), label="weight_2_8")
# plt.plot(x, np.mean(weight_test1, axis=1), label="fix_2_1")
# plt.plot(x, np.mean(weight_test2, axis=1), label="fix_1_0")
# plt.plot(x, np.mean(single_gate_1, axis=1), label="single_gate_1")
# plt.plot(x, np.mean(single_gate_2, axis=1), label="single_gate_2")
# plt.plot(x, np.mean(single_gate_3, axis=1), label="single_gate_3")

x = np.array(range(20, 80))
plt.plot(x, np.mean(weight_1_4_2, axis=1), label="weight_1_4_2")
# plt.plot(x, np.mean(weight_2_4_2, axis=1), label="weight_2_4_2")
plt.plot(x, np.mean(weight_1_4, axis=1), label="weight_1_4")
# plt.plot(x, np.mean(weight_2_4, axis=1), label="weight_2_4")

plt.legend()

plt.savefig("model_result.png")
