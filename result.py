import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st_node_encoder = np.load("st_node_encoder.npy")
coder_on_dir = np.load("coder_on_dir.npy")
non_dir_model = np.load("non_dir_model.npy")
_128_hidden = np.load("128_hidden.npy")

x = np.array(range(30))
plt.figure()

plt.plot(x, np.mean(st_node_encoder, axis=1), 'g--', label="base")
plt.plot(x, np.mean(coder_on_dir, axis=1), 'b--', label="complex struct")
plt.plot(x, np.mean(non_dir_model, axis=1), 'r--', label="without direction")
plt.plot(x, np.mean(_128_hidden, axis=1), 'k--', label="128 complex struct")

plt.legend()

plt.savefig("model_result.png")
