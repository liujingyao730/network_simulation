import numpy as np
import matplotlib.pyplot as plt

def rgb_map(input_data, output_file="rgb_heatmap.png", is_error=False):

    temporal, sptial, c = input_data.shape

    input_data += 1
    max_num = 90
    factor = 256 / max_num
    if is_error:
        rgb = (256-16*np.sqrt(input_data * factor)).astype('int')
    else:
        rgb = (16*np.sqrt(input_data * factor)).astype('int')
    width = 4
    height = 2
    fig = plt.figure(figsize=(15, 5))

    for i in range(temporal):
        for j in range(sptial):
            R = hex(rgb[i, j, 0])[2:]
            G = hex(rgb[i, j, 1])[2:]
            B = hex(rgb[i, j, 2])[2:]
            if len(R) != 2:
                R = '0' + R
            if len(G) != 2:
                G = '0' + G
            if len(B) != 2:
                B = '0' + B
            colorx = "#" + R + G + B
            y = [i*width, i*width, (i+1)*width, (i+1)*width]
            x = [j*height, (j+1)*height, (j+1)*height, j*height]
            plt.fill(x, y, color=colorx)
    
    plt.xlabel("time step", fontsize=20)
    plt.ylabel("nodes", fontsize=20)
    plt.xticks([1, 51, 101, 151, 201], [0, 25, 50, 75, 100])
    plt.yticks([2, 6, 10, 14, 18, 22, 26],[28, 29, 30, 31, 32, 33, 34])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.savefig(output_file, bbox_inches="tight")

if __name__ == "__main__":

    inputs = np.random.rand(10, 5, 3) * 90
    rgb_map(inputs)
