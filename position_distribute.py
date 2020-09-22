import xml.etree.cElementTree as etree
import pandas as pd
import os 
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
import xml.dom.minidom as dom
import yaml

file = "fcd.xml"
lanes = ["-gneE1_0", "-gneE1_1", "-gneE1_2"]
root = etree.iterparse(file, events=["start"])

position = []

start_flag = False
end_flag = False
for event, elem in root:
    
    if elem.tag == "vehicle":
        lane_id = elem.attrib["lane"]
        if lane_id in lanes:
            position.append(float(elem.attrib["pos"]))

position = np.array(position)

sns.distplot(position, rug=True, hist=False)
plt.show()
print(np.max(position), np.min(position))