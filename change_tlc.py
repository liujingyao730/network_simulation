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

'''
    |    |
 —— 3 —— 4 ——
    |    |
 —— 1 —— 2 ——
    |    |
'''

tlc_id = ["gneJ1", "gneJ2"]
offset = [0, 30]

net_file = "intersection.net.xml"
root = etree.parse(net_file)

for elem in root.iter(tag="tlLogic"):
    tlc = elem.attrib["id"]
    if tlc in tlc_id:
        i = tlc_id.index(tlc)
        elem.set("offset", str(offset[i]))

root.write(net_file, encoding="utf-8", xml_declaration=True)