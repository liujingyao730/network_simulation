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

from route_conf import routes, start, end, interval, probability

doc = dom.Document()

root = doc.createElement("routes")
root.setAttribute("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
root.setAttribute("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

doc.appendChild(root)

vtype = doc.createElement("vType")
vtype.setAttribute("id", "CAR")
vtype.setAttribute("accel", "2.6")
vtype.setAttribute("decel", "4.5")
vtype.setAttribute("sigma", "0")
vtype.setAttribute("length", "4")
vtype.setAttribute("minGap", "3")
vtype.setAttribute("maxSpeed", "70")
vtype.setAttribute("guiShape", "passenger")
vtype.setAttribute("speedFactor", "0.9")
vtype.setAttribute("speedDev", "0.03")
root.appendChild(vtype)

begin = start
d_end = start + interval
for time in range(int((end-start)/interval)):
    for i in range(len(routes)):
        rou = " "
        rou = rou.join(routes[i])
        flow_id = str(i) + "_" + str(time)
        flow = doc.createElement("flow")
        flow.setAttribute("id", flow_id)
        flow.setAttribute("begin", str(begin))
        flow.setAttribute("end", str(d_end))
        flow.setAttribute("probability", str(probability[i][time]))
        flow.setAttribute("type", "CAR")
        flow.setAttribute("departSpeed", "random")
        flow.setAttribute("departLane", "allowed")
        route = doc.createElement("route")
        route.setAttribute("edges", rou)
        flow.appendChild(route)
        root.appendChild(flow)

    begin += interval
    d_end += interval

file = open("intersection.rou.xml", "w")
doc.writexml(file, indent='\t', addindent='\t', newl='\n', encoding="utf-8")