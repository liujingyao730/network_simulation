import time
import pickle

routes = []
start_edge = {"gneE0":0, "-gneE3":0, "-gneE4":0, "-gneE5":1, "-gneE2":1, "-gneE6":1}
end_edge = {"-gneE0":0, "gneE3":0, "gneE4":0, "gneE6":1, "gneE2":1, "gneE5":1}
start_label = {"gneE0":0, "-gneE3":1, "-gneE4":2, "-gneE5":3, "-gneE2":4, "-gneE6":5}
end_label = {"-gneE0":0, "gneE3":1, "gneE4":2, "gneE5":3, "gneE2":4, "gneE6":5}
connect = {0:{1:"gneE1"}, 1:{0:"-gneE1"}}
starigth_move = {
    "gneE0":["gneE2"], 
    "-gneE4":["gneE3"],
    "-gneE3":["gneE4"],
    "-gneE5":["gneE6"],
    "-gneE6":["gneE5"],
    "-gneE2":["-gneE0"]
}

for start_e in start_edge.keys():
    for end_e in end_edge.keys():
        if start_label[start_e] == end_label[end_e]:
            continue
        if start_edge[start_e] == end_edge[end_e]:
            routes.append([start_e, end_e])
        else:
            routes.append([start_e, connect[start_edge[start_e]][end_edge[end_e]], end_e])

edge_dir = {
    "gneE0": {"gneE1":'d', "gneE4":'r', "gneE3":'l'},
    "-gneE3": {"gneE4":'d', "-gneE0":'r', "gneE1":'l'},
    "-gneE1": {"-gneE0":'d', "gneE3":'r', "gneE4":'l'},
    "-gneE4": {"gneE3":'d', "gneE1":'r', "-gneE0":'l'},
    "gneE1": {"gneE2":'d', "gneE6":'r', "gneE5":'l'},
    "-gneE6": {"gneE5":'d', "gneE2":'r', "-gneE1":'l'},
    "-gneE2":{"-gneE1":'d', "gneE5":'r', "gneE6":'l'},
    "-gneE5":{"gneE6":'d', "-gneE1":'r', "gneE2":'l'}
}
out_edge = end_edge.keys()

def vehicle_dest(vehicle, edge):

    if edge in out_edge:
        return 'd'
    
    if edge in edge_dir:
        index = routes[vehicle].index(edge)
        next_edge = routes[vehicle][index+1]
        return edge_dir[edge][next_edge]

    raise Exception("Invalid vehicle for ", vehicle)
