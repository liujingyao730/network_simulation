import numpy as np

'''
    |    |
 —— 0 —— 1 ——
    |    |
'''
start = 0
end = 600
interval = 100

straight = 5
turn = 15
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

routes = []
time_len = int((end - start)/interval)
number_routes = (len(start_edge)-1) * len(end_edge)

probability = np.random.random((number_routes, time_len))
div = np.zeros(number_routes)
i = 0
for start_e in start_edge.keys():
    for end_e in end_edge.keys():
        if start_label[start_e] == end_label[end_e]:
            continue
        if start_edge[start_e] == end_edge[end_e]:
            routes.append([start_e, end_e])
        else:
            routes.append([start_e, connect[start_edge[start_e]][end_edge[end_e]], end_e])

        if end_e in starigth_move[start_e]:
            div[i] = straight
        else:
            div[i] = turn
        i += 1
probability = probability / div[:,None] + 0.001
probability = np.around(probability, decimals=3)