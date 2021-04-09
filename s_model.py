import numpy as np

class link_u_d(object):

    def __init__(self, args):

        super(link_u_d, self).__init()

        self.cycle_time_d = args["c_d"]
        self.capacity_d = args["C_d"]
        self.green_time = args["g_u_d"]
        self.split_rate = args["beta_u_d"]
        self.saturated_flow_rate = args["mu_u_d"]
