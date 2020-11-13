import torch
import torch.nn as nn
from torch.autograd import Variable

class non_negative_loss(nn.Module):

    def __init__(self):

        super(non_negative_loss, self).__init__()

        self.mes_loss_function = nn.MSELoss()
        self.relu = nn.ReLU()

    def forward(self, target, output):

        mes_loss = self.mes_loss_function(target, output)
        negative_output = -1 * output
        negative_loss = torch.mean(self.relu(negative_output))

        return negative_loss + mes_loss
    
class narrow_output_loss(nn.Module):

    def __init__(self, upper_bound):

        super(narrow_output_loss, self).__init__()

        self.mes_loss_function = nn.MSELoss()
        self.relu = nn.ReLU()
        
        self.upper_bound = upper_bound
    
    def forward(self, target, output):

        nagetive_output = -1 * output
        upper_output = output - self.upper_bound

        mes_loss = self.mes_loss_function(target, output)
        nagetive_loss = torch.mean(self.relu(nagetive_output))
        upper_loss = torch.mean(self.relu(upper_output))

        return mes_loss + nagetive_loss + upper_loss


if __name__ == "__main__":
    output = Variable(torch.rand(5, 5), requires_grad=True)
    target = Variable(torch.rand(5, 5), requires_grad=True)
    function = non_negative_loss()

    loss = function(output, target)
    loss.backward()
