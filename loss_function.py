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

if __name__ == "__main__":
    output = Variable(torch.rand(5, 5), requires_grad=True)
    target = Variable(torch.rand(5, 5), requires_grad=True)
    function = non_negative_loss()

    loss = function(output, target)
    loss.backward()
