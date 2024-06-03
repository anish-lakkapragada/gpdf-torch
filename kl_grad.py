# %% 
import numpy as np
import math
import torch

def compute_probs(data_vector, mean, sigma): 
    """This is the function to run the entire data vector through the entire PDF function."""
    return torch.exp(-0.5 * torch.pow(((data_vector - mean) / sigma), 2)) / (sigma * math.sqrt(2 * torch.pi))

class torchPDF(torch.nn.Module):

    def __init__(self):
        super(torchPDF, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor(1) + 1, requires_grad=True) 
        self.mean = torch.nn.Parameter(torch.Tensor(1) + 1, requires_grad=True) 

    def forward(self, x):
        return compute_probs(x, self.mean, self.sigma)
    
        # out = scipy.stats.norm(self.mean.item(), self.sigma.item()).pdf(x.detach())
        # return out 


"""GET THE PDF DATA"""
import scipy.stats
TRUE_MEAN, TRUE_STD = 3, 2
X = torch.Tensor(np.random.randn(500)) * TRUE_STD + TRUE_MEAN
P = torch.Tensor(scipy.stats.norm(TRUE_MEAN, TRUE_STD).pdf(X.detach())) # target distribution

ITERATIONS = 5000
LEARNING_RATE = 0.005
eps = 1e-10

pdf = torchPDF()
optimizer = torch.optim.SGD(pdf.parameters(), lr=LEARNING_RATE)

for _ in range(ITERATIONS): 
    Q = pdf(X) # this should be a pytorch tensor. 
    
    
    optimizer.zero_grad()

    # compute the difference between P(x) and Q(x) distributions 

    KL_divergence = torch.sum(P * torch.log(P / (Q + eps)))
    KL_divergence.backward()

    print(f"current KL_divergence: {KL_divergence.item()}")

    optimizer.step()

    
print(pdf.mean, pdf.sigma)
    


    
