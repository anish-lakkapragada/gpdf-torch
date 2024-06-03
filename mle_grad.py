# %% 
import numpy as np
import math
import torch
from torch.autograd import Variable

class torchPDF(torch.nn.Module):

    def __init__(self):
        super(torchPDF, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor(1) + 1, requires_grad=True) 
        self.mean = torch.nn.Parameter(torch.Tensor(1) + 1, requires_grad=True) 

    def forward(self, x):
        out = scipy.stats.norm(self.mean.item(), self.sigma.item()).pdf(x.detach())
        return out


"""GET THE PDF DATA"""
import scipy.stats
TRUE_MEAN, TRUE_STD = 5, 2
X = torch.Tensor(np.random.randn(100)) * TRUE_STD + TRUE_MEAN
y = torch.Tensor(scipy.stats.norm(TRUE_MEAN, TRUE_STD).pdf(X.detach()))

ITERATIONS = 10000
LEARNING_RATE = 0.001
pdf = torchPDF()
optimizer = torch.optim.SGD(pdf.parameters(), lr=LEARNING_RATE)

def compute_probs(data_vector, mean, sigma): 
    """This is the function to run the entire data vector through the entire PDF function."""
    return torch.exp(-0.5 * torch.pow(((data_vector - mean) / sigma), 2)) / (sigma * math.sqrt(2 * torch.pi))

for _ in range(ITERATIONS): 
    data_out = pdf(X) # this should be a pytorch tensor. 
    
    optimizer.zero_grad()

    # compute the cost of this operation. 
    log_likelihood = - torch.sum(torch.log(compute_probs(data_vector=X, mean=pdf.mean, sigma=pdf.sigma))) # note the negative. 
    log_likelihood.backward()

    optimizer.step()

    print(f"current log-likelihood: {log_likelihood.item()}")

print(pdf.mean, pdf.sigma)
    


    
