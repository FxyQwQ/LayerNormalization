import torch

class BatchNormalization:
    def __init__(self, num_features, eps=1e-5, momentum=0.5):
        self.eps = eps
        self.momentum = momentum
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)
        self.running_mean = torch.zeros(num_features)
        self.runnning_var = torch.zeros(num_features)

    def forward(self, x, training = True):
        if training : 
            mean = torch.mean(x, [0,2], keepdim=True)
            var = torch.var(x, [0,2], keepdim=True)
            x_normalized = (x-mean) / torch.sqrt(var + self.eps)
        
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.squeeze()
            self.runnning_var = (1 - self.momentum) * self.runnning_var + self.momentum * var.squeeze()
        else:
            x_normalized = (x-self.running_mean.view(1, -1, 1)) / torch.sqrt(self.running_var.view(1, -1, 1) + self.eps)

        return self.gamma.view(1, -1, 1) * x_normalized + self.beta.view(1, -1, 1)
        

        