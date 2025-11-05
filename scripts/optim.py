import torch
import matplotlib.pyplot as plt
import numpy as np

# Rosenbrock function
def rosenbrock(xy):
    x, y = xy[0], xy[1]
    return 100 * (y - x**2)**2 + (1 - x)**2

# Gradient function
def grad_rosenbrock(xy):
    x, y = xy[0].item(), xy[1].item()
    dx = -400 * x * (y - x**2) - 2 * (1 - x)
    dy = 200 * (y - x**2)
    return torch.tensor([dx, dy], dtype=torch.float32)

# Base optimizer class
class Optimizer:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, xy, grad):
        raise NotImplementedError

# SGD
class SGD(Optimizer):
    def step(self, xy, grad):
        return xy - self.lr * grad

# RMSProp
class RMSProp(Optimizer):
    def __init__(self, lr=0.01, beta=0.9, eps=1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self.s = 0

    def step(self, xy, grad):
        self.s = self.beta * self.s + (1 - self.beta) * grad**2
        return xy - self.lr * grad / (torch.sqrt(self.s) + self.eps)

# Adam
class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = 0
        self.v = 0
        self.t = 0

    def step(self, xy, grad):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        return xy - self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)

# Training loop
def train(optimizer, start, max_iter=5000, tol=1e-3, noise=0.0):
    xy = torch.tensor(start, dtype=torch.float32)
    history_loss, history_dist = [], []
    optimum = torch.tensor([1.0, 1.0])

    for i in range(max_iter):
        f = rosenbrock(xy)
        grad = grad_rosenbrock(xy)

        if noise > 0:
            grad += noise * torch.randn_like(grad)

        if torch.norm(grad) < tol:
            break

        xy = optimizer.step(xy, grad)

        history_loss.append(f.item())
        history_dist.append(torch.norm(xy - optimum).item())

    return history_loss, history_dist

def run_experiment(lr=0.01, noise=0.0, start=[-1.2, 1.0]):
    optims = {
        "SGD": SGD(lr),
        "RMSProp": RMSProp(lr),
        "Adam": Adam(lr)
    }

    starts = [
        [-1.2, 1.0],
        [2.0, 2.0],   
        [0.0, 0.0],  
    ]

    results = {}
    index = 0
    for name, opt in optims.items():
        start = starts[index]
        loss, dist = train(opt, start, noise=noise)
        results[name] = (loss, dist)
        index = index + 1

    # Plot
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    for name, (loss, _) in results.items():
        plt.plot(loss, label=name)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title(f"Loss vs Iteration (lr={lr}, noise={noise})")

    plt.subplot(1,2,2)
    for name, (_, dist) in results.items():
        plt.plot(dist, label=name)
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Distance to Optimum")
    plt.legend()
    plt.title(f"Distance vs Iteration (lr={lr}, noise={noise})")

    plt.show()

if __name__ == "__main__":
    run_experiment(lr=1e-3, noise=0.1, start=[-1.2, 1.0])
