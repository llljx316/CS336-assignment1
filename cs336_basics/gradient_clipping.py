import torch

class GradientClip:
    def __init__(self, parameters,max_l2_norm,epslion=1e-6):
        self.parameters = parameters
        self.max_l2_norm = max_l2_norm
        self.epslion = epslion

    def __call__(self):
        all_grad = torch.cat([p.grad.view(-1) for p in self.parameters if p.grad is not None])
        l2_norm = torch.linalg.norm(all_grad, ord=2) 
        if l2_norm >= self.max_l2_norm:
            component = self.max_l2_norm/(l2_norm+self.epslion)
            for p in self.parameters:
                if p.grad is not None:
                    p.grad *= component

        

# def gradient_clipping(g, M, eps = 1e-6):
#     for i, data in enumerate(g):
#         grad = data.grad
#         if grad is None:
#             continue
#         l2_norm = torch.linalg.norm(grad, ord=2)
#         mask = l2_norm >=M
#         # if l2_norm < M:
#         #     continue
#         grad[mask] = M/(l2_norm+eps)*grad[mask]
#         data.grad = grad
#         # g[i] = data

#     return g