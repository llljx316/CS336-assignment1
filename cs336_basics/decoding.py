import torch
from cs336_basics import softmax

class decoder:
    def __init__(self, t, p):
        self.t = t
        self.p = p

    def __call__(self, x):
        # softmax with temperature
        probs = softmax(x / self.t, dim=-1)

        # Sort probabilities in descending order
        probs_sorted, indices_sorted = torch.sort(probs, descending=True, dim=-1)

        # Calculate cumulative probabilities
        cum_probs = torch.cumsum(probs_sorted, dim=-1)

        # Create a mask for nucleus truncation
        # We include the first element that crosses the threshold p
        # So we shift the cumulative probabilities to the right and compare with p
        cum_probs_shifted = torch.zeros_like(cum_probs)
        cum_probs_shifted[..., 1:] = cum_probs[..., :-1]
        
        # Probabilities to remove are those where the shifted cumulative probability is already >= p
        remove_indices = cum_probs_shifted >= self.p
        
        # Set the probabilities of tokens outside the nucleus to 0
        probs_sorted[remove_indices] = 0.0

        # Renormalize the remaining probabilities
        probs_renormalized = probs_sorted / torch.sum(probs_sorted, dim=-1, keepdim=True)

        # Scatter the renormalized probabilities back to their original positions
        final_probs = torch.zeros_like(probs)
        final_probs.scatter_(dim=-1, index=indices_sorted, src=probs_renormalized)
        
        return final_probs

        # # softmax with t
        # x = softmax(x/self.t, -1)

        # # p sampling
        # x_sorted, index_sorted = torch.sort(x, descending=True, dim=-1)
        # sum_x = 0
        # for ind, ele in enumerate(x_sorted):
        #     sum_x += ele
        #     if sum_x >= self.p:
        #         break
        
        # x_sorted = x_sorted[:ind+1]
        # index_sorted = index_sorted[:ind+1]
        # sum_x_sorted = x_sorted.sum()
        # x_sorted /= sum_x_sorted
        # new_x = torch.zeros_like(x).scatter_(-1, index_sorted, x_sorted)
        # return new_x