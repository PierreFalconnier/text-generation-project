import torch

def fct_nucleus_sampling(p: torch.Tensor, theta: float) -> torch.Tensor:
    """
    Applies nucleus sampling to the input probabilities.

    Args:
        p (torch.Tensor): Input probability distribution.
        theta (float): Threshold value for nucleus sampling.

    Returns:
        torch.Tensor: Adjusted probabilities after nucleus sampling.
    """
    p_sorted, sorted_indices = torch.sort(p, descending=True)
    p_cumsum = torch.cumsum(p_sorted, dim=0)
    k = torch.nonzero(p_cumsum >= theta)[0][0].item()
    p_prime = p_cumsum[k]
    p = torch.where(p >= p[sorted_indices][k], p / p_prime, torch.zeros_like(p))
    return p

if __name__ == "__main__":
    p = torch.tensor([0.05, 0.5, 0.1, 0.15, 0.2])
    theta = 0.7 #should return [0, 0.5, 0, 0, 0.2]/0.7
    # theta = 0.71 #should return [0, 0.5, 0, 0.15, 0.2]/0.85
    # theta = 1.0 #should leave unchanged p
    p = fct_nucleus_sampling(p, theta)
    print(p)
