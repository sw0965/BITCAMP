from typing import Tuple
import math

def normal_approximations_to_binomial(n: int, p: float):
    mu = p * n
    sigma = math.sqrt(p * (1 - p) * n)
    return mu, sigma

# from scratch.probability import normal_cdf

# normal_probability_below = normal_cdf
# def normal_probability_above(lo: float, mu: float = 0, sigma: float = 1):
#     return 1 - normal_cdf(lo, mu, sigma)

# 7.5 p해킹

# from typing import List

# def run_experiment():
#     return [random.random() < 0.5 for _ in range(1000)]

# def reject_fairness(experiment: List[bool]):
#     num_heads = len([flip for flip in experiment if flip])
#     return num_heads < 469 or num_heads > 531

# random.seed(0)
# experiments = [run_experiment() for _ in range(1000)]
# num_rejections = len([experiment for experiment in experiments if reject_fairness(experiment)])
# assert num_rejections == 46    