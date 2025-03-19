import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	return np.log(sigma_q / sigma_p) + (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 0.5


mu_p = 0.0
sigma_p = 1.0
mu_q = 0.0
sigma_q = 2.0

print(kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q))