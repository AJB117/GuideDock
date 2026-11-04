import torch
import matplotlib.pyplot as plt

def kl_sigmoid_anneal(step, k=0.0008, x0=2000):
    """
    Sigmoid KL annealing.
    k controls slope; x0 is midpoint (≈ where KL weight = 0.5)
    """
    return float(1 / (1 + torch.exp(-k*torch.tensor(step - x0))))

# visualize a bunch of sigmoid schedules
slopes = [0.0008]
midpoints = [2000, ]

plt.figure(figsize=(12, 8))
for slope in slopes:
    for midpoint in midpoints:
        schedule = [kl_sigmoid_anneal(step, k=slope, x0=midpoint) for step in range(10000)]
        plt.plot(schedule, label=f'slope={slope}, midpoint={midpoint}')

plt.title('Sigmoid Schedules with Different Slopes and Midpoints')
plt.xlabel('Step')
plt.ylabel('Value')
# plot linear schedule for reference
plt.plot([step/10000 for step in range(10000)], label='linear', linestyle='--')
plt.legend()
plt.grid()
plt.savefig('sigmoid_schedules.png')