import torch 

num_steps = 4
t_steps = torch.linspace(1.0, 0.0, num_steps+1, dtype=torch.float64)

print(1 // 5)
print(t_steps)
print(t_steps[:-2], t_steps[1:-1])