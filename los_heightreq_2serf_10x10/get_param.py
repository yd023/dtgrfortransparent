import pyvredner
import torch

obj = pyvredner.load_obj('heightmap_target.obj')[0]
param = obj.vertices[:121, 1]
torch.save(param, 'param_target.pt')

print(param)

param = torch.zeros_like(param)
torch.save(param, 'param_init.pt')

print(param)

print(len(param))
