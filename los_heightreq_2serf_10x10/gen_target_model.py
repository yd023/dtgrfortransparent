import torch
import pyvredner
import numpy as np

param = torch.load("param_target.pt")

obj = pyvredner.load_obj('heightmap.obj')[0]

with open('test.obj', 'w') as f:
    print(len(param))
    for i in range(len(param)):
        f.write('v {} {} {}\n'.format(
            obj.vertices[i][0], param[i], obj.vertices[i][2]))
            
    for i in range(len(obj.vertices)-len(param)):
        f.write('v {} {} {}\n'.format(
            obj.vertices[i+121][0], obj.vertices[i+121][1], obj.vertices[i+121][2]))

    for i in range(len(obj.indices)):
        f.write('f {} {} {}\n'.format(
            obj.indices[i][0] + 1, obj.indices[i][1] + 1, obj.indices[i][2] + 1))
