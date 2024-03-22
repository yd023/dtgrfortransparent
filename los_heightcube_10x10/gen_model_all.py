import torch
import pyvredner
import numpy as np
import os

exname = "iterations_lr1_spp_dinamic_20_tv_0.001_firstspp32_changerateofspp_2"
params = torch.tensor(np.loadtxt(f'results_standard/{exname}/iter_param.log', delimiter=','))
num_iter = len(params)
obj = pyvredner.load_obj('heightmap.obj')[0]
os.mkdir(f'tmpobj_{exname}')

for iter in range(num_iter):

    param = params[iter, 1:]

    with open(f'tmpobj_{exname}/heightmap_{iter}.obj', 'w') as f:
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
