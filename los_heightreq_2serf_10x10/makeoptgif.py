import torch
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET 
import os
import pyvredner
mi.set_variant('scalar_rgb')

exname = "iterations_lr1_spp_dinamic_20_tv_0.001_firstspp32_changerateofspp_2"
params = torch.tensor(np.loadtxt(f'results_standard/{exname}/iter_param.log', delimiter=','))
num_iter = len(params)
# obj = pyvredner.load_obj('heightmap.obj')[0]
# os.mkdir(f'tmpobj_{exname}')

# for iter in range(num_iter):

#     param = params[iter, 1:]

#     with open(f'tmpobj_{exname}/heightmap_{iter}.obj', 'w') as f:
#         print(len(param))
#         for i in range(len(param)):
#             f.write('v {} {} {}\n'.format(
#                 obj.vertices[i][0], param[i], obj.vertices[i][2]))

#         for i in range(len(obj.vertices)-len(param)):
#             f.write('v {} {} {}\n'.format(
#                 obj.vertices[i+121][0], obj.vertices[i+121][1], obj.vertices[i+121][2]))

#         for i in range(len(obj.indices)):
#             f.write('f {} {} {}\n'.format(
#                 obj.indices[i][0] + 1, obj.indices[i][1] + 1, obj.indices[i][2] + 1))

spp = 2048
os.mkdir(f'tmpfig_{exname}_{spp}')

for iter in range(num_iter):
    
    # get scene xml
    tree = ET.parse('scene_Mitsuba_target.xml') 
    root = tree.getroot()

    # change cube object
    for sh in root.iter('shape'):
        if 'id' in sh.attrib:
            print(sh.attrib['id'])
            if(sh.attrib['id'] == 'cube'):
                for str in sh.iter('string'):
                    print(str.attrib['value'])
                    str.set('value', f'tmpobj_{exname}/heightmap_{iter}.obj') 

    # save scene xml
    tree.write('scene_Mitsuba_target.xml')

    # render
    scene = mi.load_file("scene_Mitsuba_target.xml")
    image = mi.render(scene, spp=2048)

    plt.title(f"{iter}")
    plt.axis("off")
    plt.imshow(image ** (1.0 / 2.2)); # approximate sRGB tonemapping
    plt.savefig("tmpfig_{}_{}/{:0>5}.png".format(exname,spp,iter)) 