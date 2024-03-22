import vredner
import pyvredner
import torch
import os
import argparse
import numpy as np
import opt_settings as spec
import matplotlib.pyplot as plt
import dtgr_render
import random
import sys


def create_output_directory(dir_out,lr,iterspp):
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, f'iterations')):
        os.mkdir(os.path.join(dir_out, f'iterations'))

def time_to_string(time_elapsed):
    hours, rem = divmod(time_elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    minutes = int(minutes)
    if hours > 0:
        ret = "{:0>2}h {:0>2}m {:0>2.2f}s".format(hours, minutes, seconds)
    elif minutes > 0:
        ret = "{:0>2}m {:0>2.2f}s".format(minutes, seconds)
    else:
        ret = "{:0>2.2f}s".format(seconds)
    return ret

def main(args):
    dir_out = os.path.join(args.scene_name, 'results_' + args.integrator_name)

    opt_spec = spec.opt_options[args.scene_name]
    render_spec = opt_spec['render'][args.integrator_name]
    gspec = render_spec['guiding']

    lr = opt_spec['lr']
    iterspp = render_spec['spp']
    create_output_directory(dir_out,lr,iterspp)

    exposure_scale = opt_spec['exposure_scale'] if 'exposure_scale' in opt_spec else 1

    def reorg_images(imgs):
        num_bins = imgs.size(0)
        nder = imgs.size(1)
        height = imgs.size(2)
        width = imgs.size(3)
        channel = imgs.size(4)
        imgs = imgs.permute(1, 2, 3, 4, 0).contiguous().view(
            nder, height, width, channel * num_bins)
        imgs[0] *= exposure_scale
        return imgs

    if args.mode == 1:
        imgs_init = dtgr_render.render(
            args.scene_name, args.integrator_name, param_mode='init')
        dtgr_render.imwrite(imgs_init, args.scene_name,
                           args.integrator_name, 'images_init')
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_init')

        imgs_target = dtgr_render.render(
            args.scene_name, args.integrator_name, param_mode='target')
        dtgr_render.imwrite(imgs_target, args.scene_name,
                           args.integrator_name, 'images_target')
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_target')

    elif args.mode == 2:
        imgs_component = dtgr_render.render(
            args.scene_name, args.integrator_name, component=args.component)
        dtgr_render.imwrite(imgs_component, args.scene_name,
                           args.integrator_name, 'images_component')
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_component')

    else:

        file_loss = open(os.path.join(dir_out, f'iterations', 'targets_loss.log'), 'w')

        seed = random.randint(1,sys.maxsize)
        print(seed)

        imgs_ref = dtgr_render.render(
                args.scene_name, args.integrator_name, seed, param_mode='target', component='ref')
        dtgr_render.imwrite(imgs_ref, args.scene_name, args.integrator_name, 'images_ref', nder_to_output=[0])
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_ref', nder_to_output=[0])
        imgs_ref = reorg_images(imgs_ref)[0]

        for t in range(opt_spec['num_iters']):
            print('[Iter %3d]' % t, end=' ')

            spp = 100

            if t > 4:
                spp = 10000
            elif t > 2:
                spp = 1000

            seed = random.randint(1,sys.maxsize)
            print(seed)

            imgs_ref2 = dtgr_render.render(
                args.scene_name, args.integrator_name, seed, param_mode='target', spp=spp)
            dtgr_render.imwrite(imgs_ref2, args.scene_name, args.integrator_name, 'images_ref2', nder_to_output=[0])
            dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_ref2', nder_to_output=[0])
            imgs_ref2 = reorg_images(imgs_ref2)[0]

            # compute losses
            img_loss = (imgs_ref2 - imgs_ref).pow(2).mean()
            opt_loss = np.sqrt(img_loss.detach().numpy())
        
            # write image/param loss
            file_loss.write("%d, %.5e\n" %
                            (t, opt_loss))
            file_loss.flush()

        file_loss.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run optimization using DTGR.',
            epilog='Lifan Wu (lifanw@nvidia.com) Guangyan Cai (gcai3@uci.edu)')

    parser.add_argument('scene_name', metavar='scene_name',
                        type=str, help='scene name for inverse rendering')
    parser.add_argument('integrator_name', metavar='integrator_name', type=str,
                        help='supported integrator: standard, ellip_nee')
    parser.add_argument('-mode', metavar='mode', type=int, default=0, help='[0] optimization\
                                                                            [1] Tune param for target image\
                                                                            [2] Tune param for optimization')
    parser.add_argument('-component', metavar='component', type=str, default='all',
                     help='[all] [main] [direct] [indirect] [primary] [ref] are supported')
    parser.add_argument('-guiding_interval', metavar='guiding_interval', type=int, default=1,
                     help='interval for recomputing the guiding')
    parser.add_argument('-plot', metavar='plot', type=int,
                        default=0, help='plot the parameters.')
    args = parser.parse_args()
    main(args)
