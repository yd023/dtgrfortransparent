import vredner
import pyvredner
import torch
import os
import argparse
import numpy as np
import opt_settings as spec
import matplotlib.pyplot as plt
import dtgr_render


def create_output_directory(dir_out):
    if not os.path.exists(dir_out):
        os.mkdir(dir_out)
    if not os.path.exists(os.path.join(dir_out, 'iterations')):
        os.mkdir(os.path.join(dir_out, 'iterations'))

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
    create_output_directory(dir_out)

    opt_spec = spec.opt_options[args.scene_name]
    render_spec = opt_spec['render'][args.integrator_name]
    gspec = render_spec['guiding']

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
        imgs_init = dtgr_render.render_fd(
            args.scene_name, args.integrator_name, param_mode='init')
        dtgr_render.imwrite(imgs_init, args.scene_name,
                           args.integrator_name, 'images_init')
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_init')

        imgs_target = dtgr_render.render_fd(
            args.scene_name, args.integrator_name, param_mode='target')
        dtgr_render.imwrite(imgs_target, args.scene_name,
                           args.integrator_name, 'images_target')
        dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_target')

    else:
        print('[INFO] optimization for inverse rendering starts...')
        if args.plot:
            if args.plot == 1:
                fig = plt.figure(figsize=(3*(vredner.nder+1), 3))
                gs1 = fig.add_gridspec(nrows=1, ncols=vredner.nder+1)
            elif args.plot == 2:
                fig = plt.figure(figsize=(3*2, 3))
                gs1 = fig.add_gridspec(nrows=1, ncols=2)
            if not os.path.exists(os.path.join(dir_out, 'plot')):
                os.mkdir(os.path.join(dir_out, 'plot'))
            loss_record = [[]]
            if args.plot == 1:
                for i in range(vredner.nder):
                    loss_record.append([])
            elif args.plot == 2:
                loss_record.append([])

        if opt_spec['gen_ref']:
            imgs_ref = dtgr_render.render_fd(
                args.scene_name, args.integrator_name, param_mode='target', component='ref')
            dtgr_render.imwrite(imgs_ref, args.scene_name, args.integrator_name, 'images_ref', nder_to_output=[0])
            dtgr_render.exr2png(args.scene_name, args.integrator_name, 'images_ref', nder_to_output=[0])
            imgs_ref = reorg_images(imgs_ref)[0]
        else:
            imgs_ref = []
            if opt_spec['pif'] == 0:
                dir_ref = os.path.join(dir_out, 'images_ref')
                exr_path = os.path.join(dir_ref, "{}_{}_{:d}.exr".format(
                    args.scene_name, args.integrator_name, 0))
                imgs_ref.append(pyvredner.imread(exr_path))
            else:
                for i in range(opt_spec['num_bins']):
                    tau = str(opt_spec['tau'] + opt_spec['stepTau'] * i)
                    dir_ref = os.path.join(dir_out, 'images_ref')
                    exr_path = os.path.join(dir_ref, "{}_{}_{:d}_tau_{}.exr".format(
                        args.scene_name, args.integrator_name, 0, tau))
                    imgs_ref.append(pyvredner.imread(exr_path))
            imgs_ref = torch.stack(imgs_ref, dim=0)
            imgs_ref = torch.unsqueeze(imgs_ref, 1)
            imgs_ref = reorg_images(imgs_ref)[0]

        lossFunc = pyvredner.ADLossFunc.apply
        #param = torch.tensor(opt_spec['param_init'], dtype=torch.float, requires_grad=True)
        param = opt_spec['param_init'].clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([param], lr=opt_spec['lr'])
        grad_out_range = torch.tensor([0]*vredner.nder, dtype=torch.float)

        file_loss = open(os.path.join(dir_out, 'iterations', 'iter_loss.log'), 'w')
        file_param = open(os.path.join(dir_out, 'iterations', 'iter_param.log'), 'w')
        num_pyramid_lvl = 9
        weight_pyramid = 4
        times = []

        for t in range(opt_spec['num_iters']):
            print('[Iter %3d]' % t, end=' ')
            optimizer.zero_grad()
            seed = t + 1
            guiding = t % args.guiding_interval == 0
            imgs = reorg_images(dtgr_render.render_fd(
                args.scene_name, 
                args.integrator_name,
                seed=seed,
                param_mode='custom',
                custom_params=param,
                use_cached=True,
                grad_out_range=grad_out_range,
                times=times))

            imgs_iter = lossFunc(
                imgs, 
                param,
                grad_out_range, 
                torch.tensor([10000.0]*vredner.nder, dtype=torch.float),
                num_pyramid_lvl, 
                weight_pyramid, 
                -1, 
                0)

            # compute losses
            img_loss = (imgs_iter - imgs_ref).pow(2).mean()
            opt_loss = np.sqrt(img_loss.detach().numpy())
            param_loss = (param - opt_spec['param_target']).pow(2).sum().sqrt()

            print('param: ', param.detach().numpy())
            print('render time: %s; opt. loss: %.3e; param. loss: %.3e' %
                  (time_to_string(times[-1]), opt_loss, param_loss))

            # write image/param loss
            file_loss.write("%d, %.5e, %.5e, %.2e\n" %
                            (t, opt_loss, param_loss, times[-1]))
            file_loss.flush()

            # write param values
            file_param.write("%d" % t)
            for i in range(vredner.nder):
                file_param.write(", %.5e" % param[i])
            file_param.write("\n")
            file_param.flush()

            # plot the results
            if args.plot:
                # image loss
                loss_record[0].append(opt_loss)
                ax = fig.add_subplot(gs1[0])
                ax.plot(loss_record[0], 'b')
                ax.set_title('Img. RMSE')
                ax.set_xlim([0, opt_spec['num_iters']])
                ax.set_yscale('log')

                if args.plot == 1:
                    # param record
                    for i in range(vredner.nder):
                        loss_record[i+1].append(param[i].detach()-opt_spec['param_target'][i])
                        ax = fig.add_subplot(gs1[i+1])
                        ax.plot(loss_record[i+1], 'b')
                        ax.set_title('Img. RSE')
                        ax.set_xlim([0, opt_spec['num_iters']])
                        rng = max(abs(loss_record[i+1][0])*1.5, 10*opt_spec['lr'])
                        ax.set_ylim([-rng, rng])
                        ax.set_title('Param. %d' % (i+1))
                elif args.plot == 2:
                    loss_record[1].append(param_loss.detach())
                    ax = fig.add_subplot(gs1[1])
                    ax.plot(loss_record[1], 'b')
                    ax.set_xlim([0, opt_spec['num_iters']])
                    rng = max(abs(loss_record[1][0])*1.5, 10*opt_spec['lr'])
                    ax.set_ylim([-rng, rng])
                    ax.set_title('Param. L2 Loss')

                plt.savefig(os.path.join(dir_out, 'plot', 'frame_{:03d}.png'.format(t)), bbox_inches='tight')
                plt.clf()

            img_loss.backward()
            optimizer.step()

        file_loss.close()
        file_param.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Run optimization using finite difference.',
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
