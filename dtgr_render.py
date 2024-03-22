import vredner
from pyvredner import SceneManager
import pyvredner
import torch
import os
import argparse
import numpy as np
import opt_settings as spec
from img_utils import convertEXR2ColorMap, convertEXR2PNG

class DTGRCache:
    def __init__(self, scene_manager, integrator, options):
        self.scene_manager = scene_manager
        self.integrator = integrator
        self.options = options

dtgr_cache_table = {}

def set_indirect_guiding(scene_manager, gspec, integrator, options, quiet = False):
    if 'indirect_param' in gspec:
        if gspec['indirect_type'] == 'knn':
            index_guide_type = 1
            scene_manager.set_indirect_guiding(index_guide_type, gspec['indirect_param'],
                                               gspec['num_cam_path'], gspec['num_light_path'], gspec['min_radius'],
                                               integrator, options, quiet)
        elif gspec['indirect_type'] == 'rsearch':
            index_guide_type = 2
            scene_manager.set_indirect_guiding(index_guide_type, gspec['indirect_param'],
                                               gspec['num_cam_path'], gspec['num_light_path'], gspec['search_radius'],
                                               integrator, options, quiet)
        elif gspec['indirect_type'] == 'old':
            index_guide_type = 3
            scene_manager.set_indirect_guiding(index_guide_type, gspec['indirect_param'], 0, 0, 0.0, integrator, options, quiet)

def set_direct_guiding(scene_manager, gspec, integrator, options, quiet = False):
    if 'direct_param' in gspec:
        scene_manager.set_direct_guiding(gspec['direct_param'], integrator, options, quiet)

def set_primary_guiding(scene_manager, gspec, integrator, options, quiet = False):
    if 'primary_param' in gspec:
        scene_manager.set_primary_guiding(gspec['primary_param'], integrator, options, quiet)

def render(scene_name, integrator_name, seed=0, param_mode='init', custom_params=None, 
           component='all', quiet=False, guiding=True, use_cached=False, 
           grad_out_range=None, times=[], scene_filename='scene.xml', steady=False, config=False, spp=False):
    global dtgr_cache_table
    
    scene_path = os.path.join(scene_name, scene_filename)
    opt_spec = spec.opt_options[scene_name]
    render_spec = opt_spec['render'][integrator_name]
    gspec = render_spec['guiding']

    print(f"checking renderspec spp {spp}")
    if spp == False:
        spp = 10000
        #spp = render_spec['spp']

    if (use_cached and scene_path not in dtgr_cache_table) or not use_cached: 
        scene, _ = pyvredner.load_mitsuba(scene_path)

        options = vredner.RenderOptions(
            seed, spp, opt_spec['max_bounces'], render_spec['sppe'], render_spec['sppse0'], False)
        options.sppse1 = render_spec['sppse1']
        if 'num_ellipsoidal_connections' in render_spec:
            options.num_ellipsoidal_connections = render_spec['num_ellipsoidal_connections']
        if 'sppte' in render_spec:
            options.sppte = render_spec['sppte']

        if 'use_antithetic_boundary' in opt_spec:
            options.use_antithetic_boundary = opt_spec['use_antithetic_boundary']
            if not options.use_antithetic_boundary:
                options.sppte *= 2
        
        if 'use_antithetic_interior' in opt_spec:
            options.use_antithetic_interior = opt_spec['use_antithetic_interior']
            if not options.use_antithetic_interior:
                options.spp *= 2
        
        options.grad_threshold = 5e9
        
        integrator = vredner.PathTracer() if config else render_spec['integrator']

        xforms = opt_spec['xforms']

        if opt_spec['pif'] == 0 or steady: # steady
            scene.camera.pif = 0
            scene.camera.num_bins = 1
        else:
            scene.camera.pif = opt_spec['pif'] if 'delta' not in integrator_name else 3
            scene.camera.tau = opt_spec['tau']
            scene.camera.deltaTau = opt_spec['deltaTau']
            scene.camera.stepTau = opt_spec['stepTau']
            scene.camera.num_bins = opt_spec['num_bins']
        scene_args = pyvredner.serialize_scene(scene)
        scene_manager = SceneManager(scene_args, xforms, check_range=True)

        if use_cached:
            dtgr_cache_table[scene_path] = DTGRCache(scene_manager, integrator, options)
    else:
        cache = dtgr_cache_table[scene_path]
        scene_manager = cache.scene_manager
        integrator = cache.integrator
        options = cache.options

    assert(param_mode in ['init', 'target', 'custom'])
    if param_mode == 'init':
        params = opt_spec['param_init']
    elif param_mode == 'target':
        params = opt_spec['param_target']
    elif param_mode == 'custom':
        params = custom_params
    if grad_out_range is not None:
        grad_out_range = scene_manager.set_arguments(params)
    else:
        print(params.size())
        scene_manager.set_arguments(params)

    assert(component in ['all', 'interior', 'visibility', 'path_length', 'direct', 'indirect', 'primary', 'ref'])
    if component != 'all':
        options.spp = options.sppse0 = options.sppse1 = options.sppe = options.sppte = 0
        if component == 'interior':
            options.spp = spp
        elif component == 'visibility':
            options.sppse0 = render_spec['sppse0']
            options.sppse1 = render_spec['sppse1']
            options.sppe = render_spec['sppe']
        elif component == 'path_length':
            options.sppte = render_spec['sppte']
        elif component == 'direct':
            options.sppse0 = render_spec['sppse0']
        elif component == 'indirect':
            options.sppse1 = render_spec['sppse1']
        elif component == 'primary':
            options.sppe = render_spec['sppe']
        elif component == 'ref':
            options.spp = opt_spec['spp_target']
    
    options.quiet = quiet
    options.seed = seed

    if guiding and not config:
        if options.sppse0 > 0:
            set_direct_guiding(scene_manager, gspec, integrator, options, quiet)
        if options.sppse1 > 0:
            set_indirect_guiding(scene_manager, gspec, integrator, options, quiet)
        if options.sppe > 0:
            set_primary_guiding(scene_manager, gspec, integrator, options, quiet)
    imgs_hdr = scene_manager.render(integrator, options, times)

    return imgs_hdr

def imwrite(imgs_hdr, scene_name, integrator_name, dir_out, nder_to_output=[], steady=False):
    dir_out = os.path.join(scene_name, "results_" + integrator_name, dir_out)

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    opt_spec = spec.opt_options[scene_name]

    if not nder_to_output:
        nder_to_output = list(range(vredner.nder + 1))

    for i, img_hdr in enumerate(imgs_hdr):
        if opt_spec['pif'] == 0 or steady:
            for j in nder_to_output:
                img = img_hdr[j, :, :, :]
                exr_path = os.path.join(dir_out, "{}_{}_{:d}.exr".format(
                    scene_name, integrator_name, j))
                pyvredner.imwrite(img, exr_path)
        else:
            tau = str(opt_spec['tau'] + opt_spec['stepTau'] * i)
            for j in nder_to_output:
                img = img_hdr[j, :, :, :]
                exr_path = os.path.join(dir_out, "{}_{}_{:d}_tau_{}.exr".format(
                    scene_name, integrator_name, j, tau))
                pyvredner.imwrite(img, exr_path)

def exr2png(scene_name, integrator_name, dir_out, nder_to_output=[], steady=False):
    dir_out = os.path.join(
        scene_name, "results_" + integrator_name, dir_out)

    assert(os.path.exists(dir_out))

    opt_spec = spec.opt_options[scene_name]
    color_range = opt_spec['color_range']
    amp = opt_spec['amp']
    amp_value = opt_spec['amp_value']
    colorbar = False
    exposure_scale = opt_spec['exposure_scale'] if 'exposure_scale' in opt_spec else 1

    if not nder_to_output:
        nder_to_output = list(range(vredner.nder + 1))

    if opt_spec['pif'] == 0 or steady:
        for j in nder_to_output:
            exr_path = os.path.join(dir_out, "{}_{}_{:d}.exr".format(
                scene_name, integrator_name, j))
            png_path = os.path.join(dir_out, "{}_{}_{:d}.png".format(
                scene_name, integrator_name, j))
            if j > 0:
                convertEXR2ColorMap(exr_path, png_path, color_range[0], color_range[1], colorbar, amp, amp_value)
            else:
                convertEXR2PNG(exr_path, png_path, 1)
    else:
        for i in range(opt_spec['num_bins']):
            tau = str(opt_spec['tau'] + opt_spec['stepTau'] * i)
            for j in nder_to_output:
                exr_path = os.path.join(dir_out, "{}_{}_{:d}_tau_{}.exr".format(
                    scene_name, integrator_name, j, tau))
                png_path = os.path.join(dir_out, "{}_{}_{:d}_tau_{}.png".format(
                    scene_name, integrator_name, j, tau))
                
                if j > 0:
                    convertEXR2ColorMap(exr_path, png_path, color_range[0], color_range[1], colorbar, amp, amp_value)
                else:
                    convertEXR2PNG(exr_path, png_path, exposure_scale)

def render_iter(scene_name, integrator_name, iter):
    param_log = os.path.join(scene_name, "results_" + integrator_name, 'iterations', 'iter_param.log')
    if not os.path.exists(param_log):
        print('No param log.')
        assert(False)
    param = torch.tensor(np.loadtxt(param_log, delimiter=','))[iter, 1:]
    return render(scene_name, integrator_name, param_mode='custom', custom_params=param)

def render_fd(scene_name, integrator_name, seed=0, eps=0.5, param_mode='init', custom_params=None,
              component='', use_cached=False, grad_out_range=None, times=[], scene_filename='scene.xml', steady=False,
              save_tmp=False):
    
    opt_spec = spec.opt_options[scene_name]
    _times = []
    _component = 'ref' if component == 'ref' else 'interior'

    imgs = [render(scene_name, 
                    integrator_name, 
                    seed=seed,
                    param_mode=param_mode,
                    custom_params=custom_params,
                    component=_component,
                    quiet=True,
                    use_cached=use_cached,
                    guiding=False,
                    grad_out_range=grad_out_range,
                    times=_times,
                    steady=steady)[:, 0]]

    assert(param_mode in ['init', 'target', 'custom'])
    if param_mode == 'init':
        params = opt_spec['param_init']
    elif param_mode == 'target':
        params = opt_spec['param_target']
    elif param_mode == 'custom':
        params = custom_params

    tmp_imgs = [] 
    for i in range(vredner.nder):
        _params = params.clone()

        _params[i] += eps
        imgs1 = render(scene_name,
                       integrator_name,
                       seed=seed,
                       param_mode='custom',
                       custom_params=_params,
                       component=_component,
                       quiet=True,
                       use_cached=use_cached,
                       guiding=False,
                       times=_times,
                       steady=steady)[:, 0]

        _params[i] -= 2 * eps
        imgs2 = render(scene_name,
                       integrator_name,
                       seed=seed,
                       param_mode='custom',
                       custom_params=_params,
                       component=_component,
                       quiet=True,
                       use_cached=use_cached,
                       guiding=False,
                       times=_times,
                       steady=steady)[:, 0]

        imgs.append((imgs1 - imgs2) / (2 * eps))
        tmp_imgs.append([imgs1, imgs2])

    imgs = torch.stack(imgs, dim=1)
    times.append(sum(_times))

    if save_tmp:
        return imgs, tmp_imgs
    else:
        return imgs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='Render .',
            epilog='Lifan Wu (lifanw@nvidia.com) Guangyan Cai (gcai3@uci.edu)')
    parser.add_argument('scene_name', metavar='scene_name', type=str, help='scene name for inverse rendering')
    parser.add_argument('integrator_name', metavar='integrator_name', type=str, help='supported integrator: standard, ellip_nee')
    parser.add_argument('-dir', metavar='dir', type=str, default='preview', help='dir to output the images')
    parser.add_argument('-component', metavar='component', type=str, default='all',
                     help='[all] [main] [direct] [indirect] [primary] [ref] are supported')
    parser.add_argument('-quiet', metavar='quiet', type=int, default=0)
    args = parser.parse_args()
    
    imgs = render(args.scene_name, args.integrator_name, component=args.component, quiet=bool(args.quiet))
    imwrite(imgs, args.scene_name, args.integrator_name, args.dir)
    exr2png(args.scene_name, args.integrator_name, args.dir)
