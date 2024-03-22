from pyvredner import SceneTransform
import torch
import os
import numpy as np
import vredner

vlist = [0,1,2,3,4,7,13,14,16,17,22,23]
ballVList = [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, i)]
                                        for i in range(482)
                                    
                                ]
ballVList2 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, j)]
                                        for j in range(482)
                                    ]
ballVList3 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, k)]
                                        for k in range(482)
                                        ]
                                        
ballVList.extend(ballVList2)
ballVList.extend(ballVList3)

cubeVList = [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, i)]
                                        for i in range(26)
                                    
                                ]
cubeVList2 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, j)]
                                        for j in range(26)
                                    ]
cubeVList3 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, k)]
                                        for k in range(26)
                                        ]
                                        
cubeVList.extend(cubeVList2)
cubeVList.extend(cubeVList3)

cube25VList = [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, i)]
                                        for i in range(98)
                                    
                                ]
cube25VList2 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, j)]
                                        for j in range(98)
                                    ]
cube25VList3 = [[SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, k)]
                                        for k in range(98)
                                        ]
                                        
cube25VList.extend(cube25VList2)
cube25VList.extend(cube25VList3)

opt_options = {
    # Teaser 
    'plant_teaser': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          1000,
                'sppse0':       1000,
                'sppse1':       1000,
                'sppe':         1000,
                'sppte':        0,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [ [SceneTransform("SHAPE_GLOBAL_ROTATE", torch.tensor([0, 1, 0], dtype=torch.float), 1),
        						   SceneTransform("SHAPE_GLOBAL_ROTATE", torch.tensor([0, 1, 0], dtype=torch.float), 2)] ],
        'spp_target':           10,
        'max_bounces':          5,
        'param_init':           torch.tensor([0.0]),
        'param_target':         torch.tensor([0.0]),
        'lr':                   1e-2,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  0,
        'deltaTau':             3,
        'stepTau':              10,
        'num_bins':             30,

        "use_antithetic_boundary": True,
        "use_antithetic_interior": True,

        'color_range':          [-0.01, 0.01],
        'amp':                  True,
        'amp_value':            1,
    },

    # Validation (Fig. 8)
    'cbox_diffuse_boxcar': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          50000,
                'sppse0':       50000,
                'sppse1':       50000,
                'sppe':         50000,
                'sppte':        50000,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [[SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1.0, 0], dtype=torch.float), 1)]],
        'spp_target':           1000000,
        'max_bounces':          5,
        'param_init':           torch.tensor([1.0]),
        'param_target':         torch.tensor([0.0]),
        'lr':                   1e-2,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  1,
        'tau':                  1200,
        'deltaTau':             10,
        'stepTau':              50,
        'num_bins':             1,

        'color_range':          [-0.01, 0.01],
        'amp':                  True,
        'amp_value':            2,
        'exposure_scale':       4,
    },

    'cbox_diffuse_gaussian': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          50000,
                'sppse0':       50000,
                'sppse1':       50000,
                'sppe':         50000,
                'sppte':        0,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [[SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1.0, 0], dtype=torch.float), 1)]],
        'spp_target':           1000000,
        'max_bounces':          5,
        'param_init':           torch.tensor([1.0]),
        'param_target':         torch.tensor([0.0]),
        'lr':                   1e-2,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  1250,
        'deltaTau':             10,
        'stepTau':              50,
        'num_bins':             1,

        'color_range':          [-0.01, 0.01],
        'amp':                  True,
        'amp_value':            2,
        'exposure_scale':       4,
    },

    'cbox_diffuse_truncated_gaussian': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          50000,
                'sppse0':       50000,
                'sppse1':       50000,
                'sppe':         50000,
                'sppte':        50000,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [[SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1.0, 0], dtype=torch.float), 1)]],
        'spp_target':           1000000,
        'max_bounces':          5,
        'param_init':           torch.tensor([1.0]),
        'param_target':         torch.tensor([0.0]),
        'lr':                   1e-2,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  4,
        'tau':                  1300,
        'deltaTau':             10,
        'stepTau':              50,
        'num_bins':             1,

        'color_range':          [-0.01, 0.01],
        'amp':                  True,
        'amp_value':            2,
        'exposure_scale':       8,
    },

    # Cube (Fig. 9 & Fig. 11)
    'cube_with_anti': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          256,
                'sppse0':       256,
                'sppse1':       256,
                'sppe':         256,
                'sppte':        256,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
            'ellip_nee': {
                'integrator':   vredner.BinnedTofPathTracerADps(),
                'spp':          128,
                'sppse0':       128,
                'sppse1':       128,
                'sppe':         128,
                'sppte':        0,
                "num_ellipsoidal_connections": 1,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            }
        },

        'xforms':               [[SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1.0, 0], dtype=torch.float), 1)] ],
        'spp_target':           1000,
        'max_bounces':          5,
        'param_init':           torch.tensor( [20.0] ),
        'param_target':         torch.tensor( [15.0] ),
        'lr':                   1e-1,
        'num_iters':            100,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  1130,
        'deltaTau':             0.001,
        'stepTau' :             5,
        'num_bins':             4, 
        'exposure_scale':       10000,

        "use_antithetic_boundary": True,
        "use_antithetic_interior": True,

        'color_range':          [-0.001, 0.001],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    'cube_without_anti': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          128 + 64,
                'sppse0':       128 + 64,
                'sppse1':       128 + 64,
                'sppe':         128 + 64,
                'sppte':        0,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [ [SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1.0, 0], dtype=torch.float), 1)] ],
        'spp_target':           1000,
        'max_bounces':          5,
        'param_init':           torch.tensor( [20.0] ),
        'param_target':         torch.tensor( [15.0] ),
        'lr':                   1e-1,
        'num_iters':            100,
        'gen_ref':              False,

        'pif':                  2,
        'tau':                  1130,
        'deltaTau':             0.001,
        'stepTau' :             5,
        'num_bins':             4, 
        'exposure_scale':       10000,

        "use_antithetic_boundary": False,
        "use_antithetic_interior": False,

        'color_range':          [-0.001, 0.001],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    # Corridor (Fig.10)
    'corridor_10_bounces': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          1000,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         10,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 15)
                                    ],
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 15)
                                    ]
                                ],
        'spp_target':           10000,
        'max_bounces':          10,
        'param_init':           torch.tensor([-30.0, 0]),
        'param_target':         torch.tensor([30.0, 50]),
        'lr':                   1,
        'num_iters':            200,
        'gen_ref':              True,

        'pif':                  0,
        'tau':                  200,
        'deltaTau':             25,
        'stepTau':              50,
        'num_bins':             10,

        'color_range':          [-0.002, 0.002],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    'corridor_10_bounces_fd': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          1000,
                'sppse0':       0,
                'sppse1':       0,
                'sppe':         0,
                'guiding':      {}
            },
        },

        'xforms':               [
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 15)
                                    ],
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 15)
                                    ]
                                ],
        'spp_target':           10000,
        'max_bounces':          10,
        'param_init':           torch.tensor([-30.0, 0]),
        'param_target':         torch.tensor([30.0, 50]),
        'lr':                   1,
        'num_iters':            200,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  200,
        'deltaTau':             25,
        'stepTau':              50,
        'num_bins':             10,

        'color_range':          [-0.002, 0.002],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    'corridor_3_bounces': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          1000 * 2,
                'sppse0':       100 * 2,
                'sppse1':       100 * 2,
                'sppe':         10 * 2,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1, 15)
                                    ],
                                    [
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 6),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 12),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 14),
                                        SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, 15)
                                    ]
                                ],
        'spp_target':           10000,
        'max_bounces':          3,
        'param_init':           torch.tensor([-30.0, 0]),
        'param_target':         torch.tensor([30.0, 50]),
        'lr':                   1,
        'num_iters':            200,
        'gen_ref':              False,

        'pif':                  2,
        'tau':                  200,
        'deltaTau':             25,
        'stepTau':              50,
        'num_bins':             10,

        'color_range':          [-0.002, 0.002],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    # Branches (Fig. 12)
    'tree': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          32,
                'sppse0':       32,
                'sppse1':       32,
                'sppe':         0,
                'sppte':        0,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
            'ellip_nee': {
                'integrator':   vredner.BinnedTofPathTracerADps(),
                'spp':          32,
                'sppse0':       32,
                'sppse1':       32,
                'sppe':         0,
                "num_ellipsoidal_connections": 10,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            }
        },

        'xforms':               [ 
                                    [SceneTransform("SHAPE_GLOBAL_ROTATE", torch.tensor([0, 0, 1], dtype=torch.float), 1)],
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.tensor( [0.4], dtype=torch.float ),
        'param_target':         torch.tensor( [0.0], dtype=torch.float ),
        'lr':                   1e-2,
        'num_iters':            100,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  100,
        'deltaTau':             0.5,
        'stepTau' :             1,
        'num_bins':             5,

        'color_range':          [-1, 1],
        'amp':                  False,
        'amp_value':            1,

        'paper_2x2_indices':    [0, 1, 2, 3],
    },

    # Height field (Fig. 13 & Fig. 14)
    'los_heightmap_10x10': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in range(121)
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('los_heightmap_10x10/param_init.pt'),
        'param_target':         torch.load('los_heightmap_10x10/param_target.pt'),
        'lr':                   1,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },

    'los_heightmap_10x10_steady': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100 * 12,
                'sppse0':       100 * 12,
                'sppse1':       100 * 12,
                'sppe':         100 * 12,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in range(121)
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('los_heightmap_10x10_steady/param_init.pt'),
        'param_target':         torch.load('los_heightmap_10x10_steady/param_target.pt'),
        'lr':                   1,
        'num_iters':            500,
        'gen_ref':              True,

        'pif':                  0,
        'tau':                  1000,
        'deltaTau':             999,
        'stepTau':              1000,
        'num_bins':             1,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },

    # Bunny (Fig. 15)
    'nlos_room': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          3036,
                'sppse0':       1024,
                'sppse1':       1024,
                'sppe':         10,
                'sppte':        0,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("SHAPE_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 1)], 
                                    [SceneTransform("SHAPE_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1)], 
                                ],
        'spp_target':           50000,
        'max_bounces':          10,
        'param_init':           torch.tensor( [50.0, -50.0], dtype=torch.float ),
        'param_target':         torch.tensor( [0.0, -30.0], dtype=torch.float ),
        'lr':                   1e1,
        'num_iters':            200,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  600,
        'deltaTau':             50,
        'stepTau' :             50,
        'num_bins':             20,

        'color_range':          [-0.01, 0.01],
        'amp':                  True,
        'amp_value':            5,

        'paper_2x2_indices':    [4, 6, 8, 10],
    },

    # Sofa (Fig. 16)
    'nlos_sofa': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         10,
                'sppte':        10,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in torch.load('nlos_sofa/param_id.pt')
                                ] + 
                                [
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.01, 0.0, 0.0], dtype=torch.float))],
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.0, 0.01, 0.0], dtype=torch.float))],
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.0, 0.0, 0.01], dtype=torch.float))], 
                                ],
        'spp_target':           10000,
        'max_bounces':          10,
        'param_init':           torch.load('nlos_sofa/param_init.pt'),
        'param_target':         torch.load('nlos_sofa/param_target.pt'),
        'lr':                   1,
        'num_iters':            1000,
        'gen_ref':              True,

        'pif':                  1,
        'tau':                  100,
        'deltaTau':             50,
        'stepTau':              50,
        'num_bins':             20,

        "use_antithetic_boundary": False,
        "use_antithetic_interior": False,

        'color_range':          [-0.0002, 0.0002],
        'amp':                  True,
        'amp_value':            1,
        'exposure_scale':       2,
        
        'paper_2x2_indices':    [6, 7, 8, 9],
    },

    # Tree (Fig. 17)
    'plant_light': {
        'render': {
            'ellip_nee': {
                'integrator':   vredner.BinnedTofPathTracerADps(),
                'spp':          256,
                'sppse0':       256,
                'sppse1':       256,
                'sppe':         256,
                'sppte':        256,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          1000,
                'sppse0':       500,
                'sppse1':       500,
                'sppe':         500,
                'sppte':        500,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [ 
                                    [SceneTransform("SHAPE_TRANSLATE", torch.tensor([1, 0, 0], dtype=torch.float), 0)],
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.tensor([15.0]),
        'param_target':         torch.tensor([10.0]),
        'lr':                   3e-2,
        'num_iters':            200,
        'gen_ref':              True,

        'pif':                  4,
        'tau':                  115,
        'deltaTau':             0.1,
        'stepTau':              2.5,
        'num_bins':             14,

        "use_antithetic_boundary": True,
        "use_antithetic_interior": True,

        'color_range':          [-0.1, 0.1],
        'amp':                  False,
        'amp_value':            1,
        'exposure_scale':       30,

        'paper_2x2_indices':    [3, 4, 5, 6],
    },
    
   # Our Project, optimizing z values(?)
    'OurProject': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 0, 1], dtype=torch.float), 1, i)]
                                        for i in range(66)
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('OurProject/param_init.pt'),
        'param_target':         torch.load('OurProject/param_target.pt'),
        'lr':                   1,
        'num_iters':            50,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    #OurProject2, optimizing 66 y values of v
    'OurProject2': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in range(66)
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('OurProject2/param_init.pt'),
        'param_target':         torch.load('OurProject2/param_target.pt'),
        'lr':                   1,
        'num_iters':            50,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    
    #OurProject3, optimizing 24 y values of v
    'OurProject3': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)
                                    for i in vlist]
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.tensor([0.0]),
        'param_target':         torch.tensor([40.0]),
        'lr':                   1,
        'num_iters':            100,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  100,
        'deltaTau':             5,
        'stepTau':              5,
        'num_bins':             96,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    
    #OurProject3, optimizing 24 y values of v
    'OurProject_topcamera_1': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)
                                    for i in vlist]
                                ],
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.tensor([-100.0]),
        'param_target':         torch.tensor([30.0]),
        'lr':                   1,
        'num_iters':            200,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  140,
        'deltaTau':             40,
        'stepTau':              40,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    
    # cube to ball
    'glassballToBunny': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               cubeVList,
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('glassballToBunny/param_init.pt'),
        #'param_target':         torch.load('los_heightmap_10x10/param_target.pt'),
        'lr':                   1,
        'num_iters':            100,
        'gen_ref':              False,

        'pif':                  2,
        'tau':                  140,
        'deltaTau':             40,
        'stepTau':              40,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },    
    
    # cube25 to ball
    'cube25toBall': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               cube25VList,
        'spp_target':           10000,
        'max_bounces':          5,
        'param_init':           torch.load('cube25toBall/param_init.pt'),
        'param_target':         torch.load('cube25toBall/param_target.pt'),
        'lr':                   1,
        'num_iters':            100,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  100,
        'deltaTau':             50,
        'stepTau':              50,
        'num_bins':             24,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    
    # Height field (Fig. 13 & Fig. 14)
    'los_heightcube_10x10': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in range(121)
                                ],
        'spp_target':           100000,
        'max_bounces':          5,
        'param_init':           torch.load('los_heightcube_10x10/param_init.pt'),
        'param_target':         torch.load('los_heightcube_10x10/param_target.pt'),
        'lr':                   10,
        'num_iters':            1000,
        'gen_ref':              False,

        'pif':                  2,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },

    # Height field (Fig. 13 & Fig. 14)
    'los_heightreq_2serf_10x10': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         100,
                'guiding': {
                    'direct_param':     [5000, 1, 1, 64],
                    'indirect_param':   [100, 10, 10, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in range(242)
                                ],
        'spp_target':           100000,
        'max_bounces':          5,
        'param_init':           torch.load('los_heightcube_10x10/param_init.pt'),
        'param_target':         torch.load('los_heightcube_10x10/param_target.pt'),
        'lr':                   1,
        'num_iters':            1000,
        'gen_ref':              True,

        'pif':                  2,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        'color_range':          [-0.05, 0.05],
        'amp':                  False,
        'amp_value':            1,

        'der_to_visualize':     81,
        'paper_2x2_indices':    [4, 5, 6, 7],
    },
    
    'nlos_sofa_glass': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         10,
                'sppte':        10,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)]
                                        for i in torch.load('nlos_sofa_glass/param_id.pt')
                                ] + 
                                [
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.01, 0.0, 0.0], dtype=torch.float))],
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.0, 0.01, 0.0], dtype=torch.float))],
                                    [SceneTransform("BSDF_VARY", "diffuse", 0, torch.tensor([0.0, 0.0, 0.01], dtype=torch.float))], 
                                ],
        'spp_target':           10000,
        'max_bounces':          10,
        'param_init':           torch.load('nlos_sofa_glass/param_init.pt'),
        'param_target':         torch.load('nlos_sofa_glass/param_target.pt'),
        'lr':                   1,
        'num_iters':            1000,
        'gen_ref':              True,

        'pif':                  1,
        'tau':                  160,
        'deltaTau':             10,
        'stepTau':              10,
        'num_bins':             12,

        "use_antithetic_boundary": False,
        "use_antithetic_interior": False,

        'color_range':          [-0.0002, 0.0002],
        'amp':                  True,
        'amp_value':            1,
        'exposure_scale':       2,
        
        'paper_2x2_indices':    [6, 7, 8, 9],
    },
    
    'nlos_sofa_glass_21': {
        'render': {
            'standard': {
                'integrator':   vredner.BinnedPathTracerADps(),
                'spp':          100,
                'sppse0':       100,
                'sppse1':       100,
                'sppe':         10,
                'sppte':        10,
                'guiding': {
                    'direct_param':     [50000, 1, 1, 64],
                    'indirect_param':   [1000, 30, 30, 64],
                    'num_cam_path':     10000,
                    'num_light_path':   10000,
                    'min_radius':       1e-4,
                    'indirect_type':    'knn'
                }
            },
        },

        'xforms':               [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)
                                    for i in [31,14]]
                                ] + 
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [30,15]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [29,16]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [28,17]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [27,18]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [26,19]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [5,7]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [67,62]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [61,56]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [55,50]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [49,44]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [43,38]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [37,32]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [25,20]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [66,65,64,63]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [60,59,58,57]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [54,53,52,51]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [48,47,46,45]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [42,41,40,39]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [36,35,34,33]]
                                        
                                ] +
                                [
                                    [SceneTransform("VERTEX_TRANSLATE", torch.tensor([0, 1, 0], dtype=torch.float), 1, i)for i in [24,23,22,21]]
                                        
                                ],
        'spp_target':           10000,
        'max_bounces':          10,
        'param_init':           torch.load('nlos_sofa_glass_21/param_init.pt'),
        'param_target':         torch.load('nlos_sofa_glass_21/param_target.pt'),
        'lr':                   1,
        'num_iters':            1000,
        'gen_ref':              False,

        'pif':                  1,
        'tau':                  100,
        'deltaTau':             50,
        'stepTau':              50,
        'num_bins':             20,

        "use_antithetic_boundary": False,
        "use_antithetic_interior": False,

        'color_range':          [-0.0002, 0.0002],
        'amp':                  True,
        'amp_value':            1,
        'exposure_scale':       2,
        
        'paper_2x2_indices':    [6, 7, 8, 9],
    }

}
