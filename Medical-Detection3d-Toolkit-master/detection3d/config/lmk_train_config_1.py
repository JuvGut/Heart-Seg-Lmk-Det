from easydict import EasyDict as edict
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.training_image_list_file = '/home/juval.gutknecht/Projects/Data/Dataset012_aligned/train.csv'

__C.general.validation_image_list_file = '/home/juval.gutknecht/Projects/Data/Dataset012_aligned/test.csv'

# landmark label starts from 1, 0 represents the background.
__C.general.target_landmark_label = {
    'Commissure LCC-RCC': 1,
    'Commissure NCC-RCC': 2,
    'Commissure LCC-NCC': 3,
    'Nadir LCS': 4,
    'Nadir RCS': 5,
    'Nadir NCS': 6,
    'Basis of IVT LCC-RCC': 7,
    'Basis of IVT NCC-RCC': 8,
    'Basis of IVT LCC-NCC': 9,
}

__C.general.save_dir = '/home/juval.gutknecht/Projects/Data/results/lmk_model_newmask_012_222'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.crop_spacing = [2, 2, 2]        # mm

__C.dataset.crop_size = [96, 96, 96]        # voxel default: [96, 96, 96]

__C.dataset.sampling_size = [6, 6, 6]       # voxel

__C.dataset.positive_upper_bound = 3        # voxel

__C.dataset.negative_lower_bound = 6        # voxel

__C.dataset.num_pos_patches_per_image = 8  # default: 8

__C.dataset.num_neg_patches_per_image = 16  # default: 16

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
__C.dataset.sampling_method = 'GLOBAL'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

##################################
# data augmentation parameters
##################################
__C.augmentation = {}

__C.augmentation.turn_on = True

__C.augmentation.orientation_axis = [0, 0, 0]  # [x,y,z], axis = [0,0,0] will set it as random axis.

__C.augmentation.orientation_radian = [-10, 10]  # range of rotation in degree, 1 degree = 0.0175 radian, 30 deg is a lot -> 10 deg is better

__C.augmentation.translation = [10, 10, 10]  # mm

##################################
# loss function
##################################
__C.landmark_loss = {}

__C.landmark_loss.name = 'Focal'          # 'Dice', or 'Focal'

__C.landmark_loss.focal_obj_alpha = [0.75] * (len(__C.general.target_landmark_label) + 1)  # class balancing weight for focal loss

__C.landmark_loss.focal_gamma = 2         # gamma in pow(1-p,gamma) for focal loss

##################################
# net
##################################
__C.net = {}

__C.net.name = 'vdnet'

##################################
# training parameters
##################################
__C.train = {}

__C.train.epochs = 1001

__C.train.batch_size = 16 # Default: 1

__C.train.num_threads = 4

__C.train.lr = 1e-4

__C.train.betas = (0.9, 0.999)

__C.train.save_epochs = 10

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
