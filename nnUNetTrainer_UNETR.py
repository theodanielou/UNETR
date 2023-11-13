
from time import sleep

import numpy as np
import torch
import os
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.training.data_augmentation.data_augmentation_moreDA import get_moreDA_augmentation
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.dataloading.dataset_loading import unpack_dataset
from nnunet.training.loss_functions.deep_supervision import MultipleOutputLoss2
from nnunet.training.loss_functions.dice_loss import DC_and_BCE_loss, get_tp_fp_fn_tn, SoftDiceLoss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.training.network_training.nnUNetTrainerV2_DDP import nnUNetTrainerV2_DDP
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.distributed import awesome_allgather_function
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.network_architecture.nnUNETArchitecture_UNETR import UNETR_v2, get_default_network_config
from nnunet.utilities.nd_softmax import softmax_helper



class nnUNetTrainer_UNETR(nnUNetTrainerV2):
    def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
                 unpack_data=True, deterministic=True, fp16=False):

        
        super(nnUNetTrainer_UNETR, self).__init__(plans_file, "all", output_folder, dataset_directory, batch_dice, stage,
                                                unpack_data, deterministic, fp16)

        ################# SET THESE IN INIT ################################################
        self.output_folder = output_folder
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.dataset_directory = dataset_directory

                # Modif by NR
        # self.initial_lr = 3e-4
        self.initial_lr = 1e-4
        # Modif by NR
        # self.weight_decay = 3e-5
        self.weight_decay = 1e-5


        ################# Valeurs que l'on peut modifier pour l'entrainement ###############
        self.pos_embed = 'conv' #Pas sûr qu'on puisse le changer 
        self.norm_name = 'in' #Pas sûr qu'on puisse le changer 
        self.dropout_rate = 0.0 #Pas sûr qu'on puisse le changer 
        self.qkv_bias = False #Pas sûr qu'on puisse le changer 

        ################# CE QUE J'AI AJOUTE QUI N'ETAIT PAS PRESENT DE BASE ###############
        self.plans = None
        self.plans_file = plans_file
        self.unpack_data = unpack_data
        self.init_args = (plans_file, output_folder, dataset_directory, batch_dice, unpack_data, deterministic, fp16)
        self.experiment_name = "UNETR_v2"
        self.output_folder_base = self.output_folder
        # if we are running inference only then the self.dataset_directory is set (due to checkpoint loading) but it
        # irrelevant
        if self.dataset_directory is not None and isdir(self.dataset_directory):
            self.gt_niftis_folder = join(self.dataset_directory, "gt_segmentations")
        else:
            self.gt_niftis_folder = None




        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1 #300
        self.num_batches_per_epoch = 250

        ###TEST False -> pour que ça fonctionne, mais sans do_mirroring 
        ###True pour avoir le mirroring mais ne fonctionne pas pour l'instant 
        self.do_mirroring = False 




        self.use_progress_bar = False
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        ################# Settings for saving checkpoints ##################################
        self.save_every = 50
        self.save_latest_only = True  # if false it will not store/overwrite _latest but separate files each
        # time an intermediate checkpoint is created
        self.save_intermediate_checkpoints = True  # whether or not to save checkpoint_latest
        self.save_best_checkpoint = True  # whether or not to save the best checkpoint according to self.best_val_eval_criterion_MA
        self.save_final_checkpoint = True  # whether or not to save the final checkpoint

    def initialize_network(self):
        """
        initialize self.network here
        :return:
        """
        self.props = get_default_network_config(dim= 3, nonlin="ReLU", norm_type=self.norm_name)
        self.network = UNETR_v2(in_channels=self.in_channels, out_channels=self.out_channels, img_size=self.img_size, feature_size=16, hidden_size=self.hidden_size,
                              mlp_dim=self.mlp_dim, num_heads=self.num_heads, pos_embed=self.pos_embed, norm_name=self.norm_name, dropout_rate=self.dropout_rate, qkv_bias=self.qkv_bias,
                              props = self.props, spatial_dims = 3
                             )

        if torch.cuda.is_available():
            self.network.cuda()
        self.network.inference_apply_nonlin = softmax_helper

    def initialize_optimizer_and_scheduler(self):
        """
        initialize self.optimizer and self.lr_scheduler (if applicable) here
        :return:
        """
        assert self.network is not None, "self.initialize_network must be called first"
        self.optimizer = torch.optim.AdamW(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, amsgrad=True)
        self.lr_scheduler = None

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        self.batch_size = stage_plans['batch_size']
        self.net_pool_per_axis = [5, 5, 4]
        self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        

        ###Variables que j'ai ajouté 
        self.num_heads = stage_plans['num_heads']
        self.mlp_dim = stage_plans['mlp_dim']
        self.hidden_size = stage_plans['hidden_size']
        self.feature_size = stage_plans['feature_size']
        self.batch_size = stage_plans['batch_size']
        self.in_channels = stage_plans['in_channels']
        self.out_channels = stage_plans['out_channels']
        self.img_size = stage_plans['patch_size']


        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            self.print_to_log_file("WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if 'conv_kernel_sizes' not in stage_plans.keys():
            self.print_to_log_file("WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2 
    

    def validate(self, do_mirroring: bool = True, use_sliding_window: bool = True,
                 step_size: float = 0.5, save_softmax: bool = True, use_gaussian: bool = True, overwrite: bool = True,
                 validation_folder_name: str = 'validation_raw', debug: bool = False, all_in_gpu: bool = False,
                 segmentation_export_kwargs: dict = None, run_postprocessing_on_folds: bool = True):
    

        ret = super().validate(do_mirroring=self.do_mirroring, use_sliding_window=use_sliding_window, step_size=step_size,
                               save_softmax=save_softmax, use_gaussian=use_gaussian,
                               overwrite=overwrite, validation_folder_name=validation_folder_name, debug=debug,
                               all_in_gpu=all_in_gpu, segmentation_export_kwargs=segmentation_export_kwargs,
                               run_postprocessing_on_folds=run_postprocessing_on_folds)

        return ret

    











