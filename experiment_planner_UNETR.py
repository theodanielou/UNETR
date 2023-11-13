import nnunet
from  nnunet.experiment_planning.experiment_planner_baseline_3DUNet_v21 import ExperimentPlanner3D_v21
from os.path import join
import numpy as np 

class ExperimentPlannerUNETR(ExperimentPlanner3D_v21):
    def __init__(self, folder_with_cropped_data, preprocessed_output_folder):
        super(ExperimentPlannerUNETR, self).__init__(folder_with_cropped_data, preprocessed_output_folder)
        self.plans_fname = join(self.preprocessed_output_folder, "nnUNetPlans_UNETR_plans_3D.pkl")
        self.data_identifier = "nnUNetData_plans_v2.1"


        #Valeurs que l'on souhaite pouvoir modifier
        self.feature_size = 16
        self.hidden_size = 768
        self.mlp_dim = 3027
        self.num_heads = 12
        self.stages = [3,6,9,12]

        #Valeurs à mettre dans le plan
        self.batch_size = 1
        self.patch_size = np.array([128, 128, 128])


    def get_properties_for_stage(self, current_spacing, original_spacing, original_shape, num_cases,
                                 num_modalities, num_classes):
        """
        Nous n'utilisons pas dans un premier temps la fonction de nnUNet permettant d'optimiser la taille des batch etc 
        Elle est basée sur le pooling effectué ce qui ne correspond pas à ce que nous utilisons 
        """ 
        
        plan = {
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'hidden_size': self.hidden_size,
            'feature_size': self.feature_size,
            'batch_size': self.batch_size,
            'in_channels': num_modalities,
            'out_channels': num_classes,
            'network_num_pool_per_axis': [5, 5, 4],
            'patch_size': self.patch_size,
            'current_spacing': current_spacing,
            'original_spacing': original_spacing,
            'do_dummy_2D_data_aug': False,
            'pool_op_kernel_sizes': [[2, 2, 2], [2, 2, 2], [2, 2, 2]],
            'conv_kernel_sizes': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]]
        }
        return plan
