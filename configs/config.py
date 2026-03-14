import os
import torch

class Config:
    def __init__(self):
        self.project_root = r'/server03_data/LYH/TOF_3T_2_7T_new'
        
        self.data_dir = os.path.join(self.project_root, 'data')
        self.train_3t_dir = os.path.join(self.data_dir, 'train', '3T')
        self.train_7t_dir = os.path.join(self.data_dir, 'train', '7T')
        self.val_3t_dir = os.path.join(self.data_dir, 'val', '3T')
        self.val_7t_dir = os.path.join(self.data_dir, 'val', '7T')
        self.test_3t_dir = os.path.join(self.data_dir, 'test', '3T')
        self.test_7t_dir = os.path.join(self.data_dir, 'test', '7T')
        
        self.checkpoint_dir = os.path.join(self.project_root, 'checkpoints')
        self.log_dir = os.path.join(self.project_root, 'logs')
        self.result_dir = os.path.join(self.project_root, 'results')
        
        self.patch_size = 64
        self.stride = 32
        self.mip_thickness = 50
        self.mip_stride = 1
        
        self.in_channels = 1
        self.out_channels = 1
        self.base_features = 32
        self.num_layers = 5
        self.disc_num_layers = 4
        
        self.batch_size = 24
        self.batch_per_sample = 10
        self.num_epochs = 160
        self.learning_rate = 0.0002
        self.lr_decay_start = 80
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        self.alpha = 10.0
        self.beta = 1.0
        self.gamma = 1.0
        
        self.num_workers = 4
        self.pin_memory = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        self.gpu_ids = None
        
        self.save_interval = 5
        self.log_interval = 5
        self.val_interval = 5
        
        self.time_verbose = True

        self.seed = 999
    
    def set_gpu(self, gpu_ids):
        if torch.cuda.is_available():
            if gpu_ids:
                self.gpu_ids = [int(x) for x in gpu_ids.split(',')]
                self.device = torch.device(f'cuda:{self.gpu_ids[0]}')
                self.num_gpus = len(self.gpu_ids)
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.num_gpus = torch.cuda.device_count()
                self.gpu_ids = list(range(self.num_gpus))
        else:
            self.device = torch.device('cpu')
            self.num_gpus = 0
            self.gpu_ids = None
        
    def __str__(self):
        return '\n'.join([f'{k}: {v}' for k, v in self.__dict__.items()])

config = Config()
