import copy
import os

import torch


class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, G, D, g_opt, d_opt, g_sche=None, d_sche=None, config=None, total_rank=1, writer=None, rank=0):
        self.config = config
        self.rank = rank
        if rank == 0:
            self.logger = config.get_logger('trainer')
        else:
            self.logger = None

        # setup GPU device if available, move models into configured device
        # self.device, device_ids = self._prepare_device(total_rank)
        self.device = rank
        self.G = G
        self.D = D
        if hasattr(G, 'module'):
            self.G_ema = copy.deepcopy(G.module).eval()
        else:
            self.G_ema = copy.deepcopy(G).eval()
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_sche = g_sche
        self.d_sche = d_sche

        self.total_step = config['trainer']['total_step']
        self.sample_period = config['trainer']['sample_period']
        self.eval_period = config['trainer']['eval_period']
        self.save_period = config['trainer']['save_period']

        self.sample_path = os.path.join(config.log_dir, 'samples')
        os.makedirs(self.sample_path, exist_ok=True)
        self.eval_path = os.path.join(config.log_dir, 'validation')
        os.makedirs(self.eval_path, exist_ok=True)

        self.global_step = 0
        self.best_metric = dict()
        self.metric = dict()

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance
        self.writer = writer

    def train(self):
        raise NotImplementedError

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move models into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, postfix='last'):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        raw_G = self.G.module if hasattr(self.G, "module") else self.G
        raw_D = self.D.module if hasattr(self.D, "module") else self.D
        raw_Gema = self.G_ema.module if hasattr(self.G_ema, "module") else self.G_ema
        state = {
            'global_step': self.global_step,
            'G_model': raw_G.state_dict(),
            'D_model': raw_D.state_dict(),
            'G_ema': raw_Gema.state_dict(),
            'G_opt': self.g_opt.state_dict(),
            'D_opt': self.d_opt.state_dict(),
            'best_metric': self.best_metric,
            'metric': self.metric,
            'config': self.config
        }
        save_path = str(self.checkpoint_dir / f'ckpt_{postfix}.pth')
        torch.save(state, save_path)
        self.logger.info(f"Saving current model to: ckpt_{postfix}.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        if self.rank == 0:
            self.logger.info("Loading checkpoint: {} ...".format(resume_path))
            print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location='cpu')
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']

        state_dict = {}
        for k, v in checkpoint['G_model'].items():
            state_dict[k.replace('module.', '')] = v
        if hasattr(self.G, 'module'):
            self.G.module.load_state_dict(state_dict)
        else:
            self.G.load_state_dict(state_dict)
        state_dict = {}
        for k, v in checkpoint['D_model'].items():
            state_dict[k.replace('module.', '')] = v
        if hasattr(self.D, 'module'):
            self.D.module.load_state_dict(state_dict)
        else:
            self.D.load_state_dict(state_dict)

        if hasattr(self.G_ema, 'module'):
            self.G_ema.module.load_state_dict(checkpoint['G_ema'])
        else:
            self.G_ema.load_state_dict(checkpoint['G_ema'])

        # load opt
        if self.rank == 0:
            print("Loading optimizer: {} ...".format(resume_path))
        self.g_opt.load_state_dict(checkpoint['G_opt'])
        self.d_opt.load_state_dict(checkpoint['D_opt'])

        # load sche
        for _ in range(self.global_step):
            self.g_sche.step()
            self.d_sche.step()

        if self.rank == 0:
            print("Checkpoint loaded. Resume training from global step {}".format(self.global_step))
