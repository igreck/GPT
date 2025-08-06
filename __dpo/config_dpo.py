import torch
from datetime import datetime

class Config:
    def __init__(self):
                # Jurnal
        self.timestamp = datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
        # === Models ===
        self.policy_model_name   = "Qwen/Qwen3-1.7B-Base"
        self.reward_model_name   = "lvwerra/distilbert-imdb"
        self.lora_from_sft_dir   = "./models/sft_imdb_qlora_qwen"
        self.sft_dir             = self.lora_from_sft_dir
        self.resume_dir          = None

        # Device
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                       else "cpu")
        self.reward_device       = "cpu"

        # === DPO hyperparameters ===
        self.beta = 0.1  # DPO temperature controlling strength of KL penalty citeturn3file9

        # === Training ===
        self.n_epochs            = 6
        self.batch_size          = 64
        self.lr                  = 1e-6
        self.warmup_steps        = 150
        self.optimizer           = "RMSprop"  # use RMSprop optimizer citeturn3file9
        self.accum_steps         = 2
        self.microbatch_size     = 4


        # === Generation ===
        self.max_new_tokens      = 128
        self.top_p               = 0.9
        self.temperature         = 0.5
        self.repetition_penalty  = 1.1

        # === Sequence lengths ===
        self.policy_max_length   = 512
        self.reward_max_length   = 512

        # === Data ===
        self.prompt_ratio_range  = (0.45, 0.7)
        self.split               = "train[:10%]"
        self.shuffle             = True
        self.num_workers         = 1
        self.seed                = 42

        # === Logging / saving ===
        self.log_every           = 10
        self.save_every          = 5
        self.log_dir             = f"runs/dpo_{self.timestamp}"
        self.save_dir            = f"models/dpo_{self.timestamp}"  # corectat (qlora)

        # === Optional extras ===
        # if you want keyword-based or other shaping rewards
        # self.length_bonus        = 0.1
        # self.diversity_coef      = 0.2
