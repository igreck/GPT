import torch

class Config:
    def __init__(self):
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

        # === GRPO hyperparameters ===
        self.group_size          = 4      # G completions / prompt
        self.update_epochs       = 4      # µ update steps per batch (Algorithm 1)
        self.clip_epsilon        = 0.3    # ε clipping for ratio
        self.kl_coef             = 0.2    # β KL penalty weight
        # note: policy_ref never changes; policy_old gets copied each batch

        # === Training ===
        self.n_epochs            = 6
        self.batch_size          = 8
        self.lr                  = 1e-5
        self.warmup_steps        = 500

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
        self.log_dir             = "runs/grpo_qwen3"
        self.save_dir            = "rl_imdb_grpo_qwen"

        # === Optional extras ===
        # if you want keyword-based or other shaping rewards
        # self.length_bonus        = 0.1
        # self.diversity_coef      = 0.2