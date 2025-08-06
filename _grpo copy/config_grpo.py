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

        # === GRPO hyperparameters ===
        self.clip_epsilon        = 0.3
        self.clip_epsilon_final  = 0.05  # scheduling liniar pe parcursul epocilor PPO
        self.group_size          = 4      # G completions / prompt
        self.update_epochs       = 4      # µ update steps per batch (Algorithm 1)
        self.clip_epsilon        = 0.3    # ε clipping for ratio
        self.kl_coef             = 0.2    # β KL penalty weight

        # coeficient pentru penalizarea față de modelul de referință
        # (se aplică pe |Δlogp| mediu per token)
        self.target_kl           = 0.15
        self.kl_adapt_rate       = 1.5
        self.kl_window           = 50
        self.max_kl_coef         = 1.0
        self.min_kl_coef         = 1e-4
        self.kl_stop_factor      = 1.5  # early-stop epocă PPO dacă |Δ| depășește factor * target

        # note: policy_ref never changes; policy_old gets copied each batch

        # === Training ===
        self.n_epochs            = 6
        self.batch_size          = 8
        self.lr                  = 1e-5
        self.warmup_steps        = 500
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
        self.log_dir             = f"runs/grpo_{self.timestamp}"
        self.save_dir            = f"models/grpo_qlora_{self.timestamp}"  # corectat (qlora)


        # === Optional extras ===
        # if you want keyword-based or other shaping rewards
        # self.length_bonus        = 0.1
        # self.diversity_coef      = 0.2

        # === Replay buffer & micro-batching pe număr de tokeni ===
        self.buffer_size = 100              # mărimea bufferului de replay
        self.replay_batch_size = 16         # câte mostre tragi pentru off-policy update
        self.token_microbatch_size = 8000   # ținta de tokeni / micro-batch
        self.microbatch_size = 4
