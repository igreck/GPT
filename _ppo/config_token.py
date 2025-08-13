import torch
from datetime import datetime

class Config:
    def __init__(self):
        # === Logging / identificare rulare ===
        self.timestamp = datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
        self.log_dir = f"runs/ppo_math_{self.timestamp}"
        self.save_dir = f"models/ppo_math_qlora_{self.timestamp}"
        self.save_every = 5
        self.log_every = 10

        # === Modele ===
        self.policy_model_name = "Qwen/Qwen3-4B-Base"
        self.reward_model_name = "lvwerra/distilbert-imdb"

        # Adaptoare SFT (LoRA) inițiale
        self.lora_from_sft_dir = "./models/sft_math_qlora_qwen_4b"
        self.sft_dir = self.lora_from_sft_dir
        # reluare PPO (opțional)
        self.resume_dir = None

        # Device
        self.device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
        self.reward_device = "cpu"

        # Dim. hidden pentru value head
        self.value_hidden_dim = 512

        # === PPO / RLHF ===
        self.value_loss_weight = 0.3
        self.entropy_weight = 0.005
        self.clip_epsilon = 0.3
        self.clip_epsilon_final = 0.05

        # KL adaptive control
        self.target_kl = 0.15
        self.kl_coef = 0.2
        self.kl_adapt_rate = 1.5
        self.kl_window = 50
        self.max_kl_coef = 1.0
        self.min_kl_coef = 1e-4
        self.kl_stop_factor = 1.3  # mai strict decât 1.5

        # Value clipping
        self.value_clip_range = 0.3

        # === Antrenare ===
        self.n_epochs = 6
        self.lr = 1e-5
        self.batch_size = 2
        self.microbatch_size = 1
        self.accum_steps = 2
        self.ppo_epochs = 4
        self.warmup_steps = 500

        # === Generare ===
        self.max_new_tokens = 1024
        self.top_p = 0.9
        self.temperature = 0.5
        self.repetition_penalty = 1.1

        # === Lungimi ===
        self.policy_max_length = 2048
        self.reward_max_length = 512

        # === Date ===
        self.prompt_ratio_range = (0.45, 0.7)
        self.split = "train[:10%]"
        self.shuffle = True
        self.num_workers = 1
        self.seed = 42

        # === Avansat PPO ===
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.target_entropy = -1.5
        self.entropy_adapt_lr = 1e-4
        self.min_entropy_coef = 0.0
        self.max_entropy_coef = 0.1

        # === Reward shaping ===
        self.use_margin_reward = True
        self.add_length_bonus = True
        self.min_words_bonus = 8
        self.length_bonus = 0.1
        self.diversity_coef = 0.2
        # shaping incremental pe token
        self.rep_ngram = 3
        self.rep_penalty = 0.01
        self.rouge_shaping = 0.1

        # === Replay buffer / micro-batching pe tokeni ===
        self.buffer_size = 100
        self.replay_batch_size = 16
        self.token_microbatch_size = 3000
