import torch
from datetime import datetime

class Config:
    def __init__(self):
        # Jurnal
        self.timestamp = datetime.now().strftime("%Y-%d-%m-%H-%M-%S")
        # === Models ===
        self.policy_model_name = "Qwen/Qwen3-4B-Base"
        self.reward_model_name = "lvwerra/distilbert-imdb"

        # SFT adapters (LoRA) de la care pornești PPO
        self.lora_from_sft_dir = "./models/sft_math_qlora_qwen_4b"
        # alias pentru compatibilitate cu codul principal
        self.sft_dir = self.lora_from_sft_dir

        # reluare PPO dintr-un director cu adapters + value_head.pt (opțional)
        # self.resume_dir = "./models/ppo_imdb_qlora_qwen_best_iter2"
        self.resume_dir = None

        # device pentru reward model (implicit același cu policy)
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
                       else "cpu")
        # self.reward_device = self.device
        self.reward_device = self.device

        # value head dim (pentru PPO)
        self.value_hidden_dim = 512

        # === PPO / RLHF terms ===
        self.value_loss_weight = 0.3
        self.entropy_weight = 0.0  # start stabil, crește dinamic cu scheduler
        self.clip_epsilon = 0.10
        self.clip_epsilon_final = 0.05  # scheduling liniar pe parcursul epocilor PPO
        self.reward_scale = 3.0            # factor global de scalare a reward-ului normalizat (mărește amplitudinea avantajului)

        # coeficient pentru penalizarea față de modelul de referință (forward-KL token-level)
        self.target_kl = 0.06
        self.kl_coef = 0.20
        self.kl_adapt_rate = 1.5
        self.kl_window = 32
        self.max_kl_coef = 0.2
        self.min_kl_coef = 1e-4
        self.kl_stop_factor = 1.2  # oprește mai devreme când KL sare de target

        # Value clipping
        self.value_clip_range = 0.2

        # === Train (batching) ===
        self.n_epochs = 6
        self.lr = 1e-5
        self.batch_size = 4
        self.microbatch_size = 2
        self.accum_steps = 2
        self.ppo_epochs = 4
        self.warmup_steps = 500

        # === Generation ===
        self.max_new_tokens = 1024  # mai lung pentru a prinde <SOLUTION>…</SOLUTION>
        self.top_p = 0.75
        self.temperature = 0.25
        self.repetition_penalty = 1.15

        # === Lengths ===
        self.policy_max_length = 2048
        self.reward_max_length = 512

        # === Data ===
        self.prompt_ratio_range = (0.45, 0.7)
        self.split = "train[:10%]"
        self.shuffle = True
        self.num_workers = 1
        self.seed = 42

        # === Logging / saving ===
        self.log_every = 10
        self.log_interval = self.log_every
        self.save_every = 5
        self.log_dir = f"runs/ppo_math_{self.timestamp}"
        self.save_dir = f"models/ppo_math_qlora_{self.timestamp}"  # corectat (qlora)

        # === PPO advanced hyperparameters ===
        self.gamma = 0.99
        self.lambda_gae = 0.95
        self.target_entropy = -1.5  # încurajează entropia mai mică (răspunsuri mai coerente)
        self.entropy_adapt_lr = 1e-4        # learning rate pentru adaptarea coef. de entropie
        self.min_entropy_coef = 0.0         # limită inferioară pentru coef. de entropie
        self.max_entropy_coef = 0.1         # limită superioară pentru coef. de entropie

        # === Reward shaping extras (folosite în PPOAgent.compute_sentiment_reward) ===
        self.use_margin_reward = True      # dacă True: sigmoid(logit_pos - logit_neg)
        self.add_length_bonus = True
        self.min_words_bonus = 8
        self.length_bonus = 0.1
        self.diversity_coef = 0.05        # bonus mic de diversitate (bigrame unice) – nu sabota concizia

        # === Replay buffer & micro-batching pe număr de tokeni ===
        self.buffer_size = 100              # mărimea bufferului de replay
        self.replay_batch_size = 16         # câte mostre tragi pentru off-policy update
        self.token_microbatch_size = 3000   # ținta de tokeni / micro-batch
        self.format_warmup_steps = 300  # număr de pași PPO cu accent pe format înainte de răspuns corect