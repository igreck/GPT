import torch

class Config:
    def __init__(self):
        # Modele
        self.policy_model_name = "gpt2-medium"
        self.reward_model_name = "lvwerra/distilbert-imdb"

        # Hiperparametri PPO
        self.value_loss_weight = 0.3
        self.entropy_weight = 0.02
        self.clip_epsilon = 0.2
        self.kl_coef = 0.05

        # Train
        self.n_epochs = 10
        self.lr = 5e-6
        self.microbatch_size = 8
        self.accum_steps = 4  # numărul de micro-batch-uri pentru acumulare
        self.batch_size = 32

        # Generare
        self.max_new_tokens = 128
        self.top_p = 0.9
        self.top_k = 0.0
        self.temperature = 0.5
        self.repetition_penalty = 1.2

        # Lungimi (separate)
        self.policy_max_length = 256   # prompt/policy
        self.reward_max_length = 512   # clasificator (BERT/Roberta)

        # Dataset
        self.prompt_ratio_range = (0.1, 0.2)
        self.split = "train[:10%]"
        self.shuffle = True
        self.num_workers = 1
        self.seed = 42

        # Device (include MPS)
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available()
                       else "cpu")

        # Logging / Saving
        self.log_every = 1
        self.save_every = 5

        # PPO/RL hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.warmup_steps = 500
        self.ppo_epochs = 3
        self.target_kl = 0.10          # țintă pentru KL
        self.kl_adapt_rate = 1.5       # cât multiplici/divizezi coeficientul
        self.kl_window = 50            # la câți pași să adaptezi (1 = la fiecare pas)
        self.max_kl_coef = 1.0
        self.min_kl_coef = 1e-4
        self.value_clip_range = 0.2 
        self.early_stop_patience = 3
        # Logging and checkpointing
        self.log_dir = "runs/ppo_rl"
        self.save_dir = "gpt2_rl_final"
        self.log_interval = self.log_every