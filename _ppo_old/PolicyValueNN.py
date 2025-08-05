import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel

# GPT-2 with value head (shared body)
class GPT2WithValueHead(nn.Module):
    """GPT-2 language model with an attached value head for reinforcement learning."""
    def __init__(
        self,
        model_name: str = "gpt2",
        value_hidden_dim: int = 256,
        use_gradient_checkpointing: bool = False,
        use_cache: bool = True
    ):
        """
        Initialize GPT-2 with a value head.
        Args:
            model_name: pretrained model identifier.
            value_hidden_dim: hidden dimension size for the value head.
            use_gradient_checkpointing: enable gradient checkpointing to save memory.
            use_cache: enable the model’s generation cache for faster inference.
        """
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        # Configure caching and optional gradient checkpointing
        self.model.config.use_cache = use_cache
        if use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.value_head = nn.Sequential(
            nn.Linear(self.model.config.n_embd, value_hidden_dim),
            nn.Tanh(),
            nn.Linear(value_hidden_dim, 1)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: torch.LongTensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return (logits [B, T, V], values [B]) given inputs and labels."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]  # [B, T, H]
        last_token_index = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_index]  # [B, H]
        values = self.value_head(last_hidden).squeeze(-1)  # [B]
        return logits, values

    def value_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor
    ) -> torch.FloatTensor:
        """Compute only the value estimates under mixed precision."""
        with autocast():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            last_token_index = attention_mask.sum(dim=1) - 1
            last_hidden = hidden_states[torch.arange(hidden_states.size(0)), last_token_index]
            values = self.value_head(last_hidden).squeeze(-1)
        return values

    def generate(
        self,
        *args,
        output_scores: bool = False,
        return_dict_in_generate: bool = False,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate sequences, optionally returning scores and full dict output.
        """
        return self.model.generate(*args, output_scores=output_scores, return_dict_in_generate=return_dict_in_generate, **kwargs)
    
    def save_pretrained(self, path: str) -> None:
        """Save the model and value head to the given path."""
        self.model.save_pretrained(path)
        torch.save(self.value_head.state_dict(), path + "/value_head.pt")
    
    def load_pretrained(self, path: str) -> None:
        """Load the model and value head from the given path."""
        self.model = GPT2LMHeadModel.from_pretrained(path)
        state_dict = torch.load(path + "/value_head.pt")
        self.value_head.load_state_dict(state_dict)
