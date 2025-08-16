import torch

class _StopOnSubsequence(torch.nn.Module):
    """
    Helper for stopping generation when a given subsequence appears at the end of the completion (not in the prompt).
    Used as a StoppingCriteria in HuggingFace generate.
    """
    def __init__(self, stop_ids, prompt_lens):
        super().__init__()
        self.stop_ids = stop_ids
        self.prompt_lens = prompt_lens  # list of prompt lengths for each sample in batch
        self.stop_len = len(stop_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # input_ids: [B, cur_len]
        B = input_ids.size(0)
        # For each sample, check if the tail of the completion matches stop_ids
        all_stopped = True
        for i in range(B):
            prompt_len = self.prompt_lens[i]
            seq = input_ids[i]
            comp = seq[prompt_len:]  # completion tokens generated so far
            if comp.size(0) < self.stop_len:
                all_stopped = False
                continue
            if not torch.equal(comp[-self.stop_len:], torch.tensor(self.stop_ids, device=seq.device)):
                all_stopped = False
        return all_stopped
