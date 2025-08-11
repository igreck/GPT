import re
import torch

class RewardFunctions:
    def __init__(self, tokenizer, device):
        self.tokenizer = tokenizer
        self.device = device
        # Make sure we have a pad_token
        if self.tokenizer.pad_token_id is None:
             self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        # **VERY IMPORTANT** for decoder-only models
        self.tokenizer.padding_side = "left"

        self.reasoning_start = "<start_working_out>" # Acts as <think>
        self.reasoning_end   = "<end_working_out>"   # Acts as </think>
        self.solution_start  = "<SOLUTION>"
        self.solution_end    = "</SOLUTION>"
 
        # Add optional EOS token matching
        self.solution_end_regex = r"</SOLUTION>[\s]{0,}" + \
            "(?:" + re.escape(self.tokenizer.eos_token) + ")?"

        self.match_format = re.compile(
            rf"{self.reasoning_end}.*?"
            rf"{self.solution_start}(.+?){self.solution_end_regex}"\
            rf"[\s]{{0,}}$",
            flags = re.MULTILINE | re.DOTALL
        )

        self.match_numbers = re.compile(
            self.solution_start + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags = re.MULTILINE | re.DOTALL
        )

    def match_format_exactly(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores


    def match_format_approximately(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion
            # Count how many keywords are seen - we penalize if too many!
            # If we see 1, then plus some points!

            # No need to reward <start_working_out> since we always prepend it!
            # score += 0.5 if response.count(reasoning_start) == 1 else -1.0
            score += 0.5 if response.count(self.reasoning_end)   == 1 else -1.0
            score += 0.5 if response.count(self.solution_start)  == 1 else -1.0
            score += 0.5 if response.count(self.solution_end)    == 1 else -1.0
            scores.append(score)
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):
        question = prompts[1]["content"]
        responses = completions

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(-2.0)
                continue
            # Correct answer gets 5 points!
            if guess == true_answer:
                score += 5.0
            # Match if spaces are seen, but less reward
            elif guess.strip() == true_answer.strip():
                score += 3.5
            else:
                # We also reward it if the answer is close via ratios!
                # Ie if the answer is within some range, reward it!
                try:
                    ratio = float(guess) / float(true_answer)
                    if   ratio >= 0.9 and ratio <= 1.1: score += 2.0
                    elif ratio >= 0.8 and ratio <= 1.2: score += 1.5
                    else: score -= 2.5 # Penalize wrong answers
                except:
                    score -= 4.5 # Penalize
            scores.append(score)
        return scores



    def check_numbers(self, prompts, completions, answer, **kwargs):
        question = prompts[1]["content"]
        responses = completions

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None \
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(-2.5)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                # Remove commas like in 123,456
                guess       = float(guess.strip().replace(",", ""))
                scores.append(3.5 if guess == true_answer else -1.5)
            except:
                scores.append(0)
                continue
        return scores


    def compute_all_rewards(
        self,
        prompts,
        completions,
        answers=None,
        *,
        weights=None,              # ex: {"match_format_exactly":1.0, "match_format_approximately":0.5, "check_answer":2.0, "check_numbers":1.0}
        reduce: str = "sum",       # "sum" or "mean" across components
        device=None,
        dtype=torch.float32,
        return_components: bool = False,
        **kwargs
    ):
        """
        Compute rewards using all scoring methods and return a tensor of accumulated rewards per completion.
        Optionally returns the individual component tensors when `return_components=True`.
        """
        B = len(completions)
        device = device or getattr(self, "device", None) or "cpu"

        # --- component scores ---
        exact_fmt = self.match_format_exactly(completions, **kwargs)
        approx_fmt = self.match_format_approximately(completions, **kwargs)

        if answers is not None:
            answer_scores = self.check_answer(prompts, completions, answers, **kwargs)
            number_scores = self.check_numbers(prompts, completions, answers, **kwargs)
        else:
            answer_scores = [0.0] * B
            number_scores = [0.0] * B

        comps = {
            'match_format_exactly': torch.as_tensor(exact_fmt, dtype=dtype, device=device),
            'match_format_approximately': torch.as_tensor(approx_fmt, dtype=dtype, device=device),
            'check_answer': torch.as_tensor(answer_scores, dtype=dtype, device=device),
            'check_numbers': torch.as_tensor(number_scores, dtype=dtype, device=device)
        }

        # sanitize shapes and values
        for k in comps:
            comps[k] = torch.nan_to_num(comps[k], nan=0.0, posinf=0.0, neginf=0.0)
            if comps[k].ndim != 1 or comps[k].shape[0] != B:
                comps[k] = comps[k].reshape(-1)[:B]
                if comps[k].shape[0] < B:
                    pad = torch.zeros(B - comps[k].shape[0], dtype=dtype, device=device)
                    comps[k] = torch.cat([comps[k], pad], dim=0)

        # weights
        default_weights = {
            'match_format_exactly': 1.0,
            'match_format_approximately': 1.0,
            'check_answer': 1.0,
            'check_numbers': 1.0
        }
        if weights is not None:
            default_weights.update(weights)
        w = {k: float(default_weights.get(k, 0.0)) for k in comps.keys()}

        # aggregate
        stacked = torch.stack([comps[k] * w[k] for k in comps.keys()], dim=0)  # [num_comp, B]
        if reduce == 'mean':
            total = stacked.mean(dim=0)
        else:
            total = stacked.sum(dim=0)

        if return_components:
            return total, comps  # total: [B], comps: dict[str, Tensor[B]]
        return total  # Tensor[B]

