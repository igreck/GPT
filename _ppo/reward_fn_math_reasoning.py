import re

class Rewarder:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
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
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None: score += 3.0
            scores.append(score)
        return scores


    def match_format_approximately(self, completions, **kwargs):
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
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
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

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
        question = prompts[0][-1]["content"]
        responses = [completion[0]["content"] for completion in completions]

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


    def compute_all_rewards(self, prompts, completions, answers=None, **kwargs):
        """
        Compute rewards using all scoring methods and return a dict of scores.
        """
        # Format-based rewards
        exact_fmt = self.match_format_exactly(completions, **kwargs)
        approx_fmt = self.match_format_approximately(completions, **kwargs)
        # Content-based rewards (if answers provided)
        answer_scores = self.check_answer(prompts, completions, answers, **kwargs) if answers is not None else [0.0]*len(completions)
        number_scores = self.check_numbers(prompts, completions, answers, **kwargs) if answers is not None else [0.0]*len(completions)
        return {
            'match_format_exactly': exact_fmt,
            'match_format_approximately': approx_fmt,
            'check_answer': answer_scores,
            'check_numbers': number_scores
        }