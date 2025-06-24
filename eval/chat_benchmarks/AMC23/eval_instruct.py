import json
import logging
from typing import Any, Dict, List, Optional

import lm_eval.models
import numpy as np
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.tasks.hendrycks_math.utils import is_equiv, last_boxed_only_string, remove_boxed

from eval.task import BaseBenchmark

# Modified version of hendrycks_math with additional instruction to mark the solution with \\boxed
# https://github.com/mlfoundations/evalchemy/blob/e70a45e41cb2ada273d6bb98e75dba303ec31f8b/eval/chat_benchmarks/AMC23/eval_instruct.py#L15
PROMPT = """Problem: {problem}\nMark your solution with \\boxed\nAnswer:"""


class AMC23Benchmark(BaseBenchmark):
    """
    AMC23 Benchmark for evaluating the math reasoning of LLMs.
    Link: https://huggingface.co/datasets/zwhe99/amc23

    Follows the evaluation logic of hendrycks_math answer extraction.
    Added additional instruction to the prompt to mark the solution with \\boxed.
    """

    def __init__(
        self,
        data_file: str = "eval/chat_benchmarks/AMC23/data/amc23.json",
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 32768,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize AMC23 benchmark.

        Args:
            data_file: File containing the AMC23 dataset (id, problem, reference_solution, expected_answer, source)
            debug: If set, only evaluate on 2 examples
            seed: Random seed for reproducibility. Default is [0, 1234, 1234, 1234] for lm-eval-harness.
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.data_file = data_file
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.n_repeat = 10

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and temporary directory,
            or None for non-primary ranks
        """
        examples = self.load_questions()

        # Prepare instances for model
        all_instances = []
        if isinstance(model, lm_eval.models.huggingface.HFLM):
            model_name = model.pretrained
        elif isinstance(model, lm_eval.models.openai_completions.OpenAIChatCompletion):
            model_name = str(f"openai/{model.model}")
        else:
            model_name = model.model_args["model"]

        all_outputs = []
        for i in range(self.n_repeat):
            seed = [s + i for s in self.seed]
            all_instances = []
            for idx, example in enumerate(examples):
                messages = [
                    {"role": "user", "content": PROMPT.format(problem=example["question"])},
                ]

                templated_messages = self._prepare_messages(messages, model)

                instance = Instance(
                    "generate_until",
                    example,
                    (
                        templated_messages,
                        {
                            "do_sample": False,
                            "max_new_tokens": self.max_new_tokens,
                            "temperature": 0.7,
                            "seed": seed,
                        },
                    ),
                    idx,
                )

                # Add repetition information to instance metadata
                instance.repeat_idx = i
                instance.metadata = {
                    "problem_id": str(example["id"]) if "id" in example else str(idx),
                    "expected_answer": str(example["answer"]),
                    "reference_solution": str(example["solution"]) if "solution" in example else "",
                }

                all_instances.append(instance)

            # Generate model responses
            self.logger.info("Generating responses for AMC23...")
            outputs = self.compute(model, all_instances)
            all_outputs.append(outputs)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        for example, outputs in zip(examples, zip(*all_outputs)):
            example["model_outputs"] = list(outputs)
            example["model_answers"] = [self.extract_answer(o) for o in outputs]

        return {"examples": examples}

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated solution completions."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        num_questions = len(examples)

        # Calculate accuracy for each repetition
        all_results = []
        for i in range(self.n_repeat):
            solved = sum([is_equiv(str(example["answer"]), example["model_answers"][i]) for example in examples])
            all_results.append(
                {
                    "repetition": i + 1,
                    "num_total": num_questions,
                    "num_solved": solved,
                    "accuracy": solved / num_questions,
                }
            )

        # Calculate overall statistics
        solved_avg = np.mean([result["num_solved"] for result in all_results])
        accuracy_avg = np.mean([result["accuracy"] for result in all_results])
        accuracy_std = np.std([result["accuracy"] for result in all_results])
        accuracy_std_err = np.std([result["accuracy"] for result in all_results]) / np.sqrt(self.n_repeat)

        results.update(
            {
                "num_total": num_questions,
                "solved_avg": solved_avg,
                "run_stats": all_results,
                "accuracy_avg": accuracy_avg,
                "accuracy_std_err": accuracy_std_err,
                "num_repeat": self.n_repeat,
            }
        )
        return results

    def load_questions(self) -> List[Dict[str, str]]:
        """Load AMC23 questions from the data file."""
        with open(self.data_file, "r") as f:
            questions = [json.loads(x) for x in f]
        self.logger.info(f"Loaded {len(questions)} questions from {self.data_file}")
        return questions

    def extract_answer(self, output: str) -> str:
        """Extract the final answer from a model-generated solution, which is expected to be in the format of \boxed{answer}.

        Uses the same logic as hendrycks_math.

        Args:
            output (str): Model-generated solution text

        Returns:
            str: Extracted final answer. Returns empty string if no answer found in \boxed.
        """
        try:
            # 먼저 기본 방법으로 시도
            answer = remove_boxed(last_boxed_only_string(output))
            if answer:  # 빈 문자열이 아니면 반환
                return answer
        except:
            pass
        
        try:
            # 응답이 잘린 경우를 위한 대안적 방법들
            # 1. \\boxed{ 패턴 찾기 (닫힌 괄호가 없어도)
            import re
            boxed_pattern = r'\\boxed\{([^}]*)\}'
            matches = re.findall(boxed_pattern, output)
            if matches:
                return matches[-1]  # 마지막 매치 반환
            
            # 2. "Final Answer"나 "Answer:" 뒤의 숫자 찾기
            final_answer_patterns = [
                r'Final Answer[:\s]*.*?(\d+)',
                r'Answer[:\s]*(\d+)',
            ]
            
            for pattern in final_answer_patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    return matches[-1]  # 마지막 매치 반환
            
            # 3. 가장 마지막에 나타나는 숫자 (일반적으로 답일 가능성이 높음)
            numbers = re.findall(r'\b\d+\b', output)
            if numbers:
                return numbers[-1]
                
        except:
            pass
            
        return ""
