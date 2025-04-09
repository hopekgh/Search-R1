import re
import random
from transformers.utils import logging
# def extract_solution(solution_str):
#     #find the solution in the string between [answer] and [answer]
#     solution = re.search("\\[answer\\](.*?)\\[answer\\]", solution_str)
#     assert solution is not None
#     final_solution = solution.group(0)
#     return final_solution
logger = logging.get_logger(__name__)

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    #<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    elif "<|eot_id|>" in solution_str:
        solution_str = solution_str.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("no first thing found\n")
        return None
    
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        print("No match\n")
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()

def check_too_many_answer(solution_str):
    return solution_str.count("[answer]") > 5

def check_is_correct(answer, ground_truth):
    # 동일한지 확인
    return ground_truth == answer

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.1, partial_score=0.5):
    """The scoring function for GSM8k.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth. 만약 딕셔너리 형태라면 {'target': 'D'}에서 'D'를 사용.
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
        partial_score: score awarded if ground truth is contained in answer
    """
    # random between 0 to 64 int and if it is 0, do print
    do_print = random.randint(0, 2) == 0
    if do_print:
        print(f"Solution: {solution_str}, Ground Truth: {ground_truth}")
    answer = extract_solution(solution_str=solution_str)
    if answer is None:
        final_score = 0.0
        return final_score

    normalized_answer = answer
    normalized_truth = ground_truth['target']


    is_correct = check_is_correct(normalized_answer, normalized_truth)
    final_score = 0.0
    if do_print:
        print(f"Answer: {normalized_answer}, Ground Truth: {normalized_truth}, Is Correct: {is_correct}")
        #logger.info(f"Answer: {normalized_answer}, Ground Truth: {normalized_truth}, Is Correct: {is_correct}")
    
    if is_correct:
        final_score = score
    else:
        final_score = format_score

    if check_too_many_answer(solution_str):
        final_score = final_score - format_score

    if do_print:
        print(f"Final Score: {final_score}")
        #logger.info(f"Final Score: {final_score}")
    return final_score