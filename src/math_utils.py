import re
from typing import List, Optional
import sympy
from sympy.parsing.latex import parse_latex


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract the answer from LaTeX \\boxed{} format.
    
    Args:
        text: The text containing the boxed answer
        
    Returns:
        The extracted answer string, or None if not found
    """
    # Try to find \\boxed{...} pattern
    # Use a more robust regex that handles nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the last boxed answer (final answer)
        return matches[-1].strip()
    
    # Fallback: try to find content after common answer indicators
    answer_patterns = [
        r'(?:answer|Answer|ANSWER):\s*([^\n]+)',
        r'(?:final answer|Final Answer|FINAL ANSWER):\s*([^\n]+)',
        r'(?:result|Result|RESULT):\s*([^\n]+)',
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].strip()
    
    return None


def normalize_answer(answer: str) -> str:
    """
    Normalize a mathematical answer for comparison.
    
    Args:
        answer: The answer string to normalize (LaTeX format)
        
    Returns:
        Normalized answer string
    """
    if answer is None:
        return ""
    
    # Remove extra whitespace
    answer = answer.strip()
    
    # Remove common LaTeX commands that don't affect the answer
    answer = answer.replace('\\,', '')
    answer = answer.replace('\\:', '')
    answer = answer.replace('\\;', '')
    answer = answer.replace('\\!', '')
    answer = answer.replace('\\text', '')
    answer = answer.replace('{', '')
    answer = answer.replace('}', '')
    
    # Remove dollar signs
    answer = answer.replace('$', '')
    
    # Normalize whitespace
    answer = ' '.join(answer.split())
    
    # Try to parse as LaTeX and convert to sympy for normalization
    try:
        # Parse the LaTeX expression
        expr = parse_latex(answer)
        # Convert back to string in canonical form
        answer = str(expr)
    except:
        # If parsing fails, just use the cleaned string
        pass
    
    # Remove spaces around operators for consistency
    answer = answer.replace(' ', '')
    
    # Convert to lowercase for case-insensitive comparison
    answer = answer.lower()
    
    return answer


def is_equiv(str1: str, str2: str) -> bool:
    """
    Check if two mathematical expressions are equivalent.
    
    Args:
        str1: First expression
        str2: Second expression
        
    Returns:
        True if expressions are equivalent, False otherwise
    """
    if str1 is None or str2 is None:
        return str1 == str2
    
    # First try exact string match after normalization
    norm1 = normalize_answer(str1)
    norm2 = normalize_answer(str2)
    
    if norm1 == norm2:
        return True
    
    # Try to evaluate as sympy expressions
    try:
        expr1 = parse_latex(str1) if '\\' in str1 else sympy.sympify(str1)
        expr2 = parse_latex(str2) if '\\' in str2 else sympy.sympify(str2)
        
        # Check if expressions are equal
        diff = sympy.simplify(expr1 - expr2)
        return diff == 0
    except:
        # If sympy fails, fall back to string comparison
        return norm1 == norm2


def compute_pass_at_1(predictions: List[str], references: List[str]) -> dict:
    """
    Compute Pass@1 accuracy for mathematical reasoning.
    
    Pass@1 measures the percentage of problems where the model's
    first generated answer exactly matches the correct answer.
    
    Args:
        predictions: List of predicted answers
        references: List of reference (correct) answers
        
    Returns:
        Dictionary containing pass@1 score and additional metrics
    """
    if len(predictions) != len(references):
        raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(references)} references")
    
    if len(predictions) == 0:
        return {
            "pass@1": 0.0,
            "total": 0,
            "correct": 0,
            "no_answer_extracted": 0
        }
    
    correct = 0
    no_answer_extracted = 0
    
    for pred, ref in zip(predictions, references):
        # Extract boxed answers
        pred_answer = extract_boxed_answer(pred)
        ref_answer = extract_boxed_answer(ref)
        
        # If we couldn't extract from reference, use the whole reference
        if ref_answer is None:
            ref_answer = ref
        
        # Track cases where we couldn't extract an answer from prediction
        if pred_answer is None:
            no_answer_extracted += 1
            # Try using the whole prediction as fallback
            pred_answer = pred
        
        # Check if answers are equivalent
        if is_equiv(pred_answer, ref_answer):
            correct += 1
    
    pass_at_1 = correct / len(predictions)
    
    return {
        "pass@1": pass_at_1,
        "total": len(predictions),
        "correct": correct,
        "no_answer_extracted": no_answer_extracted
    }
