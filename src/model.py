"""
LLM API wrappers and utilities for prompt-based inference.
"""

import os
import time
from typing import Dict, List, Optional, Any
import openai


class LLMClient:
    """Wrapper for LLM API calls with retry logic."""
    
    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        api_key_env: str = "OPENAI_API_KEY",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the model (e.g., gpt-3.5-turbo)
            provider: API provider (currently only openai)
            api_key_env: Environment variable name for API key
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.model_name = model_name
        self.provider = provider
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get API key from environment
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")
        
        if provider == "openai":
            openai.api_key = api_key
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        n: int = 1,
        **kwargs
    ) -> List[str]:
        """
        Generate completions from the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            List of generated text completions
        """
        for attempt in range(self.max_retries):
            try:
                if self.provider == "openai":
                    response = openai.ChatCompletion.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=n,
                        **kwargs
                    )
                    
                    # Extract completions
                    completions = [choice.message.content for choice in response.choices]
                    return completions
                
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise
        
        return []


def verify_calc_block(calc_block: str, claimed_answer: float, tolerance: float = 1e-6) -> bool:
    """
    Verify a CALC block by executing it in a safe Python environment.
    
    Args:
        calc_block: String containing arithmetic expressions (one per line)
        claimed_answer: The answer claimed in FINAL:
        tolerance: Tolerance for floating point comparison
    
    Returns:
        True if CALC block is valid and computes the claimed answer
    """
    import ast
    import operator
    
    # Whitelist of allowed operations
    ALLOWED_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }
    
    def safe_eval(node, variables):
        """Safely evaluate an AST node with only whitelisted operations."""
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):  # Python 3.7 compatibility
            return node.n
        elif isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Undefined variable: {node.id}")
            return variables[node.id]
        elif isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_OPS:
                raise ValueError(f"Disallowed operation: {type(node.op).__name__}")
            left = safe_eval(node.left, variables)
            right = safe_eval(node.right, variables)
            return ALLOWED_OPS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPS:
                raise ValueError(f"Disallowed operation: {type(node.op).__name__}")
            operand = safe_eval(node.operand, variables)
            return ALLOWED_OPS[type(node.op)](operand)
        else:
            raise ValueError(f"Disallowed node type: {type(node).__name__}")
    
    try:
        # Parse CALC block line by line
        variables = {}
        lines = calc_block.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Check if it's an assignment
            if '=' in line:
                var_name, expr = line.split('=', 1)
                var_name = var_name.strip()
                expr = expr.strip()
                
                # Parse and evaluate the expression
                tree = ast.parse(expr, mode='eval')
                result = safe_eval(tree.body, variables)
                variables[var_name] = result
            else:
                # Standalone expression (shouldn't happen in well-formed CALC)
                tree = ast.parse(line, mode='eval')
                result = safe_eval(tree.body, variables)
        
        # Check if 'answer' variable was computed
        if 'answer' not in variables:
            return False
        
        # Check if computed answer matches claimed answer
        computed_answer = float(variables['answer'])
        return abs(computed_answer - claimed_answer) < tolerance
    
    except Exception as e:
        # Any parsing or evaluation error means verification failed
        return False


def score_with_verifier(
    llm_client: LLMClient,
    problem: str,
    solution: str,
    verifier_prompt_template: str,
) -> float:
    """
    Score a solution using an LLM verifier.
    
    Args:
        llm_client: LLM client to use for scoring
        problem: The original problem
        solution: The candidate solution to score
        verifier_prompt_template: Prompt template with {problem} and {solution} placeholders
    
    Returns:
        Numeric score (0-10)
    """
    # Format verifier prompt
    prompt = verifier_prompt_template.format(problem=problem, solution=solution)
    
    # Get score from LLM
    try:
        responses = llm_client.generate(prompt, temperature=0.0, max_tokens=10, n=1)
        response = responses[0].strip()
        
        # Extract numeric score
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            score = float(match.group(1))
            # Clamp to 0-10 range
            return max(0.0, min(10.0, score))
        else:
            # If we can't extract a score, return neutral score
            return 5.0
    except Exception as e:
        print(f"Verifier scoring failed: {e}")
        return 5.0
