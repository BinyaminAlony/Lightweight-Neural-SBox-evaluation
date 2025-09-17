# Best Practices for Argument Type and Documentation Formatting

## Overview
This guide outlines best practices for documenting function signatures using argument type hints and structured docstrings. Following these conventions ensures clarity, maintainability, and consistency across projects.

## Function Documentation Format
Each function should include a detailed docstring that describes its purpose, parameters, and return values. The format should follow these guidelines:

### General Structure
```python
 def function_name(param1: Type1, param2: Type2, *args: Type3, **kwargs: Type4) -> ReturnType:
     """
     Brief description of the function.

     -----------
     Parameters:
         param1: (Type1)
             Description of the first parameter.
         param2: (Type2)
             Description of the second parameter.
         *args: (Type3)
             Description of additional positional arguments.
         **kwargs: (Type4)
             Description of additional keyword arguments.

     -----------
     Returns:
         ReturnType
             Description of the return value.
     """
     pass
```

### Example Implementation
```python
from typing import Callable, Any

def get_cumulative_runtime(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """
    Profiles the cumulative runtime of a given function with arguments and saves the profiling results to a file.
    The results are saved even if the function calls sys.exit().

    -----------
    Parameters:
        func:  (Callable[..., Any])
            The function to be profiled.
        *args: (Any)
            Positional arguments to pass to the function.
        **kwargs: (Any)
            Keyword arguments to pass to the function.

    -----------
    Returns:
        None
    """
    pass
```

## Key Principles
1. **Consistent Formatting:** Use a consistent section separator (e.g., `-----------`).
2. **Explicit Type Annotations:** Use type hints for all function parameters and return types.
3. **Descriptive Parameter Explanations:** Each parameter should have a brief but clear description.
4. **Clear Return Documentation:** Always describe the return value, even if the function returns `None`.
5. **Readability:** Maintain readability with proper indentation and spacing.

Following these best practices will help improve code documentation quality and maintainability.