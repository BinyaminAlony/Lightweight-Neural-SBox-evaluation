import cProfile
import pstats
import io
import atexit
import sys
from typing import Callable, Any

def get_cumulative_runtime(func: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
    """
    Profiles the cumulative runtime of a given function with arguments,and saves the profiling results to a file. The results are saved even if the function calls sys.exit().
    
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
    profiler = cProfile.Profile()
    profiler.enable()

    # Generate filename dynamically using function name
    file_name = f"profile_{func.__name__}.txt"

    def save_profile():
        """This function runs on exit to save profiling results."""
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.strip_dirs().sort_stats("cumtime").print_stats(50)  # Show top 50 functions

        with open(file_name, "w") as f:
            f.write(stream.getvalue())

        print(f"Profiling complete! Results saved in '{file_name}'.")

    # Ensure profile data is saved no matter how the program exits
    atexit.register(save_profile)

    try:
        func(*args, **kwargs)
    except SystemExit:
        pass  # Prevent immediate exit to allow atexit to run

    # If the function completes normally (no sys.exit), save profile data now
    save_profile()

# Example usage:
# from my_script import my_function
# get_cumulative_runtime(my_function, arg1, arg2, key=value)
