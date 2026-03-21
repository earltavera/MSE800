import functools
import time
import logging

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"), # Saves to file
        logging.StreamHandler()         # Prints to console
    ]
)

def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter() # High-precision timer
        
        logging.info(f"Executing '{func.__name__}' | Args: {args} | Kwargs: {kwargs}")
        
        try:
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            logging.info(f"Success: '{func.__name__}' | Result: {result} | Time: {duration:.6f}s")
            return result
            
        except Exception as e:
            end_time = time.perf_counter()
            logging.error(f"Error in '{func.__name__}' after {end_time - start_time:.6f}s: {e}")
            return None # Or re-raise with 'raise e' depending on your needs
            
    return wrapper

@log_decorator
def add(a, b):
    return a + b

@log_decorator
def divide(a, b):
    return a / b

# Testing the new features
if __name__ == "__main__":
    add(10, 25)
    divide(10, 2)
    divide(10, 0)  # This will trigger the error handling