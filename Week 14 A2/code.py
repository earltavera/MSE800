import functools

def log_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      
        print(f"Calling function: '{func.__name__}' with args: {args} and kwargs: {kwargs}")
        
        result = func(*args, **kwargs)
        
        print(f"Result of '{func.__name__}': {result}\n")
        return result
    return wrapper

@log_decorator
def add(a, b):
    """Returns the sum of two numbers."""
    return a + b

@log_decorator
def multiply(a, b):
    """Returns the product of two numbers."""
    return a * b

# Testing the functions
if __name__ == "__main__":
    add(5, 10)
    multiply(4, 7)
    add(a=20, b=22)
