import math

#Fibonacci Series Factorial Calculator
#Author: Earl Tavera
#Description: This program will ask for a positive number and will take as the value of N before generating the Fibonacci series and compute the factorial of N.

## --- Calculation Functions ---

def fibonacci_series(N):
    """
    Generates the first N terms of the Fibonacci series (0, 1, 1, 2, 3, 5...).
    Uses an efficient iterative method.
    """
    if N <= 0:
        return []
    elif N == 1:
        return [0]
    
    series = [0, 1]
    # Calculate subsequent terms up to N
    for _ in range(2, N):
        # Next term is the sum of the last two
        series.append(series[-1] + series[-2])
    return series

def factorial(N):
    """
    Computes the factorial of N (N! = 1 * 2 * 3 * ... * N) 
    using the simplified math.factorial() function.
    """
    if N < 0:
        return "Factorial is not defined for negative numbers."
    
    # SIMPLIFICATION: Replace the entire loop with math.factorial
    return math.factorial(N)


## --- User Input and Execution ---

def N_from_user():
    """Prompts the user for a non-negative integer N with validation."""
    while True:
        try:
            user_input = input("Enter a positive integer (N) to calculate: ")
            N = int(user_input)
            
            if N < 0:
                print("Input must be a non-negative integer (0, 1, 2, ...).")
                continue
            return N
        except ValueError:
            print("Invalid input. Please enter a whole number.")

# Get N from user
N = N_from_user()
        
# Generate and display results
fib_result = fibonacci_series(N)
fact_result = factorial(N)

print(f"\n--- Results for N = {N} ---")
print(f"Fibonacci Series (first {N} terms): {fib_result}")
print(f"Factorial of {N} ({N}!): {fact_result}")
