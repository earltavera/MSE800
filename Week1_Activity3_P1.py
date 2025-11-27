#Fibonacci Series Factorial Calculator
#Author: Earl Tavera
#Description: This program will ask for a positive number and will take as the value of N before generating the Fibonacci series and compute the factorial of N.

def fibonacci_series(N):
    """
    Generates the first N terms of the Fibonacci series (0, 1, 1, 2, 3, 5...).
    """
    if N <= 0:
        return []
    elif N == 1:
        return [0]
    elif N == 2:
        return [0, 1]
    else:
        # Start with the initial two terms
        series = [0, 1]
        # Calculate subsequent terms
        for i in range(2, N):
            next_term = series[i-1] + series[i-2]
            series.append(next_term)
        return series

def factorial(N):
    """
    Computes the factorial of N (N! = 1 * 2 * 3 * ... * N).
    """
    if N < 0:
        return "Factorial is not defined for negative numbers."
    elif N == 0:
        return 1
    else:
        result = 1
        # Multiply result by every integer from 1 up to N
        for i in range(1, N + 1):
            result *= i
        return result

# --- Main execution block with User Input ---

def N_from_user():
    """Prompts the user for a non-negative integer N."""
    while True:
        try:
            user_input = input("Enter a positive integer (N) to calculate: ")
            N = int(user_input)
            
            if N < 0:
                print("Input must be a positve integer (0, 1, 2, ...).")
                continue # Go back to the start of the loop
            return N # Return N if it's a valid positve integer
        except ValueError:
            print("Invalid input. Please enter a whole number.")

# Getting the value of N
N = N_from_user()
        
# 1. Generate the Fibonacci series
fib_result = fibonacci_series(N)
print(f"\n--- Results for N = {N} ---")
print(f"Fibonacci Series (first {N} terms): {fib_result}")

# 2. Compute the factorial of N
fact_result = factorial(N)
print(f"Factorial of {N} ({N}!): {fact_result}")
