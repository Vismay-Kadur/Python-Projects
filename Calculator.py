# Introduction
import math
print("Hello World ✋🏻")
print("-"*172)
print("Calculator")
print("-"*172)
print("Addition:+")
print("Substraction:-")
print("Multiplication:*")
print("Divition:/")
print("Integer Division://")
print("Percentage:%")
print("Power:^")
print("logarithm (Value of c does not matter):log")
print("Logarithm base e (value of c does not matter):ln")
print("e to the power:e")
print("-"*172)

# User input & calculation
command = ""
while True:
    print("Enter Values")
    Number_1 = float(input("Number_1= "))
    Operator = str(input("Operator= "))
    Number_2 = float(input("Number_2= "))
    Result = float(0)
    if Operator == "+":
        Result = Number_1 + Number_2
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "-":
        Result = Number_1-Number_2
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "*":
        Result = Number_1*Number_2
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "/":
        Result = Number_1/Number_2
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "//":
        Result = Number_1//Number_2
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "%":
        Result = (Number_1/Number_2)*100
        print("Result =", Result, "%")
        print("-"*172)
    elif Operator == "^":
        Result = math.pow(Number_1, Number_2)
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "log":
        Result = math.log(Number_1, Number_2)
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "ln":
        Result = math.log(Number_1, 2.71828)
        print("Result = ", Result)
        print("-"*172)
    elif Operator == "e":
        Result = math.pow(2.71828, Number_1)
        print("Result = ", Result)
        print("-"*172)
    else:
        print("Error")
        print("-"*172)

    # User input to continue or quit calculator
    command = input(">")
    if command.lower() == "quit":
        break
