# Introduction
import math

# Definng Variables
Set_A = {}
Set_B = {}
Result_Set = {}

# Use input for Sets
Set_A = input("Enter the elemets of set A seperated by space: ").split()
Set_B = input("Enter the elemets of set A seperated by space: ").split()

# Display the sets
print(f"Set A:{Set_A}")
print(f"Set B:{Set_B}")

# User input for operation
print("Enter the operation you want to perfrom on the sets (Union, Intersection, Difference(A-B)): ")
Operation = str(input(">")).lower()

# Calculation
if Operation == "union":
    Result_Set = set(Set_A).union(set(Set_B))
elif Operation == "intersection":
    Result_Set = set(Set_A).intersection(set(Set_B))
elif Operation == "difference":
    Result_Set = set(Set_A).difference(set(Set_B))
else:
    print("No Operation performed")

# Display the results
print(f"Resultant Set: [ {Result_Set} ] for operation {Operation}")
