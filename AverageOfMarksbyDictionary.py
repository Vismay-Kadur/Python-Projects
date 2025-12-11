A = {}
n = int(input())
if 2 <= n <= 10:
    for i in range(n):
        input_values = (input()).split()
        name = input_values[0]
        marks = []
        for i in range(len(input_values)):
            if i == 0:
                continue
            if 0 <= float(input_values[i]) <= 100:
                marks.append(float(input_values[i]))
        A[name] = marks
query = input()
if query in A.keys():
    marks = A[query]
    total_marks = sum(marks)
    average_marks = total_marks/len(marks)
    print(f"{average_marks:.2f}")
