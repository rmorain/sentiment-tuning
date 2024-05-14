import csv

with open("test_results/k1nb1tli_perspective.csv", newline="") as my_file:
    reader = csv.reader(my_file)
    total_toxic = 0
    total_rows = 0
    for row in reader:
        total_rows += 1
        if float(row[-1]) >= 0.5:
            total_toxic += 1
    print(total_toxic / total_rows)
