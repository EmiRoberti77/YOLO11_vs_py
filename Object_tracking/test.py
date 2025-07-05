from collections import defaultdict

grouped = defaultdict(list)
grouped["emi"].append(100)
grouped["rose"].append(99)
grouped["hari"].append(98)
print(grouped)
print(len(grouped))

names = ["emi", "rose", "hari", "nivi", "brijal", "mo"]
scores = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90]

for name, score in zip(names, scores):
  print(f"{name}:{score}")