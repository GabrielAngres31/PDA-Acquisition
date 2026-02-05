import re

with open("C:/Users/Gabriel/Documents/console_history_forensic.txt") as file:
    counter = 0
    prog_run = 0
    programs = set()
    for line in file:
        counter += 1
        if ".py" in line.lower():
            prog_run += 1
            # match = re.findall(r'/(.*?)\.py', line.lower())
            match = re.search(r"\\([^\\]+)\.py", line.lower())
            if match:
                programs.add(match[0])
print(programs)