with open('nell_hazy_svo_604m') as f:
    lines = []
    for i in range(200000):
        line = f.readline().strip().split('\t')
        if int(line[3])>500 and line[0] != '##':
            lines += [line]

print(lines)

