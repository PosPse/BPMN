with open('/home/rjxy1406/user/btr/graphSAGE2/data/2_data1.txt', 'r') as file:
    data = file.readlines()
    print(len(data))
    stence = []
    for line in data:
        word = line.split(' ')[0]
        stence.append(word)
    stence = ' '.join(stence)
    print(stence)
    