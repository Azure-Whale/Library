import numpy as np
a=[1,2]
b=[3,4]
combine=[]
combine.append([a[0],b[0]])
combine.append([a[1],b[1]])
print(combine)
combine=np.array(combine)
print(np.mean(combine))
x=input()
a = ['', 1, 2, 3, 4, 5]
for i in range(5, -1, -1):  # for
    print(i * '*', end='')
    for i in range(0, 6 - i):
        print(a[i], end='')
    print('')   # the default end of print() is \n, so some modifications are needed

string='12345'
for i in range(0,5):
    print("{:*>5}".format(string[0:i+1]))      # Left Align