import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

fin = open('res.txt', 'r')

res = []
for l in fin:
    t = l.split()
    res.append(float(t[1]))

plt.plot(range(len(res)), res)

plt.savefig('res.png')
