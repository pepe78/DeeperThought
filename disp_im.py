import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]
with open(filename, 'rt') as f:
	while True:
		content = f.readline()
		if len(content) == 0:
			break
		field = content.replace('\n','').split(',')

		m = []
		for i in range(28):
			tmp = ''
			t = []
			for j in range(28):
				t.append(float(field[10+i*28+j]))
				if float(field[10+i*28+j]) < 0.5:
					tmp += ' '
				else:
					tmp += 'X'
			m.append(t)	
			print(tmp)
		print('-------------------------------------------------------')
		plt.imshow(m)
		plt.show()

