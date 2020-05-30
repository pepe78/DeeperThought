import matplotlib.pyplot as plt
import sys

filename = 'debug.csv'

t=[]
e1=[]
a1=[]
e2=[]
a2=[]
with open(filename, 'rt') as f:
	while True:
		content = f.readline()
		if len(content) == 0:
			break
		field = content.replace('\n','').split(',')
		
		if len(field)!=7:
			break
		
		t.append(int(field[0]))
		e1.append(float(field[1]))
		a1.append(float(field[2]))
		e2.append(float(field[4]))
		a2.append(float(field[5]))

print('Max train accuracy:', max(a1))
print('Max test accuracy:', max(a2))

plt.subplot(211)
plt.plot(t,a1,'b')
plt.plot(t,a2,'r')

plt.subplot(212)
plt.plot(t,e1,'b')
plt.plot(t,e2,'r')

plt.show()

#plt.imshow(m)
#plt.show()

