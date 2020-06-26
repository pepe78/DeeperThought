import matplotlib.pyplot as plt
import sys

filename = 'debug.csv'

def addsubgraph(which, t, q1, q2, lab):
	plt.subplot(which)
	plt.plot(t,q2,'r', label='test')
	plt.plot(t,q1,'b', label='train')
	plt.xlabel('epoch')
	plt.ylabel(lab)
	plt.grid(color='k', linestyle=':', linewidth=1)
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
		   ncol=2, mode="expand", borderaxespad=0.)

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

plt.figure(figsize=(20,15))

addsubgraph(221,t,a1,a2,'accuracy')
addsubgraph(223,t,e1,e2,'error')
addsubgraph(222,t[-len(t)//2:],a1[-len(t)//2:],a2[-len(t)//2:],'accuracy')
addsubgraph(224,t[-len(t)//2:],e1[-len(t)//2:],e2[-len(t)//2:],'error')
           
plt.show()

