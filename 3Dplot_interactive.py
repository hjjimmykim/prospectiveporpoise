import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import pickle

# GUI
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw() # Hide root window
filename = askopenfilename()

data = pickle.load(open(filename,'rb'))

# Extract data
b1_list = data['b1_list']
p_list_list = data['p_list_list']
r_list_list = data['r_list_list']
p_init = data['p_init']
b2 = data['b2']
b3 = data['b3']

# Turn b1_list into a string

b1_list_unique = np.unique(b1_list)
b1_str_list = []
for b1 in b1_list_unique:
    b1_str_list.append(str('%.1f' % b1))
    
n_repeat = int(len(b1_list)/len(b1_list_unique))

# Trajectory
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
color=iter(cm.jet(np.linspace(0,1,len(b1_list_unique))))

norm = mpl.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array([])

c = next(color)
for i in range(len(p_list_list)):
    if i!=0 and i%n_repeat == 0:
        c=next(color)
    ax.plot(p_list_list[i][0], p_list_list[i][1], p_list_list[i][2],c=c)
    ax.scatter(p_list_list[i][0][0], p_list_list[i][1][0], p_list_list[i][2][0],c='k',s=100) #Initial
    ax.scatter(p_list_list[i][0][-1], p_list_list[i][1][-1], p_list_list[i][2][-1],c=c, s=50) #End

cbar = plt.colorbar(sm, ticks = np.linspace(0.1,1.0,len(b1_list_unique)),
             boundaries=np.linspace(0.0,1.1,len(b1_list_unique)))
cbar.set_ticklabels(b1_str_list)
cbar.set_label(r'$\beta_1$', rotation=270)

ax.set_xlabel('$p_1$',fontsize=20)
ax.set_ylabel('$p_2$',fontsize=20)
ax.set_zlabel('$p_3$',fontsize=20)

plt.title(r'$\beta_2$ = ' + str(b2) + r', $\beta_3$ = ' + str(b3) + r', $p_{init}$ = ' + str(p_init))
ax.set_xlim3d([0,1])
ax.set_ylim3d([0,1])
ax.set_zlim3d([0,1])

filename = 'beta2=' + str(b2) + '_beta3=' + str(b3) + '_p_init=' + str(p_init) + '.png'
plt.savefig('princeton_plots/' + filename)
#plt.show()