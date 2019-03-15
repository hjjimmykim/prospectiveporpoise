
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')# Data processing --------------------------------------------------------------------------------

# Turn b_list into a string
b_str_list = []
for b in b_list:
  b_str_list.append(str('%.1f' % b))
  
# Extract average reward of last 1000 turns
rb1 = []
rb2 = []
rb3 = []
for r_list in r_list_list:
  rb1.append(np.mean(r_list[0][-1000:]))
  rb2.append(np.mean(r_list[1][-1000:]))
  rb3.append(np.mean(r_list[2][-1000:]))
  
# Reward plot ------------------------------------------------------------------
import matplotlib.pyplot as plt
plt.figure()
plt.plot(r_list_list[-1][0],"r-", label = 'Agent 1')
plt.plot(r_list_list[-1][1],"b-", label = 'Agent 2')
plt.plot(r_list_list[-1][2],"g-", label = 'Agent 3')
plt.legend(loc = 'upper right')
plt.xlabel('Iteration')
plt.ylabel('Reward')

plt.show()

# Peak plot ----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
plt.plot(b_list, rb1,"r-", label = 'Agent 1')
plt.plot(b_list, rb2,"b-", label = 'Agent 2')
plt.plot(b_list, rb3,"g-", label = 'Agent 3')
plt.legend(loc = 'upper left')
plt.xlabel('Beta1')
plt.ylabel('Reward')

locs, labels = plt.xticks()

plt.show()

# Trajectory ---------------------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import axes3d
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')
color=iter(cm.jet(np.linspace(0,1,len(b_list))))

norm = mpl.colors.Normalize(vmin=0,vmax=1)
sm = plt.cm.ScalarMappable(cmap=cm.jet, norm=norm)
sm.set_array([])

for i in range(len(p_list_list)):
  c=next(color)
  ax.plot(p_list_list[i][0], p_list_list[i][1], p_list_list[i][2],c=c)
  ax.scatter(p_list_list[i][0][0], p_list_list[i][1][0], p_list_list[i][2][0],c='k',s=100) #Initial
  ax.scatter(p_list_list[i][0][-1], p_list_list[i][1][-1], p_list_list[i][2][-1],c=c, s=50) #End

cbar = plt.colorbar(sm, ticks = np.linspace(0.1,1.0,len(b_list)),
             boundaries=np.linspace(0.0,1.1,len(b_list)))
cbar.set_ticklabels(b_str_list)
cbar.set_label('Beta2', rotation=270)

ax.set_xlabel('$p_1$',fontsize=20)
ax.set_ylabel('$p_2$',fontsize=20)
ax.set_zlabel('$p_3$',fontsize=20)

plt.show()