import numpy as np
# Generate inputs

filename = 'inputs.txt'

# Real
b1_list_list = [np.arange(2.6,3.6,0.1),np.arange(0.6,1.6,0.1),np.arange(0.6,2.6,0.1),np.arange(0.6,2.6,0.1)]
# b1_list = np.array([3,1,2,2])
b2_list = np.array([1,1,2,1])
b3_list = np.array([1,3,2,2])

p_init_list = [[0.2,0.2,0.2],[0.5,0.5,0.5],[0.8,0.8,0.8]]

# Just for test
'''
b1_list_list = [np.arange(1,8)]
b2_list = np.array([2.0])
b3_list = np.array([2.0])
p_init_list = [[0.2,0.2,0.2]]
'''

nentries = len(b1_list_list) * len(p_init_list) # Number of entries

f = open(filename, 'w')

f.write(str(nentries) + '\n') # Record number of entries
for i1 in range(len(p_init_list)):
    p_init = p_init_list[i1]
    for i2 in range(len(b1_list_list)):
        b1_list = b1_list_list[i2]
        b2 = b2_list[i2]
        b3 = b3_list[i2]
        

        # Write b1_list
        for b1 in b1_list:
            f.write('%f ' % (b1))
        f.write('\t')
        # Write b2 & b3
        f.write('%f \t %f \t' % (b2,b3))
        # Write p_init
        f.write('%f %f %f \n' % (p_init[0],p_init[1],p_init[2]))

f.close()