# Standard
import numpy as np
import time
import pickle # For saving results
import os # For receiving input arguments

def main(num_agents, min_crash, win_reward, p_init, alpha, beta, n_batch, n_steps, n_rec, n_traj, verbose):
    def initialize(p_init):
        # Q-value = [num_agents, 2]
        # 1st coord. = Agent id
        # 2nd coord. = Action
        return np.vstack([1/beta * np.log(p_init), 1/beta * np.log(1-p_init)]).T

    def softmax(Q,beta):
        return np.exp(Q[:,0]*beta)/np.sum(np.exp(Q*beta.reshape([3,1])),1)

    # Initialize Q-values
    Q = initialize(p_init)

    # Outputs
    t_list = np.zeros(int(n_steps/n_batch))              # Step
    p_list = np.zeros([num_agents,int(n_steps/n_traj)]) # Policy trajectory
    r_list = np.zeros([num_agents,int(n_steps/n_batch)]) # Average reward

    # Keep track of sum rewards for calculating average
    r_total = np.zeros(num_agents)
    # Set counter for counting n_steps/n_batch
    counter = 0
    # Keep track of winner of each episode

    # Initialize TD list
    TD = np.zeros([num_agents,2])

    t_start = time.time()
    t1 = time.time()

    for i_ep in range(n_steps):
        # Convert Q-values to policy
        p = softmax(Q, beta) 
        
        # Action selection
        a = 1*(p <= np.random.random(3))
        
        # Reward distribution
        crash = ((num_agents-np.sum(a)) >= min_crash) # If at least min_crash players stay straight (a=0), crash
        allchicken = np.sum(a) == num_agents # Everyone chickens out

        # Crashing gives -2, swerving when at least one other stays gives -1, everyone swerving gives 0, staying when everyone else swerves gives variable reward
        r = (crash*(-2) + (1-crash)*win_reward)*(a==0) + (allchicken*0 + (1-allchicken)*(-1))*(a==1)

        # Keep track of sum for calculating average below
        r_total += r
        
        # TD learning
        for i in range(num_agents):
            TD[i][a[i]] += r[i] - Q[i][a[i]]
            
        # Update every n_batch steps
        if (i_ep+1) % n_batch == 0:
            Q = Q + alpha.reshape([3,1]) * TD/n_batch
            
            TD = np.zeros([num_agents,2]) # Reset TD
            # Record average reward
            t_list[counter] = i_ep
            r_list[:,counter] = r_total/n_batch
            r_total = np.zeros(num_agents)       # Reset sum
            counter += 1
            
        # Save trajectory
        if i_ep % n_traj == 0:
            p_list[:,int(i_ep/n_traj)] = p

        # Time stuff
        if i_ep % n_rec == 0 and i_ep != 0:
            t2 = time.time()
            
            if verbose:
                print('-----------------------------')
                print('Episode ' + str(i_ep))
                print('Runtime for episodes ' + str(i_ep - n_rec) + '-' + str(i_ep) + ': ' + str(t2-t1) + ' s')
                print('-----------------------------')
                
            t1 = t2

    return t_list, p_list, r_list
            
if __name__ == "__main__":
    tstart_total = time.time()

    # Inputs
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        index = int(os.getenv('SLURM_ARRAY_TASK_ID'))
        
        # Read input
        f = open('inputs.txt')
        lines = f.readlines()
        f.close()
        
        # Parse input
        line = lines[index].split('\t')
        
        b1_unique_list = np.array([float(x) for x in line[0].split()])
        b2 = float(line[1])
        b3 = float(line[2])
        p_init = np.array([float(x) for x in line[3].split()])
        
    else:
        # Default inputs
        
        # [1:1:7, 2.0, 2.0]
        b1_unique_list = []
        for i in np.arange(1,8):
            b1_unique_list.append(np.array([i,b2,b3]))
        #b2 = 2.0
        #b3 = 2.0

        p_init = np.array([0.2,0.2,0.2]) # Initial policy (prob. of action 0 = straight)

    # Stag Hunt parameters
    num_agents = 3                   # Number of agents
    win_reward = 3                   # Reward for successful bluff
    min_crash = 2                    # Number of staying straights for crash to occur

    # RL parameters
    alpha = 0.001*np.ones(3)  # alpha parameters
    
    # Simulation parameters
    n_repeat = 10       # Repeat each run (with same parameters) n_repeat times
    n_batch = 100       # Batch size
    n_steps = 10000000    # Training steps
    n_rec = n_steps/10  # For printing purposes
    n_traj = 1000       # Save trajectory every n_traj iterations
    verbose = False      # Display option
    
    # Output lists
    t_list_list = []
    p_list_list = []
    r_list_list = [] # list (per beta) of lists (per step).
    b1_list = []
    
    t1 = time.time()
    for b1 in b1_unique_list:
        beta = np.array([b1,b2,b3])
        for i in range(1,n_repeat+1):
            t_list, p_list, r_list = main(num_agents, min_crash, win_reward, p_init, alpha, beta, n_batch, n_steps, n_rec, n_traj, verbose)
            
            t_list_list.append(t_list)
            p_list_list.append(p_list)
            r_list_list.append(r_list)
            b1_list.append(b1)
            
            t2 = time.time()
            print('Beta = ' + str(beta[0]) + ', run = ' + str(i) + '/' + str(n_repeat) \
                + ': ' + str(t2-t1) + ' s.')
            t1 = t2
        
    # Save files
    data = {'b1_list':b1_list, 'b2':b2, 'b3':b3, 'p_init':p_init, 't_list_list':t_list_list, 'p_list_list':p_list_list, 'r_list_list':r_list_list}
    
    filename = 'Chicken' \
             + '_alpha_' + str(alpha[0]) \
            + '_n_batch_' + str(n_batch) \
            + '_n_steps_' + str(n_steps) \
            + '_p_init_' + str(p_init[0]) + '_' + str(p_init[1]) + '_' + str(p_init[2]) \
            + '_beta_x_' + str(b2) + '_' + str(b3)  \
            + '.pickle'
    
    with open(filename, 'wb') as f:
      pickle.dump(data, f)

    tend_total = time.time()
    
    print('Total runtime: ' + str(tend_total-tstart_total) + ' s')