# Standard
import numpy as np
import time
import pickle # For saving results

def main(num_agents, min_hunt, hunt_reward, p_init, alpha, beta, n_batch, n_steps, n_rec, n_traj, verbose):
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
        hunt_successful = (np.sum(a) >= min_hunt) # Hunt successful only if at least min_hunt hunters

        # Foraging always gives 1, failed hunt gives 0, successful hunt is variable
        r = 1*(a==0) + hunt_successful*hunt_reward*(a==1)
        
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
            
if __name__=='__main__':
    # Stag Hunt parameters
    num_agents = 3                   # Number of agents
    hunt_reward = 4                  # Reward for successful hunt
    min_hunt = 3                     # Number of hunters required for hunt to be successful

    # RL parameters
    p_init = np.array([0.7,0.7,0.7]) # Initial policy (prob. of action 0 = forage)
    alpha = 0.01*np.ones(3)  # alpha parameters
    
    # Simulation parameters
    n_batch = 100       # Batch size
    n_steps = 10000000    # Training steps
    n_rec = n_steps/10  # For printing purposes
    n_traj = 1000       # Save trajectory every n_traj iterations
    verbose = False      # Display option
    
    # Beta list
    b_list = 0.1*np.arange(6,14) # 0.1 - 1.9
    t_list_list = []
    p_list_list = []
    r_list_list = [] # list (per beta) of lists (per step).
    
    t1 = time.time()
    for b in b_list:
        beta = np.array([1.0,b,1.0])

        t_list, p_list, r_list = main(num_agents, min_hunt, hunt_reward, p_init, alpha, beta, n_batch, n_steps, n_rec, n_traj, verbose)
        
        t_list_list.append(t_list)
        p_list_list.append(p_list)
        r_list_list.append(r_list)
        
        t2 = time.time()
        print('Beta = ' + str(b) + ': ' + str(t2-t1) + ' s.')
        t1 = t2
        
    # Save files
    data = {'b_list':b_list, 't_list_list':t_list_list, 'p_list_list':p_list_list, 'r_list_list':r_list_list}
    
    filename = 'StagHunt' + str(num_agents) + 'P' + '_huntreward_' +str(hunt_reward) \ 
             + '_alpha_' + str(alpha[0]) + '_beta_0.6_1.3_0.1' + '_n_batch_' + str(n_batch) + \
            + '_n_steps_' + str(n_steps) + '.pickle'
    
    with open(filename, 'wb') as f:
      pickle.dump(data, f)

