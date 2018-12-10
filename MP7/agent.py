import utils
import numpy as np

class Agent:
    
    def __init__(self, actions, two_sided = False):
        self._actions = actions     #[-1, 0, 1]
        self._train = True
        self._x_bins = utils.X_BINS
        self._y_bins = utils.Y_BINS
        self._v_x = utils.V_X
        self._v_y = utils.V_Y
        self._paddle_locations = utils.PADDLE_LOCATIONS
        self._num_actions = utils.NUM_ACTIONS
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.two_sided = two_sided
        self.prev_action = 0
        self.prev_state = [0, 0.5, 0, 0, 0.8]
        self.N = utils.create_q_table()
        self.pbounce = 0
        self.first = True
        self.itrs = 0
        self.explore = True
        self.explore_print = True
        self.epsilon = 100
        self.decay = 0.9999
        self.min_epsilon = 0.1
        self.c_val = 50
        self.min_c = 30

    def act(self, state, bounces, done, won):
         #TODO - fill out this function
        self.itrs += 1

        #Discretize the continuous state space
        if(self.prev_state[0] == 0 and self.prev_state[1] == 0.5 and self.prev_state[2] == 0 and self.prev_state[3] == 0 and self.prev_state[4] == 0.8):
            x, y, vx, vy, pad = self.discretize(self.prev_state)
            self.prev_state = [x, y, vx, vy, pad]

        ball_x, ball_y, v_x, v_y, paddle = self.discretize(state)

        if(self._train == True):

            #TODO
            r = np.random.randint(0, 100)

            #From current state s, choose action a based on exploration vs. exploitation policy (implement epsilon-greedy approach)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            gamma = .87

            if(r <= self.epsilon or self.first == True):
                
                if self.N[ball_x][ball_y][v_x][v_y][paddle][0] == 0:
                    action = 0
                elif self.N[ball_x][ball_y][v_x][v_y][paddle][1] == 0:
                    action = 1
                elif self.N[ball_x][ball_y][v_x][v_y][paddle][2] == 0:
                    action = 2
                else:
                    action = np.random.randint(0,2)

            else:
                action = np.argmax(self.Q[ball_x][ball_y][v_x][v_y][paddle])

            self.c_val = max(self.min_c, self.c_val*self.decay)

            alpha = self.c_val / (self.c_val + self.N[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action])

            #Observe reward R(s) and next state s'
            if (done): # R->1
                reward = 10
                self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] -= reward
            elif(won):
                reward = 5
                self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] += reward
            elif (self.pbounce < bounces): # R->-1
                reward = bounces
                self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] = self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] + alpha*(reward - self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] + gamma*self.Q[ball_x][ball_y][v_x][v_y][paddle][action])
            else: # R=>0
                reward = 0
                self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] = self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] + alpha*(reward - self.Q[self.prev_state[0]][self.prev_state[1]][self.prev_state[2]][self.prev_state[3]][self.prev_state[4]][self.prev_action] + gamma*self.Q[ball_x][ball_y][v_x][v_y][paddle][action])

            #Save previous state
            self.prev_state = [ball_x, ball_y, v_x, v_y, paddle]
            self.prev_action = action
            self.N[ball_x][ball_y][v_x][v_y][paddle][action]+=1
            self.pbounce = bounces
            self.first = False

        else:
            action = np.argmax(self.Q[ball_x][ball_y][v_x][v_y][paddle])

        #print(action)
        return self._actions[action]

    def discretize(self, state):
        discrete_ball_x = np.floor(state[0]*12) - 1
        discrete_ball_y = np.floor(state[1]*12) - 1

        if(discrete_ball_x >= 11):
            discrete_ball_x = 11
        if(discrete_ball_y >= 11):
            discrete_ball_y = 11

        if(state[2] >= 0):
            discrete_v_x = 1
        else:
            discrete_v_x = -1

        if(state[3] >= 0.015):
            discrete_v_y = 1
        elif(abs(state[3]) < 0.015):
            discrete_v_y = 0
        else:
            discrete_v_y = -1

        if(state[4] >= 0.8):
            discrete_paddle = 11
        else:
            discrete_paddle = np.floor((12*state[4])/0.8)

        return int(discrete_ball_x), int(discrete_ball_y), int(discrete_v_x), int(discrete_v_y), int(discrete_paddle)

    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    def save_model(self,model_path):
        # At the end of training save the trained model
        utils.save(model_path,self.Q)

    def load_model(self,model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)



