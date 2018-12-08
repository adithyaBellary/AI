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
        self.prev_state = [0, 0, 0, 0, 0]

    

    def act(self, state, bounces, done, won):
         #TODO - fill out this function
        if(self._train == True):
            #Fill out Q Table according to update equation 
                #Define reward function as #bounces (weighted using average) + done + previous state 
            #(X_BINS,Y_BINS,V_X,V_Y,PADDLE_LOCATIONS,NUM_ACTIONS)
            # [self.ball_x, self.ball_y, self.velocity_x, self.velocity_y,self.paddle_y,self.opponent_y,]
            
            #Discretize the continuous state space
            ball_x, ball_y, v_x, v_y, paddle = self.discretize(state)

            #TODO
            #From current state s, choose action a based on exploration vs. exploitation policy (implement epsilon-greedy approach)
            epsilon = 0.1

            if(self.Q[ball_x][ball_y][v_x][v_y][paddle].any() == 0):
                action = self._actions[2]
                self.prev_action = action

            #Observe reward R(s) and next state s'
            


            #Update Q values for previous state s
            # self.Q[ball_x][ball_y][v_x][v_y][paddle][prev_action] = self.Q[] + learning_rate*(reward() - self.Q[] + gamma*)
            

            #Save previous state
            self.prev_state = [ball_x, ball_y, v_x, v_y, paddle]
            self.prev_action = action

        else:
            action = self._actions[2]

        

        return action

    def discretize(self, state):
        discrete_ball_x = np.floor(state[0]*12) - 1
        discrete_ball_y = np.floor(state[1]*12) - 1

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

        if(state[4] == 0.8):
            discrete_paddle = 11
        else:
            discrete_paddle = np.floor((12*state[4])/1-0.2)

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



