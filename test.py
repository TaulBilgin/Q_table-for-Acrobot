import gymnasium as gym
import numpy as np
import pickle

# select which input on which line
def q_table_line(state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    Cosine_of_theta1_tensor = np.digitize((state[0]), Cosine_of_theta1)
    Sine_of_theta1_tensor = np.digitize((state[1]), Sine_of_theta1)
    Cosine_of_theta2_tensor = np.digitize((state[2]), Cosine_of_theta2)
    Sine_of_theta2_tensor = np.digitize((state[3]), Sine_of_theta2)
    Angular_velocity_of_theta1_tensor = np.digitize((state[4]), Angular_velocity_of_theta1)
    Angular_velocity_of_theta2_tensor = np.digitize((state[5]), Angular_velocity_of_theta2)

    return Cosine_of_theta1_tensor, Sine_of_theta1_tensor, Cosine_of_theta2_tensor, Sine_of_theta2_tensor, Angular_velocity_of_theta1_tensor, Angular_velocity_of_theta2_tensor

# load the saved Q_tanble
f = open('Acrobot.pkl', 'rb')
q_table = pickle.load(f)
f.close()

env = gym.make('Acrobot-v1', render_mode="human")

# divide the input
Cosine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
Sine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
Cosine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
Sine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
Angular_velocity_of_theta1 = np.linspace(-12.56, 12.56, 20) # betwen -12.56 and 12.56
Angular_velocity_of_theta2 = np.linspace(-12.56, 12.56, 20) # betwen -12.56 and 12.56

for i in range(10):
    now_state = env.reset()[0]  # Reset environment and get initial state
    done = False  # Flag to check if the episode is finished
    step = 0
    # Play one episode
    while not done and step < 10000 :
        q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor = q_table_line(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
        action = np.argmax(q_table[q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor, :])
        step += 1
        # Take action and observe result
        new_state, reward, done, truncated, _ = env.step(action)
        
        now_state = new_state

    print(step)
