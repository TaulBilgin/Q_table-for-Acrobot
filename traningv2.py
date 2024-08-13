import gymnasium as gym
import numpy as np
import pickle
import random

def optim(memory, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2, q_table):
    reverse_memory = memory[::-1]
    for now_state, new_state, reward, action, done in reverse_memory:
        q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor = q_table_line(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
        new_q_Cosine_of_theta1, new_q_Sine_of_theta1, new_q_Cosine_of_theta2, new_q_Sine_of_theta2, new_q_Angular_velocity_of_theta1, new_q_Angular_velocity_of_theta2 = q_table_line(new_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2) 
        if done:
            q_table[q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor, action] = 10    
        q_table[q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor, action] = reward + 0.99 * np.max(q_table[new_q_Cosine_of_theta1, new_q_Sine_of_theta1, new_q_Cosine_of_theta2, new_q_Sine_of_theta2, new_q_Angular_velocity_of_theta1, new_q_Angular_velocity_of_theta2])

def q_table_line(state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    Cosine_of_theta1_tensor = np.digitize((state[0]), Cosine_of_theta1)
    Sine_of_theta1_tensor = np.digitize((state[1]), Sine_of_theta1)
    Cosine_of_theta2_tensor = np.digitize((state[2]), Cosine_of_theta2)
    Sine_of_theta2_tensor = np.digitize((state[3]), Sine_of_theta2)
    Angular_velocity_of_theta1_tensor = np.digitize((state[4]), Angular_velocity_of_theta1)
    Angular_velocity_of_theta2_tensor = np.digitize((state[5]), Angular_velocity_of_theta2)

    return Cosine_of_theta1_tensor, Sine_of_theta1_tensor, Cosine_of_theta2_tensor, Sine_of_theta2_tensor, Angular_velocity_of_theta1_tensor, Angular_velocity_of_theta2_tensor
    
def test_for_save(q_table, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2):
    env2 = gym.make('Acrobot-v1')
    totel_step = 0

    for i in range(10):
        now_state = env2.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0

        # Play one episode
        while not done and step < 500 :
            q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor = q_table_line(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
            action = np.argmax(q_table[q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor, :])
            step += 1
            totel_step += 1
            # Take action and observe result
            new_state, reward, done, truncated, _ = env2.step(action)
            
            now_state = new_state
    env2.close()
    return (totel_step / 10)

def train():
    env = gym.make('Acrobot-v1')

    Cosine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
    Sine_of_theta1 = np.linspace(-1, 1, 20) # Between -1 and 1
    Cosine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
    Sine_of_theta2 = np.linspace(-1, 1, 20) # Between -1 and 1
    Angular_velocity_of_theta1 = np.linspace(-12.56, 12.56, 20) # betwen -12.56 and 12.56
    Angular_velocity_of_theta2 = np.linspace(-12.56, 12.56, 20) # betwen -12.56 and 12.56

    """
    The code doesn't work like : Cart_Velocity = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 100)
    you must put manual value of env.observation_space.low[0] and env.observation_space.high[0]
    """
    q_table = np.zeros((len(Cosine_of_theta1)+1, len(Sine_of_theta1)+1, len(Cosine_of_theta2)+1, len(Sine_of_theta2)+1, len(Angular_velocity_of_theta1)+1, len(Angular_velocity_of_theta2)+1, env.action_space.n))

    gamma = 0.99
    run = 0
    save_count = 0
    past_best_save = 350
    memory = []
    choice_list = ['x'] * 20 + ['y'] * 80
    real_run = 0

    for i in range(10000):
        
        now_state = env.reset()[0]  # Reset environment and get initial state
        done = False  # Flag to check if the episode is finished
        step = 0
        # Increment the episode counte

        # Play one episode
        
        while not done and step < 500 :
            """if random.choice(choice_list) == "x":
                action = env.action_space.sample()  # Random action
            else:"""
            q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor = q_table_line(now_state, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
            action = np.argmax(q_table[q_Cosine_of_theta1_tensor, q_Sine_of_theta1_tensor, q_Cosine_of_theta2_tensor, q_Sine_of_theta2_tensor, q_Angular_velocity_of_theta1_tensor, q_Angular_velocity_of_theta2_tensor, :])
            step += 1
            
            # Take action and observe result
            new_state, reward, done, truncated, _ = env.step(action)

            # Update Q-value using Bellman equation
            
            memory.append((now_state, new_state, reward, action, done))
            now_state = new_state

        optim(memory, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2, q_table)
        memory = []
        if step < 500:
            real_run += 1
            """if (real_run % 100) == 0:
                if all(choice == 'y' for choice in choice_list):
                    return 0

                choice_list.remove("x")
                choice_list.append("y")"""

        run += 1
        print(step, run, real_run)
        if step < int(past_best_save):
            env.close()
            best_save = test_for_save(q_table, Cosine_of_theta1, Sine_of_theta1, Cosine_of_theta2, Sine_of_theta2, Angular_velocity_of_theta1, Angular_velocity_of_theta2)
            if best_save < past_best_save:
                save_count += 1
                past_best_save = best_save
                f = open('Acrobot.pkl','wb')
                print("----------------------")
                pickle.dump(q_table, f)
                f.close()
                
        
    
    return save_count
while True:
    save_count = train()
    if save_count != 0:
        break
