import gym
import numpy as np
import random

ispis = False

'''
A passenger waiting for a yellow taxi is marked in blue.
After the taxi picks up the passenger, he must bring him to the "Drop off" point (marked in pink).
Taxi has a maximum number of steps that can be taken per episode. (Fuel)
Defining awards:
* +20 points for each successful "Drop off"
* -1 for each step
* -10 unauthorized "Pickup" or "Drop off" passengers
'''


def next_action(action):
    check_action = ''
    if action == 0:
        check_action = 'South'
    if action == 1:
        check_action = 'North'
    if action == 2:
        check_action = 'East'
    if action == 3:
        check_action = 'West'
    if action == 4:
        check_action = 'Pickup'
    if action == 5:
        check_action = 'Drop Off'

    return check_action


def qlearning_implementation(max_episodes, max_steps, map_table, learning_coefficient, disc_rate_gamma, min_epsilon,
                             max_epsilon,
                             decay_rate, epsilon, env):
    for episode in range(max_episodes):
        current_state = env.reset()

        print('Episode: ', episode)
        print('_____________________')

        '''
        In this part of the code, either a random action or an action from the Q table is selected
        '''
        random_action_number = random.uniform(0, 1)

        for step in range(max_steps):

            if random_action_number > epsilon:
                action = np.argmax(map_table[current_state, :])
            else:
                action = env.action_space.sample()

            '''
            The maximum Q value for the action corresponding to "next_state" is calculated, and then the Q value is updated:
             Q(s,a):= Q(s,a) + lr [R(s,a) + disc_rate_gamma * max Q(s', a') - Q(s, a)]
            '''
            new_state, award, end_info, info = env.step(action)

            '''
            The maximum Q value for the action corresponding to "next_state" is calculated, and then the Q value is updated:
             Q(s,a):= Q(s,a) + lr [R(s,a) + disc_rate_gamma * max Q(s', a') - Q(s, a)]
            '''
            map_table[current_state, action] = map_table[current_state, action] + learning_coefficient * (
                    award + disc_rate_gamma * np.max(map_table
                                                     [new_state, :]) - map_table[current_state, action])
            current_state = new_state

            next_action_check = next_action(action)

            print('Learning coefficient: ', learning_coefficient)
            print('Discount rate: ', disc_rate_gamma)
            print('Award: ', award)
            print('State: ', new_state, ', Action: ', action, ' - ', next_action_check)
            print()

            if end_info:
                break

        print('Epsilon: ', epsilon)
        print('Random uniform: ', random_action_number)
        print()

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)


def learning(max_episodes, max_steps, map_table, all_awards, env):
    env.reset()
    env.render()
    rewards = []

    for episode in range(max_episodes):
        current_state = env.reset()

        total_rewards = 0

        print("***************************")
        print("Episode: ", episode)

        for step in range(max_steps):
            env.render()
            action = np.argmax(map_table[current_state, :])
            new_state, reward, end_info, info = env.step(action)
            total_rewards += reward

            if end_info:
                rewards.append(total_rewards)
                print("Score: ", total_rewards)
                all_awards.append(int(total_rewards))
                break
            current_state = new_state
    env.close()
    print()

    print("Result: " + str(sum(rewards) / max_episodes))
    return all_awards


def main():
    env = gym.make("Taxi-v3", render_mode='human')
    max_actions = 6
    '''
    All allowed actions in the environment - Up, down, left, right, pick up, leave
    '''
    print("Action size ", max_actions)

    '''
    500 possible states - 25 squares (5*5), 5 locations for passengers (4 starting locations and taxi), and 4 destinations.
    25 * 5 * 4 = 500
    '''
    max_states = 500
    print("Current state size ", max_states)

    '''
     Q-Learning boards
     The goal is to learn a rule that will tell the agent what action should be taken for each possible state.
     Q-table stores results for each state-action pair.
     To begin with, the entire table is set to "0" and during "exploration" the values in the table are updated.
    
     Exploration - Choosing random actions
     Exploitation - Selecting actions based on already learned values from the Q table
    
     As in reinforcement learning, the information according to which the program will work does not exist
     Information is collected on how the program works and accordingly to make the best decisions for the future.
    '''
    map_table = np.zeros((max_states, max_actions))
    print(map_table)

    max_episodes = 50000
    max_test_episodes = 100
    max_steps = 99
    learning_coefficient = 0.7
    disc_rate_gamma = 0.618

    '''
     The action selected in the training is either the action with the highest q-value or the action selected by random action.
     The selection is based on the epsilon value.
     We use epsilon to prevent the action from always taking the same route
    '''
    epsilon = 1.0
    max_epsilon = 1.0
    min_epsilon = 0.01

    decay_rate = 0.01
    all_rewards = []

    qlearning_implementation(max_episodes, max_steps, map_table, learning_coefficient, disc_rate_gamma,
                             min_epsilon, max_epsilon,
                             decay_rate, epsilon, env)

    print(map_table)
    all_rewards = learning(max_test_episodes, max_steps, map_table, all_rewards, env)

    print('All rewards: ', all_rewards)

    rewards_sorted_list = []

    while all_rewards:
        minimum = all_rewards[0]
        for x in all_rewards:
            if x < minimum:
                minimum = x
        rewards_sorted_list.append(minimum)
        all_rewards.remove(minimum)

    rewards_sorted_list = list(dict.fromkeys(rewards_sorted_list))

    print('All rewards sorted: ', rewards_sorted_list)


if __name__ == "__main__":
    main()
