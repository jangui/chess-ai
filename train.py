import random
import chess

from Agent import Agent
from Env import Env

def main():
    env = Env()
    agent = Agent(env.state.shape, env.output_dimensions, model_path="./model3999")

    episodes = 100000
    render_period = 50
    render = True

    episode_rewards = []


    for episode in range(episodes):
        if render and (episode % render_period == 0):
            print("EPISODE", episode, "episilon:", agent.epsilon)
        episode_reward = 0
        reward = 0
        state = env.reset()

        while not env.done:
            if agent.epsilon > random.random():
                #preform random legal move
                #while epsilon is high more random moves will be taken
                legal_moves = list(env.board.legal_moves)
                if len(legal_moves) == 0:
                    print("NO LEGAL MOVES")
                move = random.randint(0, len(legal_moves)-1)
                action = (legal_moves[move].from_square, legal_moves[move].to_square)
            else:
                #preform action based off network prediction
                #as episilon decays this will be the usual option
                action = agent.get_action(state)

            new_state, reward, done = env.step(action)


            # train
            env_info = (state, action, new_state, reward, done)
            agent.train(env_info)

            # render
            if render and (episode % render_period == 0):
                print(env.board,"\n","-"*16)

            # opponent moves
            if not done:
                legal_moves = list(env.board.legal_moves)
                move = random.randint(0, len(legal_moves)-1)
                action = (legal_moves[move].from_square, legal_moves[move].to_square)
                new_state, more_reward, done = env.step(action)
                reward += more_reward

                # render
                if render and (episode % render_period == 0):
                    print(env.board,"\n","-"*16)

            state = new_state
            episode_reward += reward

            #decay epsilon
            if agent.epsilon > agent.min_epsilon:
                agent.epsilon *= agent.epsilon_decay
                agent.epsilon = max(agent.epsilon, agent.min_epsilon)
        episode_rewards.append(episode_reward)
        if env.board.fullmove_number >= 20 and agent.epsilon < 0.7:
            agent.model.save(f"./model{episode}move{env.board.fullmove_number}")
        if episode % 1000 == 0:
            agent.model.save(f"./model{episode}autosave")


if __name__ == "__main__":
    main()
