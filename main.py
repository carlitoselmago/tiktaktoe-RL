import random
import pickle

class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9
        self.winner = None

    def display(self):
        print("\n")
        for i in range(3):
            print(" " + " | ".join(self.board[i*3:(i+1)*3]))
            if i < 2:
                print("---+---+---")
        print("\n")

    def available_moves(self):
        return [i for i in range(9) if self.board[i] == ' ']

    def make_move(self, index, player):
        if self.board[index] == ' ':
            self.board[index] = player
            self.winner = self.check_winner()
            return True
        return False

    def check_winner(self):
        win_positions = [(0,1,2), (3,4,5), (6,7,8),
                         (0,3,6), (1,4,7), (2,5,8),
                         (0,4,8), (2,4,6)]
        for a,b,c in win_positions:
            if self.board[a] == self.board[b] == self.board[c] != ' ':
                return self.board[a]
        if ' ' not in self.board:
            return 'Draw'
        return None

    def reset(self):
        self.board = [' '] * 9
        self.winner = None

    def get_state(self):
        return ''.join(self.board)

class QLearningAgent:
    def __init__(self, player, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q = {}  # Q-table
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.player = player

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        return random.choice([a for a, q in zip(actions, q_values) if q == max_q])

    def learn(self, state, action, reward, next_state, next_actions):
        max_q_next = max([self.get_q(next_state, a) for a in next_actions], default=0)
        self.q[(state, action)] = (1 - self.alpha) * self.get_q(state, action) + self.alpha * (reward + self.gamma * max_q_next)

def train(agent, episodes=50000):
    opponent = 'O' if agent.player == 'X' else 'X'
    env = TicTacToe()

    for _ in range(episodes):
        env.reset()
        state = env.get_state()

        while True:
            actions = env.available_moves()
            action = agent.choose_action(state, actions)
            env.make_move(action, agent.player)

            next_state = env.get_state()
            winner = env.check_winner()

            if winner:
                if winner == agent.player:
                    reward = 1
                elif winner == 'Draw':
                    reward = 0.5
                else:
                    reward = -1
                agent.learn(state, action, reward, next_state, [])
                break
            else:
                opponent_move = random.choice(env.available_moves())
                env.make_move(opponent_move, opponent)
                next_state_2 = env.get_state()
                winner = env.check_winner()

                if winner:
                    if winner == agent.player:
                        reward = 1
                    elif winner == 'Draw':
                        reward = 0.5
                    else:
                        reward = -1
                    agent.learn(state, action, reward, next_state_2, [])
                    break
                else:
                    agent.learn(state, action, 0, next_state_2, env.available_moves())
                    state = next_state_2

def play(agent):
    env = TicTacToe()
    human = 'O' if agent.player == 'X' else 'X'
    env.display()

    while True:
        if agent.player == 'X':
            state = env.get_state()
            move = agent.choose_action(state, env.available_moves())
            env.make_move(move, agent.player)
            env.display()
            if env.winner:
                print("Result:", env.winner)
                break
            move = int(input("Your move (0-8): "))
            while not env.make_move(move, human):
                move = int(input("Invalid. Try again: "))
        else:
            move = int(input("Your move (0-8): "))
            while not env.make_move(move, human):
                move = int(input("Invalid. Try again: "))
            env.display()
            if env.winner:
                print("Result:", env.winner)
                break
            state = env.get_state()
            move = agent.choose_action(state, env.available_moves())
            env.make_move(move, agent.player)
            env.display()
            if env.winner:
                print("Result:", env.winner)
                break

# Main flow
agent = QLearningAgent(player='X')
train(agent)
play(agent)
