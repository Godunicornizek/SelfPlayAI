#!/usr/bin/env python
# coding: utf-8

# # New Code for importing TicTacToe class

# In[ ]:


# Download the raw TicTacToe.py from GitHub
get_ipython().system('curl -o TicTacToe.py https://raw.githubusercontent.com/Godunicornizek/SelfPlayAI/main/TicTacToe.py')

# Reload the module in Colab
import importlib
import TicTacToe
importlib.reload(TicTacToe)

from TicTacToe import TicTacToe


# # Old code for cloning into Git

# In[ ]:


get_ipython().system('git clone https://github.com/Godunicornizek/SelfPlayAI.git')


# In[ ]:


get_ipython().system('git pull origin main')


# In[ ]:


get_ipython().system('git reset --hard')
get_ipython().system('git pull origin main')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/SelfPlayAI')
get_ipython().system('ls')


# In[ ]:


import sys
sys.path.append('/content/SelfPlayAI')


# In[ ]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git remote set-url origin https://GodunicornIzek:{token}@github.com/Godunicornizek/SelfPlayAI.git')


# # TicTacToe Monte Carlo Tree Search

# In[ ]:


import numpy as np
import math
import random
import torch


# In[ ]:


#from TicTacToe import TicTacToe


# In[ ]:


#tictactoe = TicTacToe()


# In[ ]:


#print(tictactoe.get_init())


# In[ ]:


class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.children = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        """Check whether node has expanded all actions, and return whether it is a terminal node or not"""
        return len(self.children) > 0

    def select(self):
        """Select a child node using the PUCT formula"""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def get_ucb(self, child):
        """Implementation of the PUCT formula"""
        # eps = 1e-8
        # # If the child has never been visited, treat Q as 0
        # if child.visit_count == 0:
        #     q_value = 0
        # else:
        #     q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        # # In TicTacToe, we want to place our opponent in a bad predicament.
        # # Hence, we want to choose the child that minimizes the q_value.
        # # This is the reason for the 1 - in the front.

        # # PUCT exploration term
        # u_value = self.args['C'] * math.sqrt(math.log(max(1, self.visit_count)) / max(eps, child.visit_count))

        # return q_value + u_value
        if child.visit_count == 0:
            q_value = 0
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * child.prior * (math.sqrt(self.visit_count) / (1 + child.visit_count))

    def expand(self, policy):
        # action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        # self.expandable_moves[action] = 0
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player = -1)

                child = Node(self.game, self.args, child_state, self, action, prob) # Stores P(s,a) in each child
                self.children.append(child)
                return child

    # def simulate(self):
    #     value, is_terminal = self.game.get_value_and_terminated(self.state, self.action_taken)
    #     value = self.game.get_opponent_value(value)

    #     if is_terminal:
    #         return value

    #     rollout_state = self.state.copy()
    #     rollout_player = 1
    #     while True:
    #         valid_moves = self.game.get_valid_moves(rollout_state)
    #         action = np.random.choice(np.where(valid_moves == 1)[0])
    #         rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
    #         value, is_terminal = self.game.get_value_and_terminated(rollout_state, action)
    #         if is_terminal:
    #             if rollout_player == -1:
    #                 value = self.game.get_opponent_value(value)
    #             return value

    #         rollout_player = self.game.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)



# In[ ]:


class MCTS:
    def __init__(self, game, args: dict, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state)

        for search in range(self.args['num_searches']):
            node = root

            # Phase 1: Selection loop: continues as long as the node has no untried actions
            # and already has children. If needed, it will continue to select with PUCT until it
            # reaches a leaf node
            while node.is_fully_expanded():
                node = node.select()

            # Important distinction: if the method below returns "won", it is referring to the opponent
            # The returned value is from the perspective of the player who made action_taken.
            # Since this node represents the opponent’s turn, the value must be negated during backpropagation.
            # Note: checking whether the node is terminal is crucial for determining whether step 2 is
            #       to be executed
            value, is_terminal = node.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            # Check terminal node.
            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state)).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node = node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs



# # Testing MCTS

# In[ ]:


if __name__ == "__main__":
    tictactoe = TicTacToe()
    player = 1

    args = {
        'C': 1.41,
        'num_searches': 1000
    }
    mcts = MCTS(tictactoe, args)

    state = tictactoe.get_init()

    while True:
        print(state)

        if player == 1:
            valid_moves = tictactoe.get_valid_moves(state)
            print("valid moves", [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"{player}:"))

            if valid_moves[action] == 0:
                print("action not valid")
                continue
        else:
            neutral_state = tictactoe.change_perspective(state, player)
            mcts_probs = mcts.search(neutral_state)
            action = np.argmax(mcts_probs)


        state = tictactoe.get_next_state(state, action, player)

        value, is_terminal = tictactoe.get_value_and_terminated(state, action)

        if is_terminal:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = tictactoe.get_opponent(player)


# # Code for pushing to Github

# The following code does not need to be run again:

# In[3]:


#!git clone https://github.com/Godunicornizek/SelfPlayAI.git


# Run the following code after a workflow for pushing to Git

# In[4]:


get_ipython().system('git config --global user.name "GodunicornIzek"')
get_ipython().system('git config --global user.email "godunicornizek@gmail.com"')


# In[6]:


if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')

    get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects')

    get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects/SelfPlayAI')
    get_ipython().system('jupyter nbconvert --to python TicTacToeMCTS.ipynb')

    get_ipython().system('git status')

    get_ipython().system('git add TicTacToe.ipynb TicTacToe.py')

    get_ipython().system('git commit -m "Create TicTacToe MCTS class"')

    import getpass
    token = getpass.getpass("Enter GitHub token: ")

    get_ipython().system('git remote set-url origin https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git')

    get_ipython().system('git push origin main')


# In[ ]:





# # New Code for pushing to Github

# The following code does not need to be run again:

# In[7]:


get_ipython().system('git clone https://github.com/Godunicornizek/SelfPlayAI.git')


# Run the following code after a workflow for pushing to Git

# In[8]:


get_ipython().system('git config --global user.name "GodunicornIzek"')
get_ipython().system('git config --global user.email "godunicornizek@gmail.com"')


# In[9]:


if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')

    get_ipython().run_line_magic('cd', '/content/drive/MyDrive')
    get_ipython().system('mkdir -p Projects')
    get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects')

    get_ipython().system('mv /content/drive/MyDrive/SelfPlayAI/TicTacToeMCTS.ipynb      /content/drive/MyDrive/Projects/SelfPlayAI/')

    get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects/SelfPlayAI')
    get_ipython().system('jupyter nbconvert --to python TicTacToeMCTS.ipynb')

    get_ipython().system('git status')

    get_ipython().system('git add TicTacToeMCTS.ipynb TicTacToeMCTS.py')

    get_ipython().system('git commit -m "Create MCTS class and Node class"')

    import getpass
    token = getpass.getpass("Enter GitHub token: ")

    get_ipython().system('git remote set-url origin https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git')

    get_ipython().system('git push origin main')



# In[ ]:




