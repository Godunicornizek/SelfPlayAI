#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Mount GDrive
from google.colab import drive
drive.mount('/content/drive')


# # Creating a TicTacToe Class

# In[ ]:


import numpy as np
np.__version__


# In[ ]:


class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count
        # self.action_correspondence = np.arange(0, action_size).reshape(self.row_count, self.column_count)

    def get_init(self) -> np.ndarray:
        return np.zeros((self.row_count, self.column_count))

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        row = action // self.column_count
        column = action % self.column_count
        state[row, column] = player
        return state
        # return np.where(state == 0, player, state)

    # def determine_legal(self, state: np.ndarray, action: int) -> bool:
    #     row = action // self.column_count
    #     column = action % self.column_count
    #     if state[row, column] != 0:
    #         return False
    #     return True

    def get_valid_moves(self, state: np.ndarray):
        return (state.reshape(-1) == 0).astype(np.uint8)

    def determine_won(self, state: np.ndarray, action: int) -> bool:
        row = action // self.column_count
        column = action % self.column_count
        player = state[row, column]

        return (
             np.sum(state[row, :] == player * self.column_count)
             or np.sum(state[:, column]) == player * self.row_count
             or np.sum(np.diag(state)) == player * self.row_count
             or np.sum(np.diag(np.fliplr(state))) == player * self.row_count
        )

    def get_value_and_terminated(self, state, action):
        if self.determine_won(state, action):
            return 1, True
        elif np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        else:
            return 0, False

    def get_opponent(self, player):
        return -player


# # Testing TicTacToe

# In[ ]:


if __name__ == "__main__":
    # Initializing the board and players
    tictactoe = TicTacToe()
    player = 1
    state = tictactoe.get_init()
    opponent = tictactoe.get_opponent

    # Starting for 1 turn
    action = 4
    state = tictactoe.get_next_state(state, action, player)
    print(state)
    value, terminated = tictactoe.get_value_and_terminated(state, action)
    print(f"The game has {'terminated' if bool(terminated) else 'not terminated'}.")

    import random

    tictactoe2 = TicTacToe()
    player = 1
    state = tictactoe.get_init()
    opponent = tictactoe.get_opponent

    while True:
        print(state)
        valid_moves = tictactoe2.get_valid_moves(state)
        print("valid moves", [i for i in range(tictactoe2.action_size) if valid_moves[i] == 1])
        action = int(input(f"{player}:"))

        if valid_moves[action] == 0:
            print("Action not valid")
            continue

        state = tictactoe2.get_next_state(state, action, player)

        value, is_terminated = tictactoe2.get_value_and_terminated(state, action)

        if is_terminated:
            print(state)
            if value == 1:
                print(player, "won")
            else:
                print("draw")
            break

        player = tictactoe2.get_opponent(player)



# # Code for pushing to Github (Not used)

# In[ ]:


get_ipython().system('git clone https://github.com/GodunicornIzek/SelfPlayAI.git')


# In[ ]:


get_ipython().system('git config --global user.email "godunicornizek@gmail.com"')
get_ipython().system('git config --global user.name "GodUnicornIzek"')


# In[ ]:


# Save the current notebook into /content with a specific name
from google.colab import drive, files, _import_hooks
import os

# If you want to save the current notebook with a known name:
get_ipython().system('cp "/content/AlphaZeroImplementation.ipynb" "/content/AlphaZeroImplementation.ipynb"')


# In[ ]:


from google.colab import files

# This will prompt you to upload a file if you want to bring it from your local machine
uploaded = files.upload()


# In[ ]:


get_ipython().system('ls /content/')


# In[ ]:


get_ipython().system('cp /content/AlphaZeroImplementation.ipynb /content/SelfPlayAI/')


# In[ ]:


# Instead of hardcoding the token:
import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornIzek/SelfPlayAI.git main')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/SelfPlayAI')

# Rename it to a clean filename
get_ipython().system('mv "AlphaZeroImplementation (2).ipynb" AlphaZeroImplementation.ipynb')
get_ipython().system('rm "AlphaZeroImplementation (1).ipynb"')


# In[ ]:


get_ipython().system('git add AlphaZeroImplementation.ipynb')
get_ipython().system('git commit -m "Add/update notebook"')


# In[ ]:


get_ipython().system('git remote set-url origin https://github.com/Godunicornizek/SelfPlayAI.git')


# In[ ]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main')


# In[ ]:


# -------------------------------
# Colab GitHub Push Workflow
# -------------------------------

from google.colab import files, drive
import os, getpass

# --- 1. Configure Git ---
get_ipython().system('git config --global user.email "godunicornizek@gmail.com"')
get_ipython().system('git config --global user.name "GodunicornIzek"')

# --- 2. Clone repo (if not already present) ---
repo_dir = "/content/SelfPlayAI"
if not os.path.exists(repo_dir):
    get_ipython().system('git clone https://github.com/Godunicornizek/SelfPlayAI.git {repo_dir}')

# --- 3. Move into repo directory ---
get_ipython().run_line_magic('cd', '{repo_dir}')

# --- 4. Upload the notebook if not already in Colab ---
uploaded = files.upload()  # Choose your tictactoe.ipynb
# This will place the file in /content/, copy it into the repo
for fname in uploaded.keys():
    get_ipython().system('cp "/content/{fname}" "{repo_dir}/{fname}"')

# --- 5. Optional: rename notebook to clean name ---
get_ipython().system('mv "{repo_dir}/tictactoe.ipynb" "tictactoe.ipynb"')

# --- 6. Stage and commit changes ---
get_ipython().system('git add tictactoe.ipynb')
get_ipython().system('git commit -m "Add/update TicTacToe notebook"')

# --- 7. Set remote URL just in case ---
get_ipython().system('git remote set-url origin https://github.com/Godunicornizek/SelfPlayAI.git')

# --- 8. Push safely using getpass ---
token = getpass.getpass("Enter GitHub token: ")
get_ipython().system('git push https://GodunicornIzek:{token}@github.com/Godunicornizek/SelfPlayAI.git main')

# --- 9. Status check ---
get_ipython().system('git status')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/SelfPlayAI')

# Copy the uploaded file into the repo folder (overwrite if necessary)
get_ipython().system('cp /content/TicTacToe.ipynb ./TicTacToe.ipynb')

# Stage the notebook
get_ipython().system('git add TicTacToe.ipynb')

# Commit
get_ipython().system('git commit -m "Add/update TicTacToe notebook"')

# Push using getpass token
import getpass
token = getpass.getpass("Enter GitHub token: ")
get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/SelfPlayAI')

# Remove AlphaZeroImplementation.ipynb from Git
get_ipython().system('git rm --cached AlphaZeroImplementation.ipynb')

# Commit the removal
get_ipython().system('git commit -m "Remove old AlphaZeroImplementation notebook from repo"')


# In[ ]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main')


# In[ ]:


get_ipython().run_line_magic('cd', '/content/SelfPlayAI')


# In[ ]:


get_ipython().system('cp /content/TicTacToe.ipynb ./TicTacToe.ipynb  # Notebook')
get_ipython().system('jupyter nbconvert --to python TicTacToe.ipynb   # Optional: update .py module')


# In[ ]:


get_ipython().system('git add TicTacToe.py TicTacToe.ipynb')


# In[ ]:


get_ipython().system('git add TicTacToe.ipynb')


# In[ ]:


get_ipython().system('git commit -m "Update TicTacToe notebook"')


# In[ ]:


get_ipython().system('git commit -m "Update TicTacToe.py: wrap demo code in __main__"')


# In[ ]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main')


# In[ ]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git pull https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main --allow-unrelated-histories')


# In[ ]:


get_ipython().system('git status')


# In[ ]:


get_ipython().system('git add TicTacToe.ipynb TicTacToe.py')
get_ipython().system('git commit -m "Update TicTacToe notebook and module"')
get_ipython().system('git push https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git main')


# # New Code for pushing to Github

# The following code does not need to be run again:

# In[ ]:


get_ipython().system('git clone https://github.com/Godunicornizek/SelfPlayAI.git')


# In[ ]:


get_ipython().system('git config --global user.name "GodunicornIzek"')
get_ipython().system('git config --global user.email "godunicornizek@gmail.com"')


# Run the following code after a workflow for pushing to Git

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[13]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive')
get_ipython().system('mkdir -p Projects')
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects')


# In[14]:


get_ipython().system('mv /content/drive/MyDrive/SelfPlayAI/TicTacToe.ipynb     /content/drive/MyDrive/Projects/SelfPlayAI/')


# In[20]:


get_ipython().run_line_magic('cd', '/content/drive/MyDrive/Projects/SelfPlayAI')
get_ipython().system('jupyter nbconvert --to python TicTacToe.ipynb')


# In[22]:


get_ipython().system('git status')


# In[23]:


get_ipython().system('git add TicTacToe.ipynb TicTacToe.py')


# In[25]:


get_ipython().system('git commit -m "Update TicTacToe game logic"')


# In[28]:


import getpass
token = getpass.getpass("Enter GitHub token: ")

get_ipython().system('git remote set-url origin https://GodUnicornIzek:{token}@github.com/GodUnicornizek/SelfPlayAI.git')

