import chess, time, random, math, torch
from Board import chessboard
from mcts import MCTS
from network import Actor, Critic, device
from Trainer import Trainer

# Create a chess board
board = chessboard()

# Create actor critic
actor, critic = Actor().to(device), Critic().to(device)
actor.load_state_dict(torch.load("actor.pth"))
critic.load_state_dict(torch.load("critic.pth"))
actor.eval()
critic.eval()

with torch.no_grad():
    while not board.game_end():
        try:
            move = input("Enter move: ")
            board.push_san(move)
            
            mcts = MCTS(board, actor, critic)
            move, probs = mcts.search(1000)
            str_move = board.board.san(move)
            board.push(move)
            
            print("Critics eval: ", probs)
            print("Eval: ",  critic(torch.tensor(board.getState()).to(device)).item())
            print("AI move: ", str_move)
            
        except chess.IllegalMoveError:
            print("bad move bruh")
        except ValueError:
            print("gg")
            break
