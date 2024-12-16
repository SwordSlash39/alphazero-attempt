import torch
from Board import chessboard
from mcts import MCTS
from network import Actor, Critic, device

def one_versus_one(new_actor: Actor, new_critic: Critic):    
    # Create new actor critic
    new_actor.eval()
    new_critic.eval()
    
    # Get old actor critic
    old_actor, old_critic = Actor().to(device), Critic().to(device)
    old_actor.load_state_dict(torch.load("actor.pth"))
    old_critic.load_state_dict(torch.load("critic.pth"))
    
    # Play 100 FAST games to determine the winner
    new_model_wins = 0
    old_model_wins = 0
    draws = 0
    for game in range(50):
        board = chessboard()
        movecounter = 0
        mcts_new = MCTS(board, new_actor, new_critic)
        mcts_old = MCTS(board, old_actor, old_critic)
        while not board.game_end() and movecounter < 100:        
            movecounter += 1
                        
            # Run MCTS new
            move, _ = mcts_new.search(250)
            
            # Push move & Update MCTS
            board.push(move)
            mcts_new.update(board)
            mcts_old.update(board)
            
            if board.game_end():
                break
            
            # Run MCTS old
            move, _ = mcts_old.search(250)
            
            # Push move & Update MCTS
            board.push(move)
            mcts_new.update(board)
            mcts_old.update(board)
        
        # Determine winner
        r = board.get_reward()
        if r == 1:
            new_model_wins += 1
        elif r == -1:
            old_model_wins += 1
        else:
            draws += 1
    
    print(f"Win ratio: {new_model_wins/old_model_wins:.2f}")
    print(f"Draws: {draws}")
    
    # IMPORTANT!!! Set the models back to train mode
    new_actor.train()
    new_critic.train()
    
    return new_model_wins/old_model_wins