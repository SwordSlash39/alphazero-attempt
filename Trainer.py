import chess, torch
import chess.pgn
from collections import deque
from Board import chessboard
from mcts import MCTS
from network import Actor, Critic, device

class Trainer:
    def __init__(self, lr=0.0001, gamma=0.99, epsilon=0.2):
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=2*lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.train_set = deque(maxlen=2000)
    
    def convStrToIndex(move: str):
        return 8**3 * (ord(move[0]) - ord('a')) + 8**2 * (ord(move[1]) - ord('1')) + 8 * (ord(move[2]) - ord('a')) + (ord(move[3]) - ord('1'))
    
    def backprop(self, winner):     # 1 if white, -1 if black, 0 if draw        
        batch_size = len(self.train_set)
        loss = torch.tensor(0.0).to(device)
        while self.train_set:
            pos = self.train_set.popleft()
            
            # Zero prev gradients
            self.critic_optim.zero_grad()
            self.actor_optim.zero_grad()
            
            # Train critic
            # no need distinction between white & black
            # as critic is trained to predict the value of the state
            gain = winner - self.critic(pos["state"])
            
            critic_loss = gain**2
            loss += critic_loss[0][0]     # critic's loss
                
            # Train actor
            try:
                actor_output = self.actor(pos["state"], pos["board"])
                actor_loss = -torch.sum(actor_output * pos["mcts_probs"])   # note actor output already logsoftmax
                loss += actor_loss
            except RuntimeError:
                # occurs if node search is lower than legal moves
                pass
            
        # backpropagate
        loss /= batch_size
        loss.backward()
        self.actor_optim.step()
        self.critic_optim.step()
        
        # Return avg loss
        return loss.item()

    def train(self, game_count: int, maxmoves: int = 1000, n=10000, custom_fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", console_log=True):
        for i in range(game_count):
            if console_log:
                print(f"----- Game {i+1} -----")
            
            # Create board
            try:
                board = chessboard(custom_fen)
            except ValueError:
                print(f"Invalid FEN: {custom_fen}, using default...")
                board = chessboard()
            
            # PGN handling to print to console (cos i like to see games)
            pgn = ""
            play_white = board.board.turn == chess.WHITE
            if not play_white:
                pgn += "1... "
            
            # Set up MCTS
            mcts = MCTS(board, self.actor, self.critic)
            
            movecounter = 0
            while not board.game_end() and movecounter < 2*maxmoves:        
                movecounter += 1
                if movecounter % 2 == 1 and play_white:
                    pgn += str(int(movecounter/2 + 0.5)) + ". "
                    if console_log:
                        print(f"move {int(movecounter/2 + 0.5)}")
                    
                # Create MCTS
                np_state = board.getState()
                state = torch.tensor(np_state).to(device)
                
                # Run MCTS
                move, mcts_probs = mcts.search(n)
                
                # Copy board first (for pgn)
                old_board = board.copyboard()
                
                # Push move & Update MCTS
                board.push(move)
                mcts.update(board)
                
                # PGN handling (use old board cos we just pushed the move)
                pgn += old_board.board.san(move) + " "
                if movecounter % 2 == 0 and not play_white:
                    pgn += str(int(movecounter/2)) + ". "
                    if console_log:
                        print(f"move {int(movecounter/2 + 0.5)}")
                
                # Store state of game (for training)  
                self.train_set.append({"state": state.clone().detach(), "color": int(np_state[10][0][0]), # 1 if white, 0 if black
                                       "board": old_board,
                                       "mcts_probs": torch.tensor(mcts_probs).to(device)})

            loss = self.backprop(board.get_reward())
            print(board.get_reward())
            print(f"Game {i+1} loss: {loss}")
            print(f"Game: {pgn}")
        
    def savemodel(self):
        torch.save(self.actor.state_dict(), f"actor.pth")
        torch.save(self.critic.state_dict(), f"critic.pth")
        torch.save(self.actor_optim.state_dict(), "actor_optim.pth")
        torch.save(self.critic_optim.state_dict(), "critic_optim.pth")
    def loadmodel(self):
        self.actor.load_state_dict(torch.load("actor.pth", map_location=device))
        self.critic.load_state_dict(torch.load("critic.pth", map_location=device))
        self.actor_optim.load_state_dict(torch.load("actor_optim.pth", map_location=device))
        self.critic_optim.load_state_dict(torch.load("critic_optim.pth", map_location=device))