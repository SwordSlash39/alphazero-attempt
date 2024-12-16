import torch, chess, time
import numpy as np
from Board import chessboard
from network import Actor, Critic, device

class Node:
    def __init__(self, state: chessboard, parent=None):
        self.state = state
        self.parent = parent
        self.children: list[Node] = []
        self.visits = 0 # to avoid division by 0
        self.value = 0
        
        self.cache_legalmoves = None
        self.cache_game_end = None
        self.marker = state.hash_board()

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)
        return child

    def update(self, value, batch=1):
        self.visits += batch
        self.value += value
    
    def cache_moves(self):
        self.cache_legalmoves = self.state.get_legal_moves()

    def fully_expanded(self):
        if self.cache_legalmoves is None:
            # cache legal moves for faster execution
            self.cache_moves()
        return len(self.children) == len(self.cache_legalmoves)
    
    def game_end(self):
        if self.cache_game_end is None:
            self.cache_game_end = self.state.game_end()
            
        return self.cache_game_end


class MCTS:
    def __init__(self, root_state: chessboard, actor: Actor, critic: Critic, gamma=0.99, exploration_weight=4, policy_weight=0.1):
        self.root: Node = Node(root_state)
        self.actor: Actor = actor
        self.critic: Critic = critic
        
        self.half_moves = 0
        self.cache_actor = {}
        self.cache_value = {}
        self.cache_states = {}
        
        # hyperparameters
        self.gamma = gamma
        self.C = exploration_weight
        self.p = policy_weight
        self.exploration_moves = 30
        
    def update(self, new_board_state: chessboard):
        self.half_moves += 1
        for child in self.root.children:
            if child.state.hash_board() == new_board_state.hash_board():
                self.root = child
                return
        
        # i guess we didnt research this
        self.root = Node(new_board_state)
        return

    def search(self, positions=10000):
        # t = time.time()
        while self.root.visits < positions:
            # Search best node
            node_list, orphan = self.select_node()
            if orphan:
                # orphan node (game ended on dis one)
                self.backpropagate(node_list[0], self.cache_value[node_list[0].state.hash_board()])
                continue
            
            # Run simulation
            state_tensors = torch.tensor(np.array([n.state.getState() for n in node_list]), device=device)
            values = self._simulate(state_tensors).view(-1).cpu().numpy()

            # Cache values and update nodes
            hashes = [n.state.hash_board() for n in node_list]
            self.cache_value.update(dict(zip(hashes, values)))

            # Update nodes in batch if possible (also calculate leaf node sum)
            leaf_node_sum = 0
            for n, v in zip(node_list, values):
                n.update(v)
                leaf_node_sum += v

            # Backpropagation
            self.backpropagate(node_list[0].parent, leaf_node_sum, len(node_list))
            
            
        # print(time.time() - t)
        return self.best_move() # return the best move, log_probs

    def select_node(self):
        node: Node = self.root
        while not node.game_end():
            if not node.fully_expanded():
                # fully expand the node & run eval on all
                nodes = [node.add_child(node.state.make_move_copyboard(move)) for move in node.cache_legalmoves]
                return nodes, False
            else:
                node = self.best_child(node)
        
        # somehow got to end of game 
        return [node], True

    def best_child(self, node: Node):
        if node.marker in self.cache_actor:
            actor_values = self.cache_actor[node.marker] 
        else:
            with torch.no_grad():
                actor_values = torch.exp(self.actor(torch.tensor(node.state.getState()).to(device), node.state))
                self.cache_actor[node.marker] = actor_values.cpu().numpy()
        
        actor_values = self.cache_actor[node.marker]
        visits = np.array([child.visits for child in node.children])
        values = np.array([child.value for child in node.children])
        color_multiplier = 1 if node.state.board.turn == chess.WHITE else -1
        try:
            uct_values = values + color_multiplier * self.C * actor_values * np.sqrt(np.log(node.visits) / (1 + visits))
        except ValueError:
            print(node.state.get_fen())
            print(actor_values)
            print(visits)
            raise ValueError
        
        best = node.children[np.argmax(uct_values)]
        
        return best


    def _simulate(self, state) -> float:
        # Receives (nodes, state_tensors)
        # finds the value of the state by simulating a game
        # What this function should do:
        # Simulate all the states with critic (NO actor)
        with torch.no_grad():
            actor_output = self.critic(state, single_input=False)
        
        return actor_output
        

    def backpropagate(self, node: Node, value: float, batch: int = 1):
        while node is not None:
            node.update(value, batch)
            node = node.parent

    def normalise(self, x):
        if self.half_moves > 2*self.exploration_moves:
            k = np.zeros(x.shape)
            k[np.argmax(x)] = 1
            return k
        return x / np.sum(x)
    
    def best_move(self) -> str:
        visits = [child.visits for child in self.root.children]
        print(visits)
        probabilities = self.normalise(np.array(visits, dtype=np.float32))  # Calculate softmax probabilities

        # Choose a move based on the softmax probabilities
        chosen_child = np.random.choice(self.root.children, p=probabilities)
        
        # Return the move and the log probabilities (for training)
        return chosen_child.state.board.peek(), probabilities
    