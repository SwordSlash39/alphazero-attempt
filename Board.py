import chess
import numpy as np

class chessboard:
    def __init__(self, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.board = chess.Board(fen)
    
    def get_fen(self) -> str:
        return self.board.fen()
    
    def hash_board(self) -> int:
        state = self.board.__dict__
        hash = state["occupied_co"][1] * state["occupied_co"][0]
        hash *= state["pawns"] if state["pawns"] != 0 else 985969
        hash *= state["knights"] if state["knights"] != 0 else 621289
        hash *= state["bishops"] if state["bishops"] != 0 else 842341
        hash *= state["rooks"] if state["rooks"] != 0 else 998273
        hash *= state["queens"] if state["queens"] != 0 else 995119
        hash *= state["kings"] if state["kings"] != 0 else 985933
        hash *= state["occupied"] if state["occupied"] != 0 else 921457
        hash *= state["castling_rights"] if state["castling_rights"] != 0 else 836477      # im the problem
        hash *= 93187 if state["turn"] else 93281
        hash *= 317887 if state["ep_square"] is None else 317923
        return hash
    
    def get_legal_moves(self) -> list:
        return list(self.board.legal_moves)
    
    def make_move(self, move: str):
        self.board.push(chess.Move.from_uci(move))
    
    def push_san(self, move: str):
        self.board.push_san(move)
        
    def push(self, move: chess.Move):
        self.board.push(move)
        
    def game_end(self) -> str:
        # only checks if game has ended; dosen't check what happened
        if self.board.can_claim_threefold_repetition() or self.board.is_fifty_moves() or self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_insufficient_material():
            # draw the game
            return True
        return False
    
    def get_reward(self) -> float:
        outcome = self.board.outcome()
        
        # Game hasnt ended
        if outcome is None:
            return 0
        
        outcome = outcome.winner    # see who won
        if outcome == chess.WHITE:    # white wins
            return 1
        elif outcome == chess.BLACK:  # black wins
            return -1
        else:
            return 0
        
    
    def copyboard(self):
        copied = chessboard()
        copied.board = self.board.copy()
        return copied
    
    def make_move_copyboard(self, move: chess.Move):
        # returns a copy of the board after making the move
        new_board = self.copyboard()
        new_board.push(move)
        return new_board
        
    
    def getState(self):
        # This function returns the state of the board in a format that can be used by the neural network
        # The state is a 8x8x11 tensor
        # The first 8x8x1 is pos of white pieces (not indicating the type of piece)
        # Next 8x8x1 is pos of black pieces
        # The next 8x8x6 is the type of piece
        # Next 8x8x1 is castling rights
        # Next 8x8x1 is en passant square
        # Last 8x8x1 is whose turn it is
        
        state = np.zeros((11, 8, 8), dtype=np.float32)
        
        # get color of pieces
        for i in range(8):
            for j in range(8):
                piece = self.board.piece_at(chess.square(i, j))
                
                if piece is not None:
                    # get color of pieces
                    state[1-piece.color, 7-j, 7-i] = 1
                    
                    # get type of pieces
                    state[1+piece.piece_type, 7-j, 7-i] = 1
                
        # get castling rights
        if self.board.has_kingside_castling_rights(chess.WHITE):
            state[8, 0, 4:] = 1
        if self.board.has_queenside_castling_rights(chess.WHITE):
            state[8, 0, :4] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            state[8, 7, 4:] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            state[8, 7, :4] = 1
                
        # get en passant square
        if self.board.ep_square is not None:
            ep_square = self.board.ep_square
            state[9, chess.square_file(ep_square), chess.square_rank(ep_square)] = 1
        
        # get whose turn it is
        if self.board.turn == chess.WHITE:
            state[10, :, :] = 1
        
        return state