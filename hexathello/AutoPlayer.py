"""
    Interface to play games (of Hexathello) and show the results
"""

from . import HexathelloEngine

import numpy as np

from typing import Protocol, Self

# -- AI Agents

class HexAgent():
    """
        Agents which play hexathello
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None
        ) -> None:
        self.size = size
        self.player_count = player_count
        self.player_id = player_id
        return
    #/def __init__
    
    def getMove_fromBoardState(
        self: Self,
        boardState: HexathelloEngine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ HexathelloEngine.CellCapture ] = []
        ) -> HexathelloEngine.PlayerMove:
        raise NotImplementedError
    #/def getMove_fromBoardState
#/class HexAgent

# Interface for Keras Network and other ML models to decide on a move
class PredictionModel( Protocol ):
    def fit( self: Self, X: np.ndarray, y: np.ndarray, **kwargs ) -> Self:
        raise NotImplementedError
    #
    
    def predict( self: Self, X: np.ndarray ) -> np.ndarray:
        raise NotImplementedError
    #
    
    def call( self: Self, X ):
        raise NotImplementedError
    #
#/class PredictionModel( Protocol )

class KerasHexAgent( HexAgent ):
    """
        Uses a tensorflow keras network to make decisions, via a PredictionModel, most likely a trained neural net
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        brain: PredictionModel | None = None,
        hexagonGridHelper: HexathelloEngine.HexagonGridHelper | None = None,
        player_id: int | None = None
    ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id
        )
        self.brain = brain
        self.hexagonGridHelper = hexagonGridHelper
        return
    #/def __init__
    
    def chooseMove(
        self: Self,
        moveChoice_vector: np.ndarray,
        rng: np.random.Generator
    ) -> int:
        """
            Get the index of a choice by one way or another
            Different implementations might use max, or soft max among those
                positive
                
            This is the default implementation where we choose the max
        """
        return np.argmax( moveChoice_vector )
    #/def chooseMove
    
    def getMove_fromBoardState(
        self: Self,
        boardState: HexathelloEngine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ HexathelloEngine.CellCapture ] = []
    ) -> HexathelloEngine.PlayerMove:
        # Turn moves into a vector suitable for brain
        boardState_vector: np.ndarray = self.hexagonGridHelper.stateVector_from_boardState( boardState )
        
        # Use our brain to choose the best move
        moveChoice_vector: np.ndarray = self.brain.predict(
            boardState_vector,
            rng = rng
        )
        # Distribution of choices might have been given; make choice, potentially probabilistically
        moveChoice_final: int = self.choseMove( moveChoice_vector )
        moveChoice_qr: HexathelloEngine.QRTuple = self.hexagonGridHelper.qr_from_index[
            moveChoice_final
        ]

        # Extra careful:
        # Check it's a legal move
        # TODO: handle better
        if True:
            moves: HexathelloEngine.MoveChoiceDict = HexathelloEngine.getMoves_forPlayer(
                player = self.player_id,
                boardState = boardState
            )
            assert moveChoice_qr in moves
        #/if True
        
        return {
            "turn_index": turn_index,
            "q": moveChoice_qr[0],
            "r": moveChoice_qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class KerasHexAgent

class RandomHexAgent( HexAgent ):
    """
        Picks randomly from legal moves
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None
        ) -> None:
        
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id
        )
        return
    #/def __init__
    
    def getMove_fromBoardState(
        self: Self,
        boardState: HexathelloEngine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ HexathelloEngine.CellCapture ] = []
        ) -> HexathelloEngine.PlayerMove:
        assert self.player_id is not None
        moves: HexathelloEngine.MoveChoiceDict = HexathelloEngine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        
        
        # Pick randomly from possible moves
        qr_list: list[ HexathelloEngine.QRTuple ] = [
            qr for qr in moves.keys()
        ]

        qr: HexathelloEngine.QRTuple = qr_list[
            rng.choice(
                len( qr_list )
            )
        ]
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class RandomHexAgent( HexAgent )

class GreedyHexAgent( HexAgent ):
    """
        Picks randomly from among moves which give the most immediate captures
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None
    ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id
        )
        return
    #/def __init__
    
    def getMove_fromBoardState(
        self: Self,
        boardState: HexathelloEngine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ HexathelloEngine.CellCapture ] = []
        ) -> HexathelloEngine.PlayerMove:
        """
            Pick a random index which has the largest number of captures
        """
        assert self.player_id is not None
        moves: HexathelloEngine.MoveChoiceDict = HexathelloEngine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        max_captures: int = max(
            len( capture_list ) for capture_list in moves.values()
        )
        
        qr_list: list[ HexathelloEngine.QRTuple ] = [
            qr for qr, capture_list in moves.items()
                if len( capture_list ) == max_captures
        ]
        
        qr: HexathelloEngine.QRTuple = qr_list[
            rng.choice(
                len( qr_list )
            )
        ]
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class GreedyHexAgent( HexAgent )

def runHexathello_withAgents(
    agents: list[ HexAgent ],
    size: int,
    logging_level: int = 0,
    rng: np.random.Generator | None = None
    ) -> None:
    """
        Plays a game with set AI and prints everything as it goes
        
        agents: An initialized set of agents to play
    """
    # Check we numbered agents appropriately
    for i in range( len( agents ) ):
        if agents[i].player_id is None:
            agents[i].player_id = i
        #
        else:
            assert agents[i].player_id == i
        #/if agents[i].player_id is None/else
    #/for i in range( len( agents ) )
    
    if rng is None:
        rng = np.random.default_rng()
    #
    
    player_count: int = len( agents )
    
    # Initialize status for the Hexathello, with random first player
    player_start: int = rng.choice( player_count )
    hexathello = HexathelloEngine.new_hexathello(
        player_count = player_count,
        size = size,
        player_start = player_start,
        logging_level = logging_level
    )
    
    # SAFETY VALVE: max number of turns is empty_count
    move_log: list[ HexathelloEngine.QRTuple ] = []
    empty_count: int = hexathello.status["empty_count"]
    for tick_index in range( empty_count ):
        turn_index: int = hexathello.status["turn_index"]
        
        if turn_index > 0:
            assert hexathello.boardState[ move_log[ turn_index-1] ]["owner"] is not None
        #
        
        next_player_index = hexathello.status["current_player"]
        next_move: HexathelloEngine.PlayerMove = agents[
            next_player_index
        ].getMove_fromBoardState(
            boardState = hexathello.boardState,
            turn_index = turn_index,
            rng = rng,
            potential_moves = hexathello.potential_moves
        )
        
        next_qr = ( next_move["q"], next_move["r"] )
        
        if next_qr in move_log:
            print("Copied move: {}".format(next_qr))
            print( next_move )
            raise Exception("Bad move")
        #
        
        move_log.append(
            ( next_move["q"], next_move["r"] )
        )
        
        # Run the one update
        hexathello.queueUpdate( next_move )
        hexathello.applyUpdates()
        
        # Print the log
        while not hexathello.log.empty():
            logUpdate: dict = hexathello.log.get()
            HexathelloEngine.print_logUpdate( logUpdate )
            #print("# possible moves: {}:".format(hexathello.potential_moves))
        #/while not hexathello.log.empty()
        
        if hexathello.boardState[ next_qr ]["owner"] is None:
            print( next_move )
            raise Exception("Failed to update board state")
        #

        # Check if game is over
        if hexathello.status["game_complete"]:
            print("# Game done")
            print( hexathello.status )
            break
        #/if hexathello.status["game_complete"]
    #/for _ in range( empty_count )
    return
#/def printHexathello_withAgents
