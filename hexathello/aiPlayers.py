"""
    Interface for AI agents to play hexathello
"""

from . import engine, jable
import numpy as np

from typing import Literal, Protocol, Self

class PredictionModel( Protocol ):
    """
        Interface for Keras Network and other ML models to decide on a move
        This prevents us from needing Keras as an import
    """
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

# -- AI Agents

class HexAgent():
    """
        Semi Abstract parent class for Agents which play Hexathello
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        p_random: float = 0.0,
        player_id: int | None = None,
        ai_id: str = '',
        hexagonGridHelper: engine.HexagonGridHelper | None = None
        ) -> None:
        self.size = size
        self.player_count = player_count
        self.p_random = p_random
        
        self.player_id = player_id
        self.ai_id = ai_id
        self.hexagonGridHelper = hexagonGridHelper
        
        return
    #/def __init__
    
    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        raise NotImplementedError
    #/def getMove_fromBoardState
#/class HexAgent

# -- HexAgent Helpers

def _random_play(
    moveChoiceDict: engine.MoveChoiceDict,
    rng: np.random.Generator
    ) -> engine.QRTuple:
    """
        Choice randomly from provided choices, assuming they are legal
    """
    qr_list: list[ engine.QRTuple ] = [
        qr for qr in moveChoiceDict.keys()
    ]

    return qr_list[
        rng.choice(
            len( qr_list )
        )
    ]
#/def _random_play

def _greedy_play(
    moveChoiceDict: engine.MoveChoiceDict,
    rng: np.random.Generator
    ) -> engine.QRTuple:
    """
        Choose randomly from those which capture the most pieces
    """
    max_captures: int = max(
        len( capture_list ) for capture_list in moveChoiceDict.values()
    )
    
    qr_list: list[ engine.QRTuple ] = [
        qr for qr, capture_list in moveChoiceDict.items()
            if len( capture_list ) == max_captures
    ]
    
    return qr_list[
        rng.choice(
            len( qr_list )
        )
    ]
#/def _greedy_play

# -- Hex Agents

class KerasHexAgent( HexAgent ):
    """
        Uses a tensorflow keras network to make decisions, via a PredictionModel, most likely a trained neural net
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        p_random: float = 0.0,
        player_id: int | None = None,
        ai_id: str = '',
        hexagonGridHelper: engine.HexagonGridHelper | None = None,
        brain: PredictionModel | None = None
        ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            p_random = p_random,
            player_id = player_id,
            ai_id = ai_id,
            hexagonGridHelper = hexagonGridHelper
        )
        if self.hexagonGridHelper is None:
            self.hexagonGridHelper = engine.HexagonGridHelper(
                size = self.size,
                player_count = self.player_count
            )
        #
        self.brain = brain

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
    
    def getBoardState_asRelativeStateVector(
        self: Self,
        boardState: engine.BoardState
        ) -> np.ndarray:
        """
            Gets the board state as a vector from the point of view of the agent, aka the agent as if they were player 0. This ensures consistent learning no matter the player index.
        """
        boardState_vector: np.ndarray = self.hexagonGridHelper.stateVector_from_boardState(
            boardState
        )
        
        return get_relativeStateVector(
            boardState_vector = boardState_vector,
            player_id = self.player_id,
            player_count = self.player_count
        )
    #/def getBoardState_asRelativeStateVector
    
    def getMove_fromBoardState_classDecision(
        self: Self
        ):
        raise Exception("UC")
    #/def getMove_fromBoardState_classDecision
    
    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        #
        assert self.player_id is not None
        moveChoiceDict: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        qr: engine.QRTuple
        if len( moveChoiceDict ) == 1:
            qr = next( _qr for _qr in moveChoiceDict.keys() )
        #
        # Check if random play
        elif self.p_random >= 1:
            # Guaranteed random
            qr = _random_play(
                moveChoiceDict,
                rng = rng
            )
        #
        else:
            # Roll for random or regular decision
            if self.p_random <= 0 or rng.choice(
                2,
                p = ( 1-self.p_random, self.p_random )
            ) == 0:
                # Did not roll random
                # Turn board state into a vector suitable for brain, by making it relative to self
                boardState_vector: np.ndarray = self.getBoardState_asRelativeStateVector(
                    boardState
                )
                
                # Use our brain to get weighs of each move
                # NOTE: We can choose from these weights using
                #   self.chooseMove(...), something like the max
                moveChoice_vector: np.ndarray = self.brain.predict(
                    boardState_vector
                )
                
                # Mask it with legal moves
                
                assert len( moves ) > 0
                legal_moves_vector: np.ndarray = np.zeros(
                    shape = (len( moveChoice_vector ),),
                    dtype = float
                )
                
                for qr in moveChoiceDict:
                    moves_vector[
                        self.hexagonGridHelper.index_from_qr_tuple(
                            qr
                        )
                    ] = 1.0
                #/for qr in moveChoiceDict

                # Distribution of choices might have been given; make choice, potentially probabilistically
                # This result is a literal spot index, not relative to anyone
                moveChoice_final: int = self.choseMove(
                    moveChoice_vector * legal_moves_vector
                )

                qr: engine.QRTuple = self.hexagonGridHelper.qr_from_index[
                    moveChoice_final
                ]
                
                assert qr in moveChoiceDict
            #
            else:
                # Rolled random
                qr = _random_play(
                    moveChoiceDict,
                    rng = rng
                )
            #/if { nonrandom }/else
        #/if len( moveChoiceDict ) == 1/switch { random }
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
    
    def prep_training_history(
        self: Self,
        history: jable.JyFrame
        ) -> jable.JyFrame:
        """
            Turn history into something we can actually learn from
            
            The default behavior is to learn only from winning moves and ignore others
        """
        assert history.get_fixed( "history_type" ) == 'pov'
        
        return jable.filter(
            history,
            {'winner': 0}
        )
    #/def prep_training_history
    
    def train(
        self: Self,
        history: jable.JyFrame,
        *args,
        **kwargs
        ) -> None:
        """
            Train the brain with a Pov History JyFrame
            
            *args, **kwargs: Passed to `brain.fit()`, see
            https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
            
            Consider:
            - epochs
            
            # TODO: handle weights
        """
        assert history.get_fixed( "history_type" ) == 'pov'
        
        # Prep history
        history_prepped: jable.JyFrame = self.prep_training_history(
            history
        )
        
        X: np.ndarray = np.array(
            history_prepped[ "board_state" ]
        )
        
        y: np.ndarray = np.array(
            history_prepped[ "player_action" ]
        )
        
        self.brain.fit( X, y, *args, **kwargs )
        
        return
    #/def train
#/class KerasHexAgent

class GreedyHexAgent( HexAgent ):
    """
        Picks randomly from among moves which give the most immediate captures
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        p_random: float = 0.0,
        player_id: int | None = None,
        ai_id: str = 'GreedyHexAgent',
        hexagonGridHelper: engine.HexagonGridHelper | None = None
        ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id,
            p_random = p_random,
            ai_id = ai_id,
            hexagonGridHelper = hexagonGridHelper
        )
        return
    #/def __init__
    
    def getMove_fromBoardState_classDecision(
        self: Self
        ):
        raise Exception("UC")
    #/def getMove_fromBoardState_classDecision
    
    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        """
            Pick a random index which has the largest number of captures
        """
        assert self.player_id is not None
        moveChoiceDict: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )

        qr: engine.QRTuple
        # Check if only one option
        if len( moveChoiceDict ) == 1:
            qr = next( _qr for _qr in moveChoiceDict.keys() )
        #
        # Check if random play
        elif self.p_random >= 1:
            # Guaranteed random
            qr = _random_play(
                moveChoiceDict,
                rng = rng
            )
        #
        else:
            # Roll for random or regular decision
            if self.p_random <= 0 or rng.choice(
                2,
                p = ( 1-self.p_random, self.p_random )
            ) == 0:
                # Did not roll ranndom
                qr = _greedy_play(
                    moveChoiceDict
                )
            #
            else:
                # Rolled random
                qr = _random_play(
                    moveChoiceDict,
                    rng = rng
                )
            #/if { nonrandom }/else
        #/if len( moveChoiceDict ) == 1/switch { random }
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class GreedyHexAgent( HexAgent )
