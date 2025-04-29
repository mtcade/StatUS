"""
    Interface for AI agents to play hexathello. They can play against eachother or humans.
    
    Using ``hexathello.autoPlayer.runHexathello_withAgents()``, we have them play games with eachother to generate data on good moves, by saving it as a history, as from ``hexathello.history.new_literalHistory()``. Agents can use this data to learn, likely with ``KerasHexAgent()`` to use `tensorflow.keras` for neural network decision making.
"""
import hexathello.engine as engine
import hexathello.history as history
import hexathello.jable as jable

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
        :param int size: Length of one side of the hexathello board
        :param int player_count: Number of players. Likely 2, 3, or 6 for hexathello
        :param float p_random: Probability of choosing randomly from among legal moves. Used to occasionally experiment.
        :param int|None player_id: Index of AI player, if in a game. Used to convert board states to a self pov.
        :param str ai_id: A name for the agent, for indexing, storage, and learning purposes.
        :param engine.HexagonGridHelper|None hexagonGridHelper: Instance of a game's helper for convering between indices of board states and qr tuple coordiates.
        
        Semi Abstract parent class for Agents which play hexathello, most likely from ``hexathello.autoPlayer.runHexathello_withAgents()`` or against a human in ``hexathello.game``
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
        #
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
        """
            :param engine.BoardState boardState: A dictionary mapping `(q,r)` tuples to board state dictionaries
            :param int turn_index: Turn when move was made, starting at 0.
            :param np.random.Generator rng: Random number generator instance for making some choices.
            :potential_moves: list of dictionaries of legal moves, mapped to captures which would be made. If not present, calculate using ``hexathello.engine.getMoves_forPlayer()``.
            :returns: A choice of move
            :rtype: engine.PlayerMove
        """
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
        Uses a tensorflow keras network to make decisions, via a PredictionModel, most likely a compiled neural network. This class can handle training the network, stored in `.brain` using `.train()`.
        
        To change the logic used, make a subclass and overwrite:
        
        `.chooseMove()`: Turn a vector of weights into the index of the chosen play
        `.prep_training_history()`: Take a training history, apply filters, and potentially set a column of "sample_weight"
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
            :param np.ndarray moveChoice_vector: Numpy array of length equal to the number of spaces on the board. It's a list of weights, where it must be `<=0.0` for illegal moves.
            :param np.random.Generator rng: Random number generator which may be used
            :returns: Index of move choice
            :rtype: int
            
            Get the index of a choice by one way or another. Different implementations might use max, or soft max among those positive. Use ``hexathello.engine.HexagonGridHelper`` to turn it back to a `(q,r)` tuple.
                
            Default implementation is `np.argmax( moveChoice_vector)`.
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
        
        return history.get_relativeStateVector(
            boardState_vector = boardState_vector,
            player_id = self.player_id,
            player_count = self.player_count
        )
    #/def getBoardState_asRelativeStateVector
    
    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        """
            :param np.ndarray moveChoice_vector: Numpy array of length equal to the number of spaces on the board. It's a list of weights, where it must be `<=0.0` for illegal moves.
            :param int turn_index: Current turn, used to turn boardState to pov via `.get_relativeStateVector()` if necessary
            :param np.random.Generator rng: Random number generator which may be used
            :returns: Dictionary of the chosen move
            :rtype: engine.PlayerMove
        """
        import tensorflow as tf
        #
        assert self.player_id is not None
        moveChoiceDict: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        qr: engine.QRTuple
        action_tags: list[ str ] = []
        if len( moveChoiceDict ) == 1:
            qr = next( _qr for _qr in moveChoiceDict.keys() )
            action_tags.append('forced')
        #
        # Check if random play
        elif self.p_random >= 1:
            # Guaranteed random
            qr = _random_play(
                moveChoiceDict,
                rng = rng
            )
            action_tags.append('random')
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
                    np.reshape(
                        boardState_vector,
                        newshape = (1, len( boardState_vector ) )
                    )
                )
                
                # Mask it with legal moves
                
                assert len( moveChoiceDict ) > 0
                legal_moves_vector: np.ndarray = np.zeros(
                    shape = (len( boardState ),),
                    dtype = float
                )
                
                for qr in moveChoiceDict:
                    legal_moves_vector[
                        self.hexagonGridHelper.index_from_qr_tuple(
                            qr
                        )
                    ] = 1.0
                #/for qr in moveChoiceDict

                # Distribution of choices might have been given; make choice, potentially probabilistically
                # This result is a literal spot index, not relative to anyone
                moveChoice_vector_masked: np.ndarray = moveChoice_vector * legal_moves_vector
                moveChoice_final: int = self.chooseMove(
                    moveChoice_vector_masked,
                    rng = rng
                )
                
                if np.all(
                    np.isclose(
                        moveChoice_vector[ moveChoice_final ], moveChoice_vector_masked
                    ):
                        # Could not effectively make a choice; go random
                        qr = _random_play(
                            moveChoiceDict,
                            rng = rng
                        )
                    #
                )
                else:
                    qr: engine.QRTuple = self.hexagonGridHelper.qr_from_index(
                        moveChoice_final
                    )
                #/if np.all( np.isclose( { move choices } ) )
                
                if qr not in moveChoiceDict:
                    print( moveChoice_vector )
                    print( moveChoice_vector_masked )
                    print( moveChoice_final )
                    print( moveChoiceDict )
                    print( qr )
                    raise Exception("Bad QR")
                #
                
                assert qr in moveChoiceDict
                action_tags.append('brain')
            #
            else:
                # Rolled random
                qr = _random_play(
                    moveChoiceDict,
                    rng = rng
                )
                action_tags.append('random')
            #/if { nonrandom }/else
        #/if len( moveChoiceDict ) == 1/switch { random }
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id,
            "action_tags": action_tags
        }
    #/def getMove_fromBoardState
    
    def prep_training_history(
        self: Self,
        game_history: jable.JyFrame
        ) -> jable.JyFrame:
        """
            :param jable.JyFrame game_history: A history already transformed to pov. See :doc:`history`
            :returns: A full history to be used as training data, with desired outcome as the "player_action" and input "boardState"
            :rtype: jable.JyFrame
            
            Turn game_history into something we can actually learn from. Subclasses can use other methods of filtering or setting "sample_weight"
            
            The default behavior is to apply a sample weight of `1.0` to moves leading to winning games, `-1/{game_spaces}` to moves leading to a losing game, and `0.0` for ties
        """
        assert game_history.get_fixed( "history_type" ) == 'pov'
        
        # Get sample_weights
        loss_weight: float = -1/len( game_history[0,"player_action"] )
        sample_weight: list[ float ] = [
            1.0 if row["winner"] == 0 else 0.0 if row["winner"] is None else loss_weight
                for row in game_history
        ]
        
        history_prepped: JyFrame = jable.copyJyFrame( game_history )
        history_prepped.addColumn(
            col = "sample_weight",
            values = sample_weight,
            dtype = float
        )
        
        return history_prepped
    #/def prep_training_history
    
    def train(
        self: Self,
        game_history: jable.JyFrame,
        *args,
        **kwargs
        ) -> None:
        """
            :param jable.JyFrame game_history: A literal or pov set of histories
            :param list args: Args passed to ``self.brain.fit()``
            :param dict kwargs: Key word args passed to ``self.brain.fit()``
            
            Train the brain with a Pov History JyFrame; see :doc:`history`. If "sample_weight" is a column with `float` values after running `.prep_training_history()`, it gets used as "sample_weight", unless it's provided as a `kwarg`.
            
            For `args` and `kwargs` consider:
            - epochs: int
            - sample_weight: np.ndarray
            
            See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
            
        """
        
        if game_history.get_fixed( "history_type" ) == 'pov':
            ...
        #
        elif game_history.get_fixed( "history_type" ) == 'literal':
            game_history = history.povHistory_from_literalHistory(
                game_history
            )
        #
        else:
            raise Exception(
                "Unrecognized game_history.keys()={}".format(
                    game_history.keys()
                )
            )
        #
        
        # Prep history
        history_prepped: jable.JyFrame = self.prep_training_history(
            game_history
        )
        
        X: np.ndarray = np.array(
            history_prepped[ "board_state" ]
        )
        
        y: np.ndarray = np.array(
            history_prepped[ "player_action" ]
        )
        
        if "sample_weight" in history_prepped.keys() and "sample_weight" not in kwargs:
            kwargs["sample_weight"] = np.array(
                history_prepped["sample_weight"]
            )
        #
        
        self.brain.fit( X, y, *args, **kwargs )
        
        return
    #/def train
#/class KerasHexAgent

class GreedyHexAgent( HexAgent ):
    """
        Picks randomly from among moves which give the most immediate captures. You likely will not subclass `GreedyHexAgent`. Initialize with `p_random = 1.0` for a fully random agent; set it somewhere `0.0 < p_random < 1.0` for sometimes random sometimes greedy choices; this is good for beginner AIs and for generating a lot of move data to bootstrap the ``KerasHexAgent`` training process.
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
        action_tags: list[ str ] = []
        # Check if only one option
        if len( moveChoiceDict ) == 1:
            qr = next( _qr for _qr in moveChoiceDict.keys() )
            action_tags.append('forced')
        #
        # Check if random play
        elif self.p_random >= 1:
            # Guaranteed random
            qr = _random_play(
                moveChoiceDict,
                rng = rng
            )
            action_tags.append('random')
        #
        else:
            # Roll for random or regular decision
            if self.p_random <= 0 or rng.choice(
                2,
                p = ( 1-self.p_random, self.p_random )
            ) == 0:
                # Did not roll ranndom
                qr = _greedy_play(
                    moveChoiceDict,
                    rng = rng
                )
                action_tags.append('greedy')
            #
            else:
                # Rolled random
                qr = _random_play(
                    moveChoiceDict,
                    rng = rng
                )
                action_tags.append('random')
            #/if { nonrandom }/else
        #/if len( moveChoiceDict ) == 1/switch { random }
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id,
            "action_tags": action_tags
        }
    #/def getMove_fromBoardState
#/class GreedyHexAgent( HexAgent )
