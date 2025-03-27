"""
    Interface to play games (of Hexathello) and show the results
"""

from . import engine, jable
import numpy as np

from typing import Literal, Protocol, Self

# -- AI Agents Helpers
def get_relativeStateVector(
    boardState_vector: np.ndarray,
    player_id: int,
    player_count: int
    ) -> np.ndarray:
    """
        Convert a board state to be from the point of view of the given player; if they own a spot, the 0th index in the player_count tuple will be 1.
    """
    assert player_id < player_count
    # For player 0, it's already the right POV
    if player_id == 0:
        return boardState_vector
    #
    
    _cursor: int = 0
    relativeStateVector: np.ndarray = np.zeros(
        shape = boardState_vector.shape,
        dtype = float
    )
    
    while _cursor + player_count <= len( boardState_vector ):
        
        relativeStateVector[
            _cursor: (_cursor+player_count)
        ] = np.roll(
            boardState_vector[
                _cursor: (_cursor+player_count)
            ],
            shift = -1*player_id
        )
        _cursor += player_count
    #
    return relativeStateVector
#/def get_relativeStateVector

# -- History: a JyFrame for learning

def new_literalHistory(
    player_count: int,
    size: int,
    winner: int | None = None,
    scores: list[ int ] = [],
    history_type: Literal['literal'] = 'literal'
    ) -> jable.JyFrame:
    
    if scores == []:
        # Default: inner ring divided by players
        scores = [ 6//player_count ]*player_count
    #
    
    return jable.fromHeaders(
        fixed = {
            "player_count": player_count,
            "size": size,
            "history_type": history_type,
            "winner": winner,
            "scores": scores
        },
        shiftHeader = [
            "turn_index",
            "current_player",
            "board_state",
            "action_choices",
            "player_action"
        ],
        shiftIndexHeader = [
            "ai_id"
        ],
        keyTypes = {
            "board_state": np.ndarray,
            "action_choices": np.ndarray,
            "player_action": np.ndarray
        }
    )
#/def new_literalHistory

def new_povHistory(
    player_count: int,
    size: int,
    history_type: Literal['pov'] = 'pov'
    ) -> jable.JyFrame:
    """
        Like literal history, but the winner and scores will change to reflect the pov player, assumed to be 0. "current_player" gets preserved in case we want to go back to literal
    """
    
    return jable.fromHeaders(
        fixed = {
            "player_count": player_count,
            "size": size,
            "history_type": history_type
        },
        shiftHeader = [
            "turn_index",
            "current_player",
            "board_state",
            "action_choices",
            "player_action",
            "scores",
            "winner"
        ],
        shiftIndexHeader = [
            "ai_id"
        ],
        keyTypes = {
            "board_state": np.ndarray,
            "action_choices": np.ndarray,
            "player_action": np.ndarray
        }
    )
#/def new_povHistory

def povHistory_from_literalHistory(
    literalHistory: jable.JyFrame
    ) -> jable.JyFrame:
    """
        Shift to make as if each move were from player 0's point of view. Shifts:
        
            - scores
            - board_state
            - winner
           
        Preserves "current_player". "player_action" needn't change since it's a choice of literal space
        
        
        
        And gives 'history_type' = 'pov'
    """
    povHistory: JyFrame = new_povHistory(
        player_count = literalHistory.get_fixed("player_count"),
        size = literalHistory.get_fixed("size")
    )
    
    literalWinner: int | None
    winner: int | None
    for row in literalHistory:
        literalWinner = row["winner"]
        if literalWinner is None:
            winner = None
        else:
            winner = (
                literalWinner - row["current_player"]
            ) % literalHistory.get_fixed("player_count")
        #/if literalWinner is None/else
        
        povHistory.append({
            "turn_index": row["turn_index"],
            "current_player": row["current_player"],
            "board_state": get_relativeStateVector(
                row["board_state"],
                player_id = row["current_player"],
                player_count = literalHistory.get_fixed("player_count")
            ),
            "action_choices": row["action_choices"],
            "player_action": row["player_action"],
            "scores": np.roll( row["scores"], -1*row["current_player"] ).tolist(),
            "winner": winner,
            "ai_id": row["ai_id"]
        })
    #/for row in literalHistory
    
    return povHistory
#/def povHistory_from_literalHistory

# -- History Serialization

def _state_asInt(
    state: np.ndarray
    ) -> int:
    # Convert numpy to a bool list
    state_bool: np.ndarray = state.astype( bool ).tolist()
    
    # Convert to binary
    state_int: bin = int(
        ''.join( str(int(_val)) for _val in state_bool ),
        2
    )
    return state_int
#/def _state_asBase64

def _state_fromInt(
    state: int,
    length: int
    ) -> np.ndarray:
    # Convert to binary string
    state_bin: str = bin( state )[2:]
    state_bin_full: str = state_bin.rjust( length, '0' )
    
    # Convert to list of '0', '1'
    state_listStr: list[ str ] = list( state_bin_full )
    
    state_bool: list[ bool ] = [ bool( int(val) ) for val in state_listStr ]
    
    
    state_np: np.ndarray = np.array( state_bool, dtype = float )
    
    return state_np
#/def _state_fromInt

def history_asInt(
    history: jable.JyFrame
    ) -> jable.JyFrame:
    """
        Encode the "board_state" and "player_action" as integers
    """
    
    history_out: jable.JyFrame = jable.likeJyFrame(
        history
    )
    
    for row in history:
        old_fields: dict[ str, any ] = {
            key: val for key, val in row.items() if key not in [
                "board_state","action_choices","player_action"
            ]
        }
        history_out.append(
            old_fields | {
                "board_state": _state_asInt(
                    row["board_state"]
                ),
                "action_choices": _state_asInt(
                    row["action_choices"],
                ),
                "player_action": _state_asInt(
                    row["player_action"]
                )
            }
        )
    #/for row in history
    
    history_out._keyTypes["board_state"] = int
    history_out._keyTypes["action_choices"] = int
    history_out._keyTypes["player_action"] = int
    
    return history_out
#/def history_asInt

def history_fromInt(
    history: jable.JyFrame
    ) -> jable.JyFrame:
    """
        Decode the "board_state" and "player_action" into numpy arrays
    """
    
    history_out: jable.JyFrame = jable.likeJyFrame(
        history
    )
    
    # Figure out our deserialization length
    board_size: int = engine.get_spaceCount_forSize(
        size = history.get_fixed("size")
    )
    state_length: int = board_size*history.get_fixed("player_count")
    
    for row in history:
        old_fields: dict[ str, any ] = {
            key: val for key, val in row.items() if key not in [
                "board_state","player_action"
            ]
        }
        history_out.append(
            old_fields | {
                "board_state": _state_fromInt(
                    row["board_state"],
                    length = state_length
                ),
                "action_choices": _state_fromInt(
                    row["action_choices"],
                    length = board_size
                ),
                "player_action": _state_fromInt(
                    row["player_action"],
                    length = board_size
                )
            }
        )
    #/for row in history
    history_out._keyTypes["board_state"] = np.ndarray
    history_out._keyTypes["action_choices"] = np.ndarray
    history_out._keyTypes["player_action"] = np.ndarray
    
    return history_out
#/def history_fromBase64

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

# -- AI Agents

class HexAgent():
    """
        Agents which play hexathello
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None,
        ai_id: str = ''
        ) -> None:
        self.size = size
        self.player_count = player_count
        self.player_id = player_id
        self.ai_id = ai_id
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
        brain: PredictionModel | None = None,
        hexagonGridHelper: engine.HexagonGridHelper | None = None,
        player_id: int | None = None,
        ai_id: str = ''
        ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id,
            ai_id = ai_id
        )
        self.brain = brain
        
        if hexagonGridHelper is None:
            self.hexagonGridHelper = engine.HexagonGridHelper(
                size = size,
                player_count = player_count
            )
        else:
            self.hexagonGridHelper = hexagonGridHelper
        #
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
    #/getBoardState_asStateVector
    
    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        # Turn moves into a vector suitable for brain
        boardState_vector: np.ndarray = self.hexagonGridHelper.stateVector_from_boardState(
            boardState
        )
        
        # Use our brain to choose the best move
        moveChoice_vector: np.ndarray = self.brain.predict(
            boardState_vector
        )
        
        # Mask it with legal moves
        moves: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        assert len( moves ) > 0
        legal_moves_vector: np.ndarray = np.zeros(
            shape = (len( moveChoice_vector ),),
            dtype = float
        )
        for qr in moves:
            moves_vector[
                self.hexagonGridHelper.index_from_qr_tuple(
                    qr
                )
            ] = 1.0
        #/for qr in moves
        
        
        # Distribution of choices might have been given; make choice, potentially probabilistically
        moveChoice_final: int = self.choseMove(
            moveChoice_vector * legal_moves_vector
        )
        moveChoice_qr: engine.QRTuple = self.hexagonGridHelper.qr_from_index[
            moveChoice_final
        ]
        
        assert moveChoice_qr in moves
        
        return {
            "turn_index": turn_index,
            "q": moveChoice_qr[0],
            "r": moveChoice_qr[1],
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

class RandomHexAgent( HexAgent ):
    """
        Picks randomly from legal moves
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None,
        ai_id: str = 'RandomHexAgent'
        ) -> None:
        
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id,
            ai_id = ai_id
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
            Choose randomly from legal moves
        """
        assert self.player_id is not None
        moveChoiceDict: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        qr: engine.QRTuple = _random_play(
            moveChoiceDict = moveChoiceDict,
            rng = rng
        )
        
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
        player_id: int | None = None,
        ai_id: str = 'GreedyHexAgent'
        ) -> None:
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id,
            ai_id = ai_id
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
        qr: engine.QRTuple = _greedy_play(
            moveChoiceDict
        )
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class GreedyHexAgent( HexAgent )

class GreendomHexAgent( HexAgent ):
    """
        With parameter `p`, choose between the greedy choice or the random choice. Hence, sometimes Greedy sometimes Random, so Greendom. Chooses greedy with probability `p`
        
        Try to name with p, but with a '-' instead of decimal '.', like
        `p = 0.5` -> `ai_id = 'GreendomHexAgent_0-5'`. This is automatic if you do not specify `ai_id`
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int,
        player_id: int | None = None,
        ai_id: str = 'GreendomHexAgent',
        p: float = 0.5
        ) -> None:
        
        _ai_id: str
        if ai_id == 'GreendomHexAgent':
            _ai_id = '{}_{}'.format(
                ai_id, str( p ).replace('.','-')
            )
        #
        else:
            _ai_id = ai_id
        #/switch ai_id
        
        super().__init__(
            size = size,
            player_count = player_count,
            player_id = player_id,
            ai_id = _ai_id
        )
        
        self.p = p
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
            Choose randomly from legal moves
        """
        assert self.player_id is not None
        moveChoiceDict: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = self.player_id,
            boardState = boardState,
            potential_moves = potential_moves
        )
        
        # Randomly choose whether to be greedy with probability p
        qr: engine.QRTuple
        if rng.choice(
            2,
            p = ( 1-self.p, self.p )
        ):
            qr = _greedy_play(
                moveChoiceDict = moveChoiceDict,
                rng = rng
            )
        #
        else:
            qr = _random_play(
                moveChoiceDict = moveChoiceDict,
                rng = rng
            )
        #/if { choose greedy }
        
        return {
            "turn_index": turn_index,
            "q": qr[0],
            "r": qr[1],
            "owner": self.player_id
        }
    #/def getMove_fromBoardState
#/class GreendomHexAgent

def runHexathello_withAgents(
    agents: list[ HexAgent ],
    size: int,
    logging_level: int = 0,
    rng: np.random.Generator | None = None,
    hexagonGridHelper: engine.HexagonGridHelper | None = None
    ) -> jable.JyFrame:
    """
        Plays a game with set AI and prints everything as it goes
        
        agents: An initialized set of agents to play
        
        Returns the history as a JyFrame
        
        fixed:
            player_count: int
            size: int
            winner: int | None
            scores: list[ int ]
        shift:
            turn_index: int
            current_player: int
            boardState: np.ndarray
            player_action: np.ndarray
    """
    from copy import deepcopy
    
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
    
    if hexagonGridHelper is None:
        hexagonGridHelper = engine.HexagonGridHelper(
            size = size,
            player_count = player_count
        )
    #
    
    # Initialize status for the Hexathello, with random first player
    player_start: int = rng.choice( player_count )
    hexathello = engine.new_hexathello(
        player_count = player_count,
        size = size,
        player_start = player_start,
        logging_level = logging_level
    )
    
    
    # Initialize game history
    history: jable.JyFrame = new_literalHistory(
        player_count = player_count,
        size = size,
        winner = None,
        scores = deepcopy( hexathello.status["scores"] ),
        history_type = 'literal'
    )
    
    # SAFETY VALVE: max number of turns is empty_count
    move_log: list[ engine.QRTuple ] = []
    empty_count: int = hexathello.status["empty_count"]
    for tick_index in range( empty_count ):
        turn_index: int = hexathello.status["turn_index"]
        
        if turn_index > 0:
            assert hexathello.boardState[ move_log[ turn_index-1] ]["owner"] is not None
        #
        
        next_player_index = hexathello.status["current_player"]
        moves: engine.MoveChoiceDict = engine.getMoves_forPlayer(
            player = next_player_index,
            boardState = hexathello.boardState,
            potential_moves = hexathello.potential_moves
        )
        action_choices: np.ndarray = np.zeros(
            shape = ( len(hexathello.boardState), ),
            dtype = float
        )
        for qr in moves:
            action_choices[
                hexagonGridHelper.index_from_qr_tuple( qr )
            ] = 1.0
        #/for qr in moves
        
        next_move: engine.PlayerMove = agents[
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
            next_qr
        )
        
        # Update history
        history.append(
            {
                "turn_index": turn_index,
                "current_player": next_player_index,
                "ai_id": agents[next_player_index].ai_id,
                "board_state": hexagonGridHelper.stateVector_from_boardState(
                    hexathello.boardState
                ),
                "action_choices": action_choices,
                "player_action": hexagonGridHelper.moveVector_from_play(
                    qr = next_qr
                )
            }
        )
        
        # Run the one update
        hexathello.queueUpdate( next_move )
        hexathello.applyUpdates()
        
        # Print the log
        while not hexathello.log.empty():
            logUpdate: dict = hexathello.log.get()
            engine.print_logUpdate( logUpdate )
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
            
            # Set history
            history["winner"] = hexathello.status["winner"]
            history["scores"] = deepcopy( hexathello.status["scores"] )
            break
        #/if hexathello.status["game_complete"]
    #/for _ in range( empty_count )
    return history
#/def runHexathello_withAgents
