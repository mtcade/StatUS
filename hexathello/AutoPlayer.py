"""
    Interface to play games (of Hexathello) and show the results
"""

from . import engine, jable
import numpy as np

from typing import Literal, Protocol, Self
from abc import abstractmethod

class AiAgentProtocol( Protocol ):
    """
        Protocol for aiPlayers.HexAgent
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
        raise NotImplementedError
    #

    def getMove_fromBoardState(
        self: Self,
        boardState: engine.BoardState,
        turn_index: int,
        rng: np.random.Generator,
        potential_moves: list[ engine.CellCapture ] = []
        ) -> engine.PlayerMove:
        raise NotImplementedError
    #

    @property
    @abstractmethod
    def player_id( self: Self ) -> int | None:
        raise NotImplementedError
    #
    
    @property
    @abstractmethod
    def ai_id( self: Self ) -> str:
        raise NotImplementedError
    #
#/class AiAgentProtocol

# -- History: a JyFrame for learning

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

def runHexathello_withAgents(
    agents: list[ AiAgentProtocol ],
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
