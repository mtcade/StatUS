"""
    History is a type of `jable.JyFrame` which records the actions taken in a game. The most common source is ``hexathello.autoPlayer.runHexathello_withAgents()``, often saved to disk.
    
    This is the fundamental data used to train AI agents from ``hexathello.aiPlayers``, primarily the ``hexathello.aiPlayers.KerasHexAgent`` class and subclasses.
    
    See the `PyJable` package: [https://mtcade.github.io/Jable/html/index.html](https://mtcade.github.io/Jable/html/index.html), notably the `PyJable.jable.JyFrame` class.
    
    There are three kinds of history tables:
    
    * Literal: `table.get_fixed["history_type"] = 'literal'`. Indexes each player from 0 to the number of players minus one. The board state is a `numpy.ndarray` of `0.0` and `1.0` of concatentated `player_count` tuples, indexed by which player owns that spot. For indexing a linear index to hexagon (q,r) coordinates and back, see the ``hexathello.engine.HexagonGridHelper`` class.
        * Fixed Keys
            * player_count: `int`
            * size: `int`
            * history_type: `Literal['pov']`
            * winner: `int|None`
            * scores: `list[ int ]`
        * Shift Keys
            * turn_index: `int`
            * current_player: `int`
            * board_state `numpy.ndarray`
            * action_choices: `numpy.ndarray`
            * player_action: `numpy.ndarray`
        * Shift Index Keys
            * ai_id: `str`
            * action_tags: `list[ str ]`
    * Point of View: `table.get_fixed["history_type"] = 'pov'`. At each turn, indexes each tuple in `board_state` with the player whose turn it is rather than their numeric id.
        * Fixed Keys
            * player_count: `int`
            * size: `int`
            * history_type: `Literal['pov']`
        * Shift Keys
            * turn_index: `int`
            * current_player: `int`
            * board_state `numpy.ndarray`
            * action_choices: `numpy.ndarray`
            * player_action: `numpy.ndarray`
            * winner: `int|None`
            * scores: `list[ int ]`
        * Shift Index Keys
            * ai_id `str`
            * action_tags: `list[ str ]`
    * Disk: encodes the binary vectors of a pov table as integers. To decode, pad with the appropriate number of zeroes on the left; vector length can be inferred from `board_size` and `player_count`.
        * Fixed Keys
            * player_count: `int`
            * size: `int`
            * history_type: `Literal['literal']`
            * winner: `int|None`
            * scores: `list[ int ]`
        * Shift Keys
            * turn_index: `int`
            * current_player: `int`
            * board_state `int`
                * `= _state_asInt( np.ndarray )`
            * action_choices: `int`
                * `= _state_asInt( np.ndarray )`
            * player_action: `int`
                * `= _state_asInt( np.ndarray )`
        * Shift Index Keys
            * ai_id: `str`
            * action_tags: `list[ str ]`
"""

import hexathello.engine as engine
import hexathello.jable as jable

import numpy as np

def get_relativeStateVector(
    boardState_vector: np.ndarray,
    player_id: int,
    player_count: int
    ) -> np.ndarray:
    """
        :param np.ndarray boardState_vector: Literal board state, with each `player_count`-tuple indexed by `player_id`
        :param int player_id: Player from whom to make the PoV from
        :param int player_count: Number of players in game
        
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
    scores: list[ int ] = []
    ) -> jable.JyFrame:
    """
        :param int player_count: Number of players
        :param int size: Size of one side of the hex board
        :param int|None winner: Which player_id won the game. `None` if a tie.
        :param list[ int ] scores: Final score at the end of the game.
        :returns: Empty history table with "history_type" = 'literal'
        :rtype: jable.JyFrame
    """
    if scores == []:
        # Default: inner ring divided by players
        scores = [ 6//player_count ]*player_count
    #
    
    return jable.fromHeaders(
        fixed = {
            "player_count": player_count,
            "size": size,
            "history_type": 'literal',
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
            "ai_id",
            "action_tags"
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
    size: int
    ) -> jable.JyFrame:
    """
        :param int player_count: Number of players
        :param int size: Size of one side of the hex board
        :returns: Empty history table with "history_type" = 'pov'
        :rtype: jable.JyFrame
        
        Like literal history, but the winner and scores will change to reflect the pov player, assumed to be 0. "current_player" gets preserved in case we want to go back to literal
    """
    
    return jable.fromHeaders(
        fixed = {
            "player_count": player_count,
            "size": size,
            "history_type": 'pov'
        },
        shiftHeader = [
            "turn_index",
            "current_player",
            "board_state",
            "action_choices",
            "player_action",
            "action_tags",
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
        :param jable.JyFrame literalHistory: Literal history, with the board state and choice indexed by player_id
        :returns: JyFrame with everything shifted from the point of view of the player making the move
        :rtype: jable.JyFrame
        
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
        
        # TEST
        if True:
            povHistory.append(
                row | {
                    "history_type": 'pov',
                    "board_state": get_relativeStateVector(
                        row["board_state"],
                        player_id = row["current_player"],
                        player_count = literalHistory.get_fixed("player_count")
                    ),
                    "scores": np.roll( row["scores"], -1*row["current_player"] ).tolist(),
                    "winner": winner
                }
            )
        #
        else:
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
        #
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
        :param jable.JyFrame history:
        :returns: Decoded JyFrame with binary np.ndarray columns
        :rtype: jable.JyFrame
        
        Encode the "board_state", "action_choices", and "player_action" as integers from binary np.ndarrays, making a 'disk' history.
    """
    history_out: jable.JyFrame = jable.likeJyFrame(
        history
    )
    
    # TEST
    for row in history:
        if True:
            history_out.append(
                row | {
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
        else:
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
        #
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
        :param jable.JyFrame history:
        :returns: History, encoding turning appropriate columns into np.ndarrays
        :rtype: jable.JyFrame
        
        Decode the "board_state", "action_choices", "player_action" columns into binary np.ndarrays
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
        # TEST
        if True:
            history_out.append(
                row | {
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
        #
        else:
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
        #
    #/for row in history
    history_out._keyTypes["board_state"] = np.ndarray
    history_out._keyTypes["action_choices"] = np.ndarray
    history_out._keyTypes["player_action"] = np.ndarray
    
    return history_out
#/def history_fromInt
