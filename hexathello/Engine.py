"""
    The Hexathello Engine enforces the rules of the game. autoPlayer contains an interface for getting AI agents to play it. The `game` interface allows it to be used for a human interactable app, with interface and events.
"""
import hexathello.jable as jable

from typing import Literal, Optional, Protocol, Self, TypedDict
import queue

import numpy as np

# -- Types
# -- A spot on the board

# CellStatus: represents a single spot on the board
# You CAN make an update with it to empty a spot but that shouldn't happen in game
class CellStatus( TypedDict ):
    """
        A single spot on the board you CAN make an update with it to empty a spot but that shouldn't happen in game.
    """
    q: int
    r: int
    occupied_adjacent: int
    owner: int | None
#/class CellStatus


# CellCapture:
class CellCapture( TypedDict ):
    """
        Changes another spot; can in theory be removing an owner by setting it to `None`
    """
    q: int
    r: int
    owner: int | None
#/class CellCapture

# PlayerMove: A play on a location location a certain player makes on a certain turn.
# Essentially an interface action.
# This does not get used as an update; rather, a series of updates, including at least one capture, will follow by generating some CellCapture.
# Since it's a move it must have an owner, not None
class PlayerMove( TypedDict ):
    """
        A play on a location location a certain player makes on a certain turn. Essentially an interface action.
        
        This does not get used as an update; rather, a series of updates, including at least one capture, will follow by generating some CellCapture, based on the board state.
        
        Since it's a move it must have an owner, not None
    """
    turn_index: int
    q: int
    r: int
    owner: int
    action_tags: list[ str ]
#/class PlayerMove

# Simple 2-d hex coordinate
QRTuple = tuple[ int, int ]

# Clockwise adjaceny vectors starting to the east
#  in a pointy top system
CLOCKWISE_UNIT_VECTOR_LIST: list[ QRTuple ] = [
    (1,0), (0,1), (-1,1),
    (-1,0), (0,-1), (1,-1)
]

# List of legal moves for a player, mapped from QRTuple (axial) to a list of
#    captures
MoveChoiceDict = dict[
    QRTuple,
    list[ CellCapture ]
]

# BoardState: (q,r) (QRTuple) axial coordinate to CellStatus

BoardState = dict[
    QRTuple,
    CellStatus
]
"""
    The board state is a dictionary from board states to the status of cells; this prevents us from having to search through a table of coordinates, since we will know the dictionary keys ahead of time. A simple example is the degenerate game of size 2
    
    [0,0,1,0,... ]
    
    [1,0,...]
    
    {
        (0,0): {
            "q": 0,
            "r": 0,
            "occupied_adjacent": 6,
            "owner": None
        },
        (1,0):{
            "q": 1,
            "r": 0,
            "occupied_adjacent": 2,
            "owner": 0
        },
        (0,1):{
            "q": 0,
            "r": 1,
            "occupied_adjacent": 2,
            "owner": 1
        },
        (-1,1):{
            "q": -1,
            "r": 1,
            "occupied_adjacent": 2,
            "owner": 0
        },
        (-1,0):{
            "q": -1,
            "r": 0,
            "occupied_adjacent": 2,
            "owner": 1
        },
        (0,-1):{
            "q": 0,
            "r": -1,
            "occupied_adjacent": 2,
            "owner": 0
        },,
        (1,-1):{
            "q": 1,
            "r": -1,
            "occupied_adjacent": 2,
            "owner": 1
        }
    }
"""

def print_boardState(
    boardState: BoardState,
    qr_list: list[ QRTuple ] = []
    ) -> None:
    # Row by row printing: {(q,r)}: {owner}
    if qr_list == []:
        qr_list = list( boardState.keys() )
    #
    
    for qr in qr_list:
        print(
            "{}: {}".format(
                qr, boardState[ qr ]["owner"]
            )
        )
    #/for qr, cellStatus in boardState.items()
    return
#/print_boardState( boardState: BoardState )

def get_emptyCount( boardState: BoardState ) -> int:
    empty_count: int = 0
    for cellStatus in boardState.values():
        if cellStatus["owner"] is None:
            empty_count += 1
        #/if cellStatus["owner"] is None
    #/for cellStatus in boardState.values()
    return empty_count
#def get_emptyCount

def adjacent_spaces( qr: QRTuple ) -> list[ QRTuple ]:
    """
        Get all possible adjacent spaces; this includes those which might not be
        in a game, so check that they are in it
        
        Gives in clockwise order starting with q+1
    """
    return [
        (qr[0] + vector[0], qr[1] + vector[1] )\
            for vector in CLOCKWISE_UNIT_VECTOR_LIST
    ]
#/def adjacent_spaces

def adjacent_occupied_count(
    qr: QRTuple,
    boardState: BoardState
    ) -> int:
    adjacent_occupied: int = 0
    for qr_adjacent in adjacent_spaces( qr ):
        if qr_adjacent not in boardState:
            continue
        #
        if boardState[ qr_adjacent ]["owner"] is not None:
            adjacent_occupied += 1
        #
    #/for qr_adjacent in adjacent_spaces( qr )
    return adjacent_occupied
#/def adjacent_occupied_count

def get_potential_moves( boardState: BoardState ) -> list[ QRTuple ]:
    """
        Hard calculation of all spaces which are empty but adjacent to a nonempty space; some might not be legal plays for any player but that's ok
    """
    return [
        qr for qr, cellStatus in boardState.items()\
            if cellStatus["owner"] is None\
            and cellStatus["occupied_adjacent"] > 0
    ]
#/get_potential_moves

def get_scores(
    boardState: BoardState,
    player_count: int
    ) -> list[ int ]:
    scores: list[ int ] = [0]*player_count
    
    for cellStatus in boardState.values():
        if cellStatus["owner"] is not None:
            scores[ cellStatus["owner"] ] += 1
        #
    #
    return scores
#/def get_scores

def getCaptures_forMove(
    update: CellCapture,
    boardState: BoardState
    ) -> list[
        CellCapture
    ]:
    """
        Check in the vector of each of 6 directions for the flank
        
        Get the list of captures made, by (q,r)
    """
    qr = ( update["q"], update["r"] )
    assert boardState[qr]["owner"] is None
    capture_list: list[
        CellCapture
    ] = []
    
    # Check each of 6 vectors
    # Keep track of each vector's _capture_list.
    # If it completes a flank, extend capture_list with it
    _qr: QRTuple
    _capture_list: list[
        CellCapture
    ]
    for vector in CLOCKWISE_UNIT_VECTOR_LIST:
        _qr = (
            qr[0] + vector[0],
            qr[1] + vector[1]
        )
        _capture_list = []
        while _qr in boardState:
            if boardState[ _qr ]["owner"] is None:
                # Not a flank, we have gotten to a blank space without a capture
                # Go to next vector
                break
            #/if self.boardState[ _qr ]["owner"] is None
            
            if boardState[ _qr ]["owner"] == update["owner"]:
                # Found a potential flank
                # Capture everything (which could be an empty list )
                # This SHOULD extend with a copy of _capture_list
                capture_list.extend( _capture_list )
                break
            #/is self.boardState[ _qr ]["owner"] == update["owner"]
            
            if boardState[ _qr ]["owner"] != update["owner"]:
                # Different owner, it's a potential capture
                _capture_list.append(
                    {
                        "q": _qr[0],
                        "r": _qr[1],
                        "owner": update["owner"]
                    }
                )
            #/if self.boardState[ _qr ]["owner"] != update["owner"]
            
            # Move to next spot on vector
            _qr = (
                _qr[0] + vector[0],
                _qr[1] + vector[1]
            )
        #/while _qr in self.boardState
    #/for vector in {...}
    return capture_list
#/def getCaptures_forMove

def getMoves_forPlayer(
    player: int,
    boardState: BoardState,
    potential_moves: list[ CellCapture ] = []
    ) -> MoveChoiceDict:
    """
        Get a dictionary of viable moves for a player, where the key is the qr, and the values are (nonempty) lists of captures for that player
        
        Can return an empty dict if there are no capturing moves for the player
        
        potential_moves: a list of (q,r) which are considered for a move, in that they are empty and adjacent to a nonempty spot. Hexathello keeps track of this, but we can consider every move on the board if it's a problem; spots with only empty spots around them (always illegal) will find an empty spot in each direction and thus not register as having any captures anway.
    """
    assert player is not None
    
    if potential_moves == []:
        # no potential moves provided; check all spots which are empty
        potential_moves = [
            qr for qr, cellStatus in boardState.items()\
                if cellStatus["owner"] is None
        ]
    #/if potential_moves == []
    
    moves: MoveChoiceDict = {}
    
    captures: list[
        CellCapture
    ]
    
    for qr in potential_moves:
        captures = getCaptures_forMove(
            update = {"q": qr[0], "r": qr[1], "owner": player},
            boardState = boardState
        )
        if len( captures ) > 0:
            moves[ qr ] = captures
        #/if len( captures ) > 0
    #/for qr in self.potential_moves
    
    return moves
#/def getMoves_forPlayer


# -- Hexathello helpers
class HexagonGridHelper():
    """
        :param int size: Length of one side of the Hexagonal Grid
        :param int player_count: Number of players. Restricted to 2, 3, or 6 so that there can be a valid starting position, with the middle 2-ring occupied.
        
        Tool for working with the hexagonal grid, such as converting coordinates, including some game logic, and ability to get things for machine learning, such as the game state as a one hot encoded vector
                
        
        
        qr_to_index: Take a coordinate (q,r) on hexagon grid, return the one hot encoded index, in an array with length equal to number of hex spots
        
        index_to_qr: Take index in the one hot encoded list and return the (q,r) index
        
        NOTE: outputting the state will give a list of length size*player_count
    """
    def __init__(
        self: Self,
        size: int,
        player_count: int
        ) -> None:
        
        self.size = size
        self.player_count = player_count
        self.qr_to_index: dict[ QRTuple, int ] = {}
        self.index_to_qr: list[ QRTuple ] = []
        
        r_min: int
        r_max: int
        qr: QRTuple
        i: int = 0
        for q in range( 1-size, size, 1 ):
            r_min = max(1-q - size, 1-size )
            r_max = min( 1-q + size, size-1-q, size - 1 )
            
            for r in range( r_min, r_max + 1, 1 ):
                qr = ( q, r )
                self.qr_to_index[ qr ] = i
                self.index_to_qr.append( qr )
                i += 1
            #/for r in range( r_min, r_max + 1, 1 )
        #/for q in range( -1*size, size + 1 , 1 )
        
        self.length = i
        return
    #/def __init__
    
    def index_from_qr_tuple(
        self: Self,
        qr: QRTuple
        ) -> int:
        """
            :param QRTuple qr: Sized 2 tuple of qr grid coordinates
            :returns: The index among all board spots
            :rtype: int
        """
        return self.qr_to_index[ qr ]
    #
    
    def index_from_qr( self: Self, q: int, r: int ) -> int:
        """
            :param int q: q coordinate of hex spot
            :param int r: r coordinate of hex spot
            :returns: The index among all board spots
            :rtype: int
        """
        return self.qr_to_index[ (q,r) ]
    #
    
    def qr_from_index( self: Self, index: int ) -> QRTuple:
        """
            :param int index: The index among all board spots
            :returns: Sized 2 tuple of qr grid coordinates
            :rtype: QRTuple
        """
        return self.index_to_qr[ index ]
    #
    
    def stateVector_from_boardState(
        self: Self,
        boardState: BoardState
        ) -> np.ndarray:
        """
            :param BoardState boardState: dictionary of `QRTuple` to `CellStatus` representing every spot on the board
            :returns: OHE vector describing the board. Each space is a series of tuples of length equal to the number of players; if all are 0.0, the space is unoccupied. If it is owned, then the jth index being 1.0 is the owner.
            :rtype: numpy.ndarray
            
            Example: for two players, if we have
            
            [0,1,0,0,1,0], then there are three spaces. Player 1 owns the first space ([0,1]), nobody owns the second ([0,1]), player 0 owns the third ([1,0]).
            
            When an agent from `autoPlayer` picks, they expect to see it as if they were the first player; as a result, they will take the board state, and shift each vector by their player_id, wrapping around. As a result, from their point of view, every board has them as player 0.
        """
        stateVector: np.ndarray = np.zeros(
            shape = (self.length*self.player_count,),
            dtype = float
        )
        
        i: int
        
        for qr, status in boardState.items():
            i = self.qr_to_index[ qr ]
            # Check the status of the space
            if status["owner"] is not None:
                # Index in the tuple is the player index
                stateVector[
                    (
                        self.player_count
                    )*i + status["owner"]
                ] = 1.0
            #/if status["owner"] is None
        #/for qr, status in boardState.items()
        return stateVector
    #/def stateVector_from_boardState
    
    def boardState_from_stateVector(
        self: Self,
        stateVector: np.ndarray
        ) -> BoardState:
        
        boardState: BoardState = {}
        # Iterate through each tuple
        start_i: int
        end_i: int
        next_tup: np.ndarray
        is_one_tup: np.ndarray
        qr: QRTuple
        for i in range( len(self.index_to_qr) ):
            start_i = i*self.player_count
            end_i = start_i + self.player_count
            next_tup = stateVector[
                start_i:end_i
            ]
            owner: int | None
            if np.all(np.isclose( next_tup, 0.0 )):
                owner = None
            #
            else:
                is_one_tup = np.isclose( next_tup, 1.0 )
                if not np.any( is_one_tup ):
                    raise Exception("Missing owner at index={}; next_tup={}".format(i, next_tup))
                #
                owner = np.argmax( is_one_tup )
            #
            qr = self.index_to_qr[ i ]
            boardState[
                qr
            ] = {
                "q": qr[0],
                "r": qr[1],
                "owner": owner
            }
        #/for i in range( self.size )
        
        # Update "occupied_adjacent"
        occupied_adjacent: int
        for qr in boardState:
            boardState[ qr ]["adjacent_occupied"] = adjacent_occupied_count(
                qr = qr,
                boardState = boardState
            )
        #/for qr in boardState
        return boardState
    #/def boardState_from_stateVector
    
    def moveVector_from_play(
        self: Self,
        qr: QRTuple
        ) -> np.ndarray:
        """
            :param QRTuple qr: Sized 2 tuple of qr grid coordinates
            :returns: An array with one value equal to 1.0, rest 0.0, representing the move taken. This is the format of a "player_action" in a history.
            :rtype: numpy.ndarray
            
            Converts a given choice to a move vector, which has the same length as the number of spots
        """
        moveVector: np.ndarray = np.zeros( shape = (self.length,), dtype = float )
        i: int = self.index_from_qr_tuple( qr )
        moveVector[ i ] = 1.0
        
        return moveVector
    #/def moveVector_from_play
    
    def play_from_moveVector(
        self: Self,
        moveVector: np.ndarray
        ) -> QRTuple:
        """
            :param numpy.ndarray moveVector: One hot encoded move. Calculates the qr using `.qr_from_index()`
            :returns: Sized 2 tuple of qr grid coordinates where the move was made
            :rtype: QRTuple
        """
        i: int = next( i for i in range( self.length ) if moveVector[i] > 0 )
        return self.qr_from_index( i )
    #/def play_from_moveVector
#/class HexagonGridHelper

SIZE_DICT: dict[ int, int ] = {3: 19, 4: 37, 5: 61, 6: 91}

def get_spaceCount_forSize(
    size: int,
    hexagonGridHelper: HexagonGridHelper | None = None,
    player_count: int | None = None
    ):
    if size in SIZE_DICT:
        return SIZE_DICT[ size ]
    #
    
    if hexagonGridHelper is None:
        assert player_count is not None
        hexagonGridHelper = HexagonGridHelper( size = size, player_count = player_count )
    #
    SIZE_DICT[ size ] = int( hexagonGridHelper.length )
    
    return SIZE_DICT[ size ]
#/get_spaceCount_forSize

def get_boardState_from_vector(
    boardState_vector: np.ndarray,
    hexagonGridHelper: HexagonGridHelper | None = None,
    player_count: int | None = None,
    size: int | None = None
    ) -> BoardState:
    if hexagonGridHelper is None:
        assert size is not None
        hexagonGridHelper = HexagonGridHelper(
            size = size,
            player_count = player_count
        )
    #
    return hexagonGridHelper.boardState_from_stateVector(
        boardState_vector
    )
#/get_boardState_from_vector

# -- Simulator: interface for Spaghett

class Simulator():
    def __init__(
        self: Self,
        status: jable.JyFrame,
        logging_level: int = 0
    ) -> None:
        self.queue = queue.Queue()
        self.log = queue.Queue()
    #/def __init__
    
    def queueUpdate(
        self: Self,
        update: dict[ str, any ]
        ) -> None:
        self.queue.put( update )
        return
    #/def queueUpdate
    
    def applyUpdates( self: Self ) -> None:
        """
            Apply all updates in the queue, and aggregate them appropriately
        """
        raise NotImplementedError
    #/def applyUpdates
    
    def as_table( self: Self ) -> jable.JyFrame:
        """
            Get the state suitable for reinitializing via `__init__`
        """
        raise NotImplementedError
    #/def as_table
#/Class Simulator

# -- Hexathello
# -- Hexathello -- Print functions

def print_logUpdate( logUpdate: dict ) -> None:
    # TODO: fancier formatting depending on what's in the log item
    print( logUpdate )
    return
#/def print_logItem

class Hexathello( Simulator ):
    """
        Keeps track of a Hex board of an arbitrary positive integer size. Implements Spaghett so it can be used in a Spaghetti bowl
        
        update: PlayerMove
            {
                "turn_index": int,
                "q": int
                "r": int,
                "owner": int
            }
        input status:
            fixed:
                {
                    "winner": int | None,
                    "turn_index": int,
                    "size": int,
                    "game_complete": bool,
                    "empty_count": int,
                    "player_count": int,
                    "current_player": int,
                    "scores": list[ int ]
                }
            shift:
                {
                    "q": int,
                    "r": int,
                    "occupied_adjacent": int,
                    "owner": Literal[ 0,1,None ]
                }
                
        output:

        Hex board info:
        https://www.redblobgames.com/grids/hexagons/
            - Axial Cordinates
    """
    def __init__(
        self: Self,
        status: jable.JyFrame,
        logging_level: int = 0
        ) -> None:
        
        super().__init__(
            status = status,
            logging_level = logging_level
        )
        
        self.logging_level = logging_level
        self.status: dict[{
            "winner": Literal[0,1,None],
            "turn_index": int,
            "size": int,
            "game_complete": bool,
            "empty_count": int,
            "player_count": int,
            "current_player": int,
            "scores": list[ int ]
        }] = status.get_fixed_withDefaultDict(
            {
                "winner": None,
                "turn_index": 0,
                "size": 4,
                "game_complete": False,
                "empty_count": 30, # Count for size 4 with missing origin
                "player_count": 2,
                "current_player": 0,
                "scores": [3,3]
            }
        )
        
        self.boardState: BoardState = {} # { (q,r): {...} }
        
        # Initialize the board hash from status (q,r,"owner")
        for row in status:
            self.boardState[ (row["q"], row["r"] ) ] = {
                "q": row["q"],
                "r": row["r"],
                "occupied_adjacent": row["occupied_adjacent"],
                "owner": row["owner"]
            }
        #/for row in status
        
        # Find spots which are open for moves: those which are empty
        #   but adjacent to a nonempty spot
        self.potential_moves: list[ QRTuple ] = get_potential_moves(
            boardState = self.boardState
        )
        
        # Double check we're good on scores and empty count
        assert self.status["scores"] == get_scores(
            self.boardState,
            self.status["player_count"]
        )
        assert self.status["empty_count"] == get_emptyCount( self.boardState )
        
        # Check if the game is over
        if self.status["empty_count"] == 0\
            or self.status["game_complete"]\
            or (self.status["winner"] is not None):
            # Check all the game over conditions
            
            # Removed empty_count check; can have a stalemate
            # assert self.status["empty_count"] == 0
            assert self.status["game_complete"]
            # Removed winner check: we can have no winner in the case of a tie
            #assert self.status["winner"] is not None
        #/if { game is over }
        
        # Validate starting status if you like
        if False:
            print( self.status )
            print("{} spaces".format(len(self.boardState)))
            print_boardState( self.boardState )
            
            raise Exception("Check status and boardState")
        #/if False
        
        return
    #/def __init__
    
    # -- Hexathello Specifics
    def getMoves_forCurrent( self: Self ) -> MoveChoiceDict:
        """
            Gives a list of captures which would be made, by (q,r)
            
            Thus each CellCapture should have owner == self.statis["currentPlayer"]
        """
        return getMoves_forPlayer(
            player = self.status["current_player"],
            boardState = self.boardState,
            potential_moves = self.potential_moves
        )
    #/def getMoves_forCurrent
    
    # -- Simulator Interface
    
    # def queueUpdate( self: Self, update: dict ) -> None
    
    def applyUpdate_literal(
        self: Self,
        update: CellCapture
        ) -> None:
        """
            Updates exactly on cell without regard to resulting captures
            
            Does update empty count, scores, and potential moves
        """
        qr: QRTuple = ( update["q"], update["r"] )
        old_owner: int | None = self.boardState[ qr ]["owner"]
        
        if old_owner is None and update["owner"] is not None:
            # Taking an empty spot
            # One fewer empty, and change potential moves
            self.status["empty_count"] -= 1
            
            # More score for new owner, others do not change
            self.status["scores"][ update["owner"] ] += 1
            
            self.boardState[ qr ]["owner"] = update["owner"]
            
            new_adjacent_spaces: list[ QRTuple ] = adjacent_spaces(
                qr
            )
            
            # Update occupied_adjacent
            for _qr in new_adjacent_spaces:
                if _qr not in self.boardState:
                    continue
                #
                self.boardState[ _qr ]["occupied_adjacent"] += 1
            #/for _qr in new_adjacent_spaces
        #/if old_owner is None and update["owner"] is not None
        elif old_owner is None and update["owner"] is None:
            # Nothing changes, not a true update
            ...
        #
        elif old_owner is not None and update["owner"] is not None:
            # It's a capture. Empty spaces and potential moves do not change, scores do
            self.boardState[ qr ]["owner"] = update["owner"]
            self.status["scores"][ old_owner ] -= 1
            self.status["scores"][ update["owner"] ] += 1
        #
        elif old_owner is not None and update["owner"] is None:
            # Removing a space. One more empty, lower score for old owner
            # Figuring out adjacency is strange so do it all from scratch
            self.boardState[ qr ]["owner"] = None
            self.status["empty_count"] += 1
            self.status["scores"][ old_owner ] -= 1
            
            # Fix adjacent spots to have one less occupied
            new_adjacent_spaces: list[ QRTuple ] = adjacent_spaces(
                qr
            )
            
            # Update occupied_adjacent
            for _qr in new_adjacent_spaces:
                if _qr not in self.boardState:
                    continue
                #
                self.boardState[ _qr ]["occupied_adjacent"] -= 1
            #/for _qr in new_adjacent_spaces
        #
        else:
            raise Exception(
                "Bad old_owner={}, update['owner']={}".format(
                    old_owner, update["owner"]
                )
            )
        #/switch old_owner, update["owner"]
        self.potential_moves = get_potential_moves(
            self.boardState
        )
        return
    #/def applyUpdate_literal
    
    def applyUpdates(
        self: Self
    ) -> None:
        """
            Updates are a raw move. This updates the scores, counts, and makes a capture
            
            "turn_index" in each update must be the SAME as current turn_index
        """
        
        while not self.queue.empty():
            # Make sure the game isn't over
            if self.status["winner"] is not None:
                raise Exception(
                    'self.status["winner"]={}'.format( self.status["winner"] )
                )
            #
            
            if self.status["game_complete"]:
                raise Exception('# Game is done but with no winner')
                continue
            #
            
            if self.status["empty_count"] <= 0:
                raise Exception(
                    '# self.status["empty_count"]={} but no winners and game is not over'.format(
                        self.status["empty_count"]
                    )
                )
            #
            
            update: PlayerMove = self.queue.get()
            
            # Make sure it's for the right turn
            if update["turn_index"] != self.status["turn_index"]:
                print("# Bad turn index: {} on turn {}".format(update["turn_index"],self.status["turn_index"]))
                continue
            #/if update["turn_index"] != self.status["turn_index"]
            
            # Check it's the right player
            if update["owner"] != self.status["current_player"]:
                print(
                    "# Found turn for player {}, expected {}".format(
                        update["owner"], self.status["current_player"]
                    )
                )
                continue
            #
            
            # Can only move to a board state which makes a capture, and
            #   is in an empty spot
            captures: list[ CellCapture ] = getCaptures_forMove(
                update = update,
                boardState = self.boardState
            )
            assert len( captures ) > 0
            
            if False:
                print( "update: {}".format( update ) )
                print( "captures: {}".format( captures ) )
            #
            
            # Apply the move and the captures; they're already meant to be valid
            self.applyUpdate_literal(
                update
            )
            
            # Log the move
            if self.logging_level > 0:
                self.log.put(
                    update
                )
            #

            # Add the captures; they will be listed as {"q": int, "r": int, "owner": int }
            # Also update scores and empty spots for captures
            # self.potential_moves does not change since this is only a flip,
            #
            for _update in captures:
                # Make sure we're only capturing taken spots
                assert self.boardState[
                    ( _update["q"], _update["r"] )
                ]["owner"] is not None
                self.applyUpdate_literal( _update )
                # get the captured coordinates

                # Log if necessary
                # Only at level 2 do we do every resulting change
                if self.logging_level > 1:
                    self.log.put(
                        _update
                    )
                #/if self.logging_level > 0
            #for _update in captures
            
            # Check if the game is over
            game_over: bool = False
            if self.status["empty_count"] <= 0:
                game_over = True
            #/if self.status["empty_count"] <= 0
            else:
                # Next turn
                self.status["turn_index"] += 1
                
                # Next player; check
                # We could have a stalemate so keep track of how many we have skipped
                player_skips: int = 0
                self.status["current_player"] += 1
                self.status["current_player"] %= self.status["player_count"]
                
                # Check if next player has moves
                potential_plays: MoveChoiceDict = getMoves_forPlayer(
                    player = self.status["current_player"],
                    boardState = self.boardState,
                    potential_moves = self.potential_moves
                )
                
                # Cycle moves if there are no plays to be made
                while len( potential_plays ) <= 0:
                    player_skips += 1
                    if player_skips >= self.status["player_count"]:
                        # End of game
                        game_over = True
                        break
                    #/if player_skips >= self.status["player_count"]/else
                    
                    # Go to next player
                    self.status["current_player"] += 1
                    self.status["current_player"] %= self.status["player_count"]
                    
                    potential_plays: MoveChoiceDict = getMoves_forPlayer(
                        player = self.status["current_player"],
                        boardState = self.boardState,
                        potential_moves = self.potential_moves
                    )
                #/while len( potential_plays ) <= 0
            #/if self.status["empty_count"] <= 0/else
            
            if game_over:
                self.status["game_complete"] = True
                winning_score: int = 0
                
                winner_list: list[ int ] = []
                
                # Winner is arg max of scores
                # Sadly we just throw in a naive arg max
                for i in range( self.status["player_count"] ):
                    if self.status["scores"][ i ] >= winning_score:
                        winning_score = self.status["scores"][ i ]
                    #/if self.status["scores"][ i ] > winning_score
                #/for i in range( self.status["player_count"] )
                
                # Check if there are any ties
                for i in range( self.status["player_count"] ):
                    if self.status["scores"][ i ] >= winning_score:
                        winner_list.append( i )
                    #/if self.status["scores"][ i ] >= winning_score
                #/for i in range( self.status["player_count"] )
                
                if self.status["empty_count"] > 0:
                    print("Early end, no move for any player")
                    
                    # Verify there indeed are no moves to be made
                    for player_index in range( self.status["player_count"] ):
                        _moves_dict: MoveChoiceDict = {}
                        for qr, cellStatus in self.boardState.items():
                            if cellStatus["owner"] is None:
                                _moves_dict[ qr ] = getCaptures_forMove(
                                    update = {
                                        "q": qr[0],
                                        "r": qr[1],
                                        "owner": player_index
                                    },
                                    boardState = self.boardState
                                )
                            #/if cellStatus["owner"] is None
                        #/for qr, cellStatus in self.boardState.items()
                        if not all(
                            len( captures ) <= 0  for captures in _moves_dict.values()
                        ):
                            print("Found potential captures for player {}".format(player_index))
                            print( _moves_dict )
                            raise Exception("Incorrect early end")
                        #/if { some capture is available }
                    #/for player_index in range( self.statu["player_count"] )
                #/if self.status["empty_count"] > 0
                
                if len( winner_list ) == 1:
                    self.status["winner"] = winner_list[0]
                    print(
                        "RESULT: {}; Player {} wins".format(
                            " - ".join(
                                [
                                    str(
                                        self.status["scores"][i]
                                    ) for i in range(
                                        self.status["player_count"]
                                    )
                                ]
                            ),
                            self.status["winner"]
                        )
                    )
                #
                else:
                    self.status["winner"] = None
                    print(
                        "RESULT: {}; Tie among {}".format(
                            " - ".join(
                                [
                                    str(
                                        self.status["scores"][i]
                                    ) for i in range(
                                        self.status["player_count"]
                                    )
                                ]
                            ),
                            ", ".join(
                                str(_winner) for _winner in winner_list
                            )
                        )
                    )
                #
            #/if game_over
            
            if self.logging_level > 0:
                self.log.put( self.status )
            #
        #/while not self.queue.empty()
        return
    #/def applyUpdates
    
    def as_table( self: Self ):
        """
            Serialize into a json table
        """
        table = jable.fromHeaders(
            fixed = self.status,
            shiftHeader = [
                "q", "r", "owner"
            ]
        )
        
        # Add in the board state
        for qr, owner in self.boardState.items():
            table.append(
                {"q": qr[0], "r": qr[1], "owner": owner}
            )
        #/for qr, owner in self.boardState.items()
        
        return table
    #/def as_table
#/class Hexathello

def new_initial_boardState(
    size: int,
    player_count: int,
    include_middle = True
    ) -> BoardState:
    """
        Set initial empty board, with a ring around the center of alternating players
    """
    assert player_count in [2,3,6]
    assert size > 2
    
    hexagonGridHelper: HexagonGridHelper = HexagonGridHelper(
        size = size,
        player_count = player_count
    )
    
    # We allow the middle play. For historical reasons, we show the logic
    #   of leaving out the middle space
    if include_middle:
        boardState: BoardState = {
            qr: {
                "q": qr[0],
                "r": qr[1],
                "occupied_adjacent": 0,
                "owner": None
            } for qr in hexagonGridHelper.index_to_qr
        }
    #
    else:
        boardState: BoardState = {
            qr: {
                "q": qr[0],
                "r": qr[1],
                "occupied_adjacent": 0,
                "owner": None
            } for qr in hexagonGridHelper.index_to_qr if qr != (0,0)
        }
        raise Exception("include_middle=False deprecated")
    #/if include_middle
    
    # Set the initial players in the innner ring starting at (1,0)
    _player_index: int = 0
    for vector in CLOCKWISE_UNIT_VECTOR_LIST:
        boardState[ vector ]["owner"] = _player_index
        # Cycle to next player
        _player_index += 1
        _player_index %= player_count
    #/for vector in CLOCKWISE_UNIT_VECTOR_LIST
    
    # Update "occupied_adjacent"
    for qr, cellStatus in boardState.items():
        _adjacent_spaces: list[ QRTuple ] = adjacent_spaces(
            qr
        )
        for _qr in _adjacent_spaces:
            if _qr not in boardState:
                continue
            #
            if boardState[ _qr ]["owner"] is not None:
                boardState[ qr ]["occupied_adjacent"] += 1
            #
        #/for _qr in _adjacent_spaces
    #/for qr, cellStatus in boardState
    
    return boardState
#/def new_initial_boardState

def new_hexathello(
    size: int,
    player_count: int,
    player_start: int = 0,
    logging_level: int = 0
    ) -> Hexathello:
    """
        Initialize everything for a standard hexathello game
    """
    
    # Initialize board state without the origin
    boardState: BoardState = new_initial_boardState(
        size = size,
        player_count = player_count
    )
    
    # Random player starts
    
    # Initialize status for the Hexathello
    initial_status: jable.JyFrame = jable.fromHeaders(
        fixed = {
            "winner": None,
            "turn_index": 0,
            "size": size,
            "game_complete": False,
            "empty_count": get_emptyCount( boardState ),
            "player_count": player_count,
            "current_player": player_start,
            "scores": get_scores(
                boardState,
                player_count
            )
        },
        shiftHeader = [
            "q","r","owner", "occupied_adjacent"
        ]
    )
    
    # Add board state
    for status in boardState.values():
        initial_status.append(
            status
        )
    #/for status in boardState.values()
    
    # Can verify initial status
    if False:
        for row in initial_status:
            print(
                "({},{}): {}".format(
                    row["q"],row["r"],row["owner"]
                )
            )
            if (row["q"],row["r"] ) == (0,0):
                raise Exception("Should not have middle in initial_status")
            #
        #/for row in initial_status
    #/if False
    
    return Hexathello(
        status = initial_status,
        logging_level = logging_level
    )
#/def new_hexathello
