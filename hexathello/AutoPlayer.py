"""
    Interface to play games (of Hexathello) and save the results.
    
    Plays games with ``hexathello.aiPlayers.HexAgent`` objects.
    
    Returns game results as a `JyFrame` history class; see ``hexathello.history.new_literalHistory()``, ``hexathello.history.new.pov_history()``. Save these to disk and use them to train AI, most likely ``hexathello.aiPlayers.KerasHexAgent`` subclasses.
"""

from . import engine, history, jable
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

def runHexathello_withAgents(
    agents: list[ AiAgentProtocol ],
    size: int,
    logging_level: int = 0,
    rng: np.random.Generator | None = None,
    hexagonGridHelper: engine.HexagonGridHelper | None = None
    ) -> jable.JyFrame:
    """
        :param list[ AiAgentProcol ] agents: Initialized list of ai agents from ``aiPlayers``
        :param int size: Length of one side of the hexagon board
        :param int logging_level: Passed to the hexathello engine for how much detail to log
        :param np.random.Generator rng: Used to make various decisions both for the game and AI agents
        :param engine.HexagonGridHelper hexagonGridHelper: Used to help with calculations for both the engine and AI agents
        :returns: The literal game history
        :rtype: jable.JyFrame
        
        Plays a game with set AI and prints everything as it goes
        
        fixed:
            player_count: int
            size: int
            winner: int|None
            scores: list[ int ]
        shift:
            turn_index: int
            current_player: int
            boardState: np.ndarray
            actionChoices: np.ndarray
            player_action: np.ndarray
            action_tags: list[ str ]
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
    
    
    # Initialize game_history
    game_history: jable.JyFrame = history.new_literalHistory(
        player_count = player_count,
        size = size,
        winner = None,
        scores = deepcopy( hexathello.status["scores"] )
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
        
        # Update game_history
        game_history.append(
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
                ),
                "action_tags": next_move["action_tags"]
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
            
            # Set game_history
            game_history["winner"] = hexathello.status["winner"]
            game_history["scores"] = deepcopy( hexathello.status["scores"] )
            break
        #/if hexathello.status["game_complete"]
    #/for _ in range( empty_count )
    return game_history
#/def runHexathello_withAgents
