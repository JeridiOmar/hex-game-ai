import copy
from math import log, sqrt, inf
from random import choice

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table


class Node(object):
    def __init__(self, logic, board, move=(None, None), wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class STRAT:
    def __init__(self, logic, ui, board_state, starting_player):
        self.logic = logic
        self.ui = ui
        self.root_state = copy.copy(board_state)
        self.state = copy.copy(board_state)
        self.starting_player = starting_player
        self.players = [1, 2]
        self.players.remove(self.starting_player)
        self.other_player = self.players[0]
        self.turn = {True: self.starting_player, False: self.other_player}
        self.turn_state = True

    def start(self) -> tuple:
        root_node = Node(self.logic, self.root_state)

        if   self.starting_player is self.ui.BLACK_PLAYER:
            # implement here Black player strategy (if needed, i.e., no human playing)
        elif self.starting_player is self.ui.WHITE_PLAYER:
            # implement here White player strategy
        return (x, y)
