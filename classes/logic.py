from typing import Optional

import numpy as np
import random as rd

from classes.strategy import STRAT



class Logic:
    def __init__(self, ui):
        self.ui = ui
        self.GAME_OVER = False
        self.logger = np.zeros(shape=(self.ui.board_size, self.ui.board_size), dtype=np.int8)

    def get_possible_moves(self, board: np.ndarray) -> list:
        """
        @return   All the coordinates of nodes where it is possible to play.
        """
        (x, y) = np.where(board == 0)
        return list(zip(x, y))

    def make_move(self, coordinates: tuple, player: Optional[int]):
        """
        This procedure updates the game by applying the given action of the player at the given coordinates of the board.
        """
        (x, y) = coordinates
        node = x * self.ui.board_size + y

        if   player is None:                  self.ui.color[node] = self.ui.black
        elif player is self.ui.BLACK_PLAYER:  self.ui.color[node] = self.ui.black
        else:                                 self.ui.color[node] = self.ui.white

    def is_game_over(self, player: int, board: np.ndarray) -> Optional[int]:
        """
        @return   The winning player:  1 or 2 (or None if the game is over by lack of playable position!)

        As a side-effect, sets GAME_OVER to True if there are no more moves to play.
        """
        if self.get_possible_moves(board) == []:
            self.GAME_OVER = True

        for _ in range(self.ui.board_size):
            if player is self.ui.BLACK_PLAYER:
                border = (_, 0)
            if player is self.ui.WHITE_PLAYER:
                border = (0, _)

            path = self.traverse(border, player, board, {})
            if path:
                return player

    def is_border(self, node: tuple, player: int) -> bool:
        """
        @return   Checks whether the given node is a border that belongs to the given player.
        """
        (x, y) = node
        return (player is self.ui.BLACK_PLAYER and y == self.ui.board_size - 1 or
                player is self.ui.WHITE_PLAYER and x == self.ui.board_size - 1)

    def traverse(self, node: tuple, player: int, board: np.ndarray, visited: dict) -> Optional[list]:
        """
        @return   the path of node connecting two borders for player, if existing
        """
        (x, y) = node
        neighbours = self.get_neighbours(node)

        try:
            if visited[node]:
                pass
        except KeyError:
            if board[x][y] == player:
                visited[node] = 1

                if self.is_border(node, player):
                        self.GAME_OVER = True

                for neighbour in neighbours:
                    self.traverse(neighbour, player, board, visited)

        if self.GAME_OVER:
            return visited

    def get_neighbours(self, coordinates: tuple) -> list:
        """
        @return   a list of the neighbours of "coordinates" node
        """
        (x, y) = coordinates
        return [ node
                 for row in range(-1, 2)
                 for col in range(-1, 2)
                 if row != col
                 for node in [ (x + row, y + col) ]
                 if self.is_valid(node) ]

    def is_valid(self, coordinates: tuple) -> bool:
        """
        @return   True iff node exists.
        """
        return all([ 0 <= c < self.ui.board_size
                     for c in coordinates ])

    def is_node_free(self, coordinates: tuple, board: np.ndarray) -> bool:
        """
        @return   True iff node is free.
        """
        (x, y) = coordinates
        return not board[x][y]

    def get_action(self, node: Optional[int], player: int) -> Optional[int]:
        """
        @return   The winning player (1 or 2) or 0 if there is not yet a winner or even None if the game is over by lack of playable position.

        As a side-effect, sets GAME_OVER to True if there are no more moves to play.
        """
        if player is self.ui.BLACK_PLAYER:
            # Human (or AI) player
            if self.ui.mode == 'man_vs_ai':
                (x, y) = self.ui.get_true_coordinates(node)
            else:
                # Debug: random player
                #    x, y = rd.choice(self.get_possible_moves(self.logger))
                self.strategy = STRAT(logic=self, ui=self.ui, board_state=self.logger, starting_player=self.ui.BLACK_PLAYER)
                (x, y) = self.strategy.start()
        elif player is self.ui.WHITE_PLAYER:
            # AI player
            # Debug: random player
            #  x, y = rd.choice(self.get_possible_moves(self.logger))
            self.strategy = STRAT(logic=self, ui=self.ui, board_state=self.logger, starting_player=self.ui.WHITE_PLAYER)
            (x, y) = self.strategy.start()

        assert self.is_node_free((x, y), self.logger), "node is busy"

        self.make_move((x, y), player)
        self.logger[x][y] = player

        return self.is_game_over(player, self.logger)
