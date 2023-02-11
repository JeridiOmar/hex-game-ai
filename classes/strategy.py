import copy
from collections import deque
from math import log, sqrt, inf
from random import choice
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
import sys
import networkx as nx

sys.setrecursionlimit(1500)


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
        self.count = 0

    def start(self) -> tuple:
        root_node = Node(self.logic, self.root_state)

        if self.starting_player is self.ui.BLACK_PLAYER:
            # implement here Black player strategy (if needed, i.e., no human playing)
            # moves=self.logic.get_possible_moves(self,self.state)
            # x, y = choice(self.logic.get_possible_moves(self.state))
            # ((x, y), value) = self.minimax(root_node, self.starting_player, self.state)
            # (x, y) = self.play_minimax_alpha_beta(root_node)
            (x, y) = self.play_heuristic(root_node, self.number_of_paths_other_side)
        elif self.starting_player is self.ui.WHITE_PLAYER:
            # x, y = choice(self.logic.get_possible_moves(self.state))  # check for what is root_node
            # ((x, y), value) = self.minimax(root_node, self.starting_player, self.state)
            # (x, y) = self.play_minimax_alpha_beta_heuristic(root_node, 2)
            # (x, y) = self.play_minimax_alpha_beta(root_node)
            (x, y) = self.play_heuristic(root_node, self.longest_player_chain)
        return (x, y)

    #######################
    ### MINIMAX
    #######################
    """
    function to call in play function to have the best move (x,y)
    this functions initiate the call for the minimax function
    """

    def play_minimax(self, root_node):
        possible_moves = self.logic.get_possible_moves(self.state)
        childs_performance = []
        for move in possible_moves:
            newBoard = copy.copy(self.state)
            (x, y) = move
            newBoard[x][y] = self.starting_player
            root_node.add_child(Node(self.logic, newBoard, move))
        for child in root_node.children:
            value = self.minimax(child, self.other_player, child.state)
            childs_performance.append((child, value))
        if self.starting_player == self.ui.BLACK_PLAYER:
            max_value = max(childs_performance, key=lambda v: v[1])
            max_indexes = []
            for index, value in enumerate(childs_performance):
                if value[1] == max_value[1]:
                    max_indexes.append(index)

            return childs_performance[choice(max_indexes)][0].move

        else:
            min_value = min(childs_performance, key=lambda v: v[1])
            min_indexes = []
            for index, value in enumerate(childs_performance):
                if value[1] == min_value[1]:
                    min_indexes.append(index)
            return childs_performance[choice(min_indexes)][0].move

    def minimax(self, node, player, board):

        if self.logic.is_game_over(self.ui.BLACK_PLAYER, board) == 1:
            self.logic.GAME_OVER = False
            return 1
        if self.logic.is_game_over(self.ui.WHITE_PLAYER, board) == 2:
            self.logic.GAME_OVER = False
            return -1
        if len(self.logic.get_possible_moves(board)) == 0:
            if player == self.ui.BLACK_PLAYER:
                return -1
            else:
                return 1

        (x, y) = np.where(board == 0)
        possible_moves = list(zip(x, y))
        for move in possible_moves:
            newBoard = copy.copy(board)
            (x, y) = move
            newBoard[x][y] = player
            node.add_child(Node(self.logic, newBoard, move))
        possible_values = []
        for child in node.children:
            if player == self.ui.BLACK_PLAYER:
                value = self.minimax(child, self.ui.WHITE_PLAYER, child.state)
            else:
                value = self.minimax(child, self.ui.BLACK_PLAYER, child.state)

            possible_values.append(value)

        if player == self.ui.BLACK_PLAYER:

            return max(possible_values)
        else:

            return min(possible_values)

    #######################
    ### MINIMAX WITH PRUNING
    #######################
    """
    function to call in start() function to have the best move (x,y)
    this functions initiate the call for the minimax pruning function
    """

    def play_minimax_alpha_beta(self, root_node):
        possible_moves = self.logic.get_possible_moves(self.state)
        childs_performance = []
        alpha = float('-inf')
        beta = float('+inf')
        bestmove = None
        if (self.starting_player == self.ui.BLACK_PLAYER):
            bestval = float("-inf")
        else:
            bestval = float("inf")
        for move in possible_moves:
            newBoard = copy.copy(self.state)
            (x, y) = move
            newBoard[x][y] = self.starting_player
            root_node.add_child(Node(self.logic, newBoard, move))
        for child in root_node.children:
            value = self.minimax_alpha_beta(child, self.other_player, child.state, alpha, beta)
            if self.starting_player == self.ui.BLACK_PLAYER and value >= bestval:
                bestval = value
                bestmove = child.move
            if self.starting_player == self.ui.WHITE_PLAYER and value <= bestval:
                bestval = value
                bestmove = child.move

        return bestmove

    def minimax_alpha_beta(self, node, player, board, alpha, beta):
        print("Mid Game", -self.heuristic_evaluation2(board, player))
        if self.logic.is_game_over(self.ui.BLACK_PLAYER, board) == 1:
            self.logic.GAME_OVER = False
            return 1
        if self.logic.is_game_over(self.ui.WHITE_PLAYER, board) == 2:
            self.logic.GAME_OVER = False
            return -1
        if len(self.logic.get_possible_moves(board)) == 0:
            if player == self.ui.BLACK_PLAYER:
                return -1
            else:
                return 1

        bestval = 0
        if player == self.ui.BLACK_PLAYER:
            bestval = float("-inf")
        else:
            bestval = float("inf")

        (x, y) = np.where(board == 0)
        possible_moves = list(zip(x, y))
        for move in possible_moves:
            newBoard = copy.copy(board)
            (x, y) = move
            newBoard[x][y] = player
            node.add_child(Node(self.logic, newBoard, move))
        possible_values = []
        for child in node.children:
            if player == self.ui.BLACK_PLAYER:
                value = self.minimax_alpha_beta(child, self.ui.WHITE_PLAYER, child.state, alpha, beta)
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            else:
                value = self.minimax_alpha_beta(child, self.ui.BLACK_PLAYER, child.state, alpha, beta)
                if value <= alpha:
                    return value
                beta = min(beta, value)

            possible_values.append(value)

        if player == self.ui.BLACK_PLAYER:

            return max(possible_values)
        else:

            return min(possible_values)

    #######################
    ### HEURISTICS
    #######################
    """
    function to call in play function to have the best move (x,y)
    with this version we can pass the maximum depth to go through 
    to be able to limit the depth we call the heuristic as evaluation of unfinished board
    this function initiate the call to minimax_alpha_beta_depth 
    """

    def play_minimax_alpha_beta_heuristic(self, root_node, max_depth):
        possible_moves = self.logic.get_possible_moves(self.state)
        childs_performance = []
        alpha = float('-inf')
        beta = float('+inf')
        bestmove = None
        if (self.starting_player == self.ui.BLACK_PLAYER):
            bestval = float("-inf")
        else:
            bestval = float("inf")
        for move in possible_moves:
            new_board = copy.copy(self.state)
            (x, y) = move
            new_board[x][y] = self.starting_player
            root_node.add_child(Node(self.logic, new_board, move))
        for child in root_node.children:
            value = self.minimax_alpha_beta_depth(child, self.other_player, child.state, alpha, beta, max_depth)
            if self.starting_player == self.ui.BLACK_PLAYER and value >= bestval:
                bestval = value
                bestmove = child.move
            if self.starting_player == self.ui.WHITE_PLAYER and value <= bestval:
                bestval = value
                bestmove = child.move

        return bestmove

    """
    this is a variant of the minimax_alpha_beta declared at the beginning 
    but with the possibility to limit the depth and call the heuristic
    """

    def minimax_alpha_beta_depth(self, node, player, board, alpha, beta, max_depth):
        # if game is finished we return the maximum score no need for the evaluation

        if self.logic.is_game_over(self.ui.BLACK_PLAYER, board) == 1:
            self.logic.GAME_OVER = False
            return 50
        if self.logic.is_game_over(self.ui.WHITE_PLAYER, board) == 2:
            self.logic.GAME_OVER = False
            return -50
        if len(self.logic.get_possible_moves(board)) == 0:
            # print(player)
            if player == self.ui.BLACK_PLAYER:
                return -50
            else:
                return 50
        if max_depth <= 0:
            # if the game is still unfinished we call the heuristic
            if player == self.ui.BLACK_PLAYER:
                # here we can change the heuristic to call
                return -self.distance_to_opposite_side(board, player)
            else:
                # here we can change the heuristic to call
                return self.distance_to_opposite_side(board, player)
        bestval = 0
        if player == self.ui.BLACK_PLAYER:
            bestval = float("-inf")
        else:
            bestval = float("inf")

        (x, y) = np.where(board == 0)
        possible_moves = list(zip(x, y))
        for move in possible_moves:
            newBoard = copy.copy(board)
            (x, y) = move
            newBoard[x][y] = player
            node.add_child(Node(self.logic, newBoard, move))
        possible_values = []
        for child in node.children:
            if player == self.ui.BLACK_PLAYER:
                value = self.minimax_alpha_beta_depth(child, self.ui.WHITE_PLAYER, child.state, alpha, beta,
                                                      max_depth - 1)
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            else:
                value = self.minimax_alpha_beta_depth(child, self.ui.BLACK_PLAYER, child.state, alpha, beta, max_depth)
                if value <= alpha:
                    return value
                beta = min(beta, value)

            possible_values.append(value)
        if player == self.ui.BLACK_PLAYER:

            return max(possible_values)
        else:

            return min(possible_values)

    ####
    # 1st heuristic
    # calculates the total distance of all pieces of a specific player to the opposite side of the board.
    # If the player is self.ui.BLACK_PLAYER, it calculates the distance of each piece to the bottom row (n-1),
    # otherwise, it calculates the distance to the rightmost column (m-1).
    # The distance is computed as the absolute difference between
    # the piece's position and the opposite side and the sum of all distances is returned as the result.
    ###
    def distance_to_opposite_side(self, board, player):
        n, m = board.shape
        if player == self.ui.BLACK_PLAYER:
            x, y = np.where(board == player)
            dist = np.abs(x - (n - 1))
            return np.sum(dist)
        else:
            x, y = np.where(board == player)
            dist = np.abs(y - (m - 1))
            return np.sum(dist)

    ######################

    ###
    # heuristic 2
    # Distance to the opposite side with number of possible paths:
    # The distance from each player's stones to the opposite side .
    # A player who is closer to the opposite side has a greater chance of forming a connection,
    # and therefore, this player is likely to have a higher score
    #
    def number_of_paths_other_side(self, board, player):
        # Check how many paths the player has to their own side
        count = 0
        if player == self.ui.BLACK_PLAYER:
            # Black player - Check paths from left to right
            for row in range(self.ui.board_size):
                for col in range(self.ui.board_size):
                    if board[row][col] == player:
                        count += self.check_path_left_right(row, col, player, board)
        else:
            # White player - Check paths from top to bottom
            for row in range(self.ui.board_size):
                for col in range(self.ui.board_size):
                    if board[row][col] == player:
                        count += self.check_path_top_bottom(row, col, player, board)

        return count

    def check_path_left_right(self, row, col, player, board):
        # Check if current position is part of a path from left to right
        count = 0
        visited = set()
        if self.dfs_left_right(row, col, player, board, visited):
            count += 1
        return count

    def check_path_top_bottom(self, row, col, player, board):
        # Check if current position is part of a path from top to bottom
        count = 0
        visited = set()
        if self.dfs_top_bottom(row, col, player, board, visited):
            count += 1
        return count

    def dfs_left_right(self, row, col, player, board, visited):
        # Perform depth-first search to find a path from left to right
        if (row, col) in visited:
            return False
        visited.add((row, col))
        if col == self.ui.board_size - 1:
            return True
        # for r, c in [(row - 1, col), (row + 1, col), (row, col + 1)]:
        #     if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
        #         if self.dfs_left_right(r, c, player, board, visited):
        #             return True
        if (row % 2) == 0:
            for r, c in [(row - 1, col - 1), (row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col + 1),
                         (row + 1, col)]:
                if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
                    if self.dfs_left_right(r, c, player, board, visited):
                        return True
        else:
            for r, c in [(row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1), (row + 1, col),
                         (row + 1, col + 1)]:
                if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
                    if self.dfs_left_right(r, c, player, board, visited):
                        return True
        return False

    def dfs_top_bottom(self, row, col, player, board, visited):
        # Perform depth-first search to find a path from top to bottom
        if (row, col) in visited:
            return False
        visited.add((row, col))
        if row == self.ui.board_size - 1:
            return True
        if (row % 2) == 0:
            for r, c in [(row - 1, col - 1), (row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col + 1),
                         (row + 1, col)]:
                if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
                    if self.dfs_top_bottom(r, col, player, board, visited):
                        return True
        else:
            for r, c in [(row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1), (row + 1, col),
                         (row + 1, col + 1)]:
                if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
                    if self.dfs_top_bottom(r, col, player, board, visited):
                        return True
        return False

    ####
    # Heuristic 3
    # Longest chain: The length of the longest chain of each player .
    # A player with a longer chain is closer to winning, so this player is likely to have a higher score.
    ###

    def longest_chain(self, board, player):
        max_chain_length = 0
        for row in range(self.ui.board_size):
            for col in range(self.ui.board_size):
                if board[row][col] == player:
                    chain_length = self._dfs_chain_length(row, col, player, board, set())
                    max_chain_length = max(max_chain_length, chain_length)
        return max_chain_length

    def _dfs_chain_length(self, row, col, player, board, visited):
        if (row, col) in visited:
            return 0
        visited.add((row, col))
        chain_length = 1
        for r, c in [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]:
            if 0 <= r < self.ui.board_size and 0 <= c < self.ui.board_size and board[r][c] == player:
                chain_length += self._dfs_chain_length(r, c, player, board, visited)
        return chain_length

    def longest_player_chain(self, board, player):
        return self.longest_chain(board, player)

    ######
    # Heuristic play function to test the heuristic without going into minimax
    ###
    def play_heuristic(self, root_node, heuristic):
        best_eval = -inf
        best_move = None
        possible_moves = self.logic.get_possible_moves(self.state)
        for move in possible_moves:
            new_board = copy.copy(self.state)
            (x, y) = move
            new_board[x][y] = self.starting_player
            root_node.add_child(Node(self.logic, new_board, move))
        for child in root_node.children:
            eval = heuristic(child.state, self.starting_player)
            print(eval)
            if eval > best_eval:
                best_eval = eval
                best_move = child.move

        return best_move
