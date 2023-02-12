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
        self.dijkstra_graph = nx.Graph()

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
        self.dijkstra_graph = nx.Graph()


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
            # Play with change
            # x, y = choice(self.logic.get_possible_moves(self.state))  # check for what is root_node

            # Play minimax
            (x,y) = self.minimax(0)

            # Play minimax with pruning
            # (x, y) = self.play_minimax_alpha_beta(root_node)

            # Play minimax with limited depth (dont forget to change the heuristic you want were mentioned)
            # (x, y) = self.play_minimax_alpha_beta_heuristic(root_node, 2)

            # Play with heuristics without minimax for test purposes (valid for heuristics 1 2 3)
            # (x, y) = self.play_heuristic(root_node, self.longest_player_chain)

            # Play with the 4th heuristic (dijkstra)
            # (x, y) = self.search_with_heuristic(0, 2)
        return (x, y)

    #######################
    ### MINIMAX
    #######################
    """
    function to call in play function to have the best move (x,y)
    this functions initiate the call for the minimax function
    """

    def minimax(self, p_next: int) -> (tuple):
        # check if the game is over

        f = self.logic.is_game_over(self.starting_player, self.state)  # starting player won
        if f is None:
            f = self.logic.is_game_over(self.other_player, self.state)  # opposing player one
        # if game is over => we reached a leaf node
        if self.logic.GAME_OVER == True:
            self.logic.GAME_OVER = False
            # if neither player won draw:
            if f is None:
                return (0, None)
            # if starting player won:
            elif f == self.starting_player:
                return (1, None)
            # if the opposing player won:
            else:
                return (-1, None)

        # if the game didn't end
        else:
            l = self.logic.get_possible_moves(
                self.state)  # finds all the possible moves to make

            # either starting player turn or opposing player turn:

            # it is the starting player turn:
            if self.turn_state == True:
                # the next turn would be the opposing player's turn:
                self.turn_state = not self.turn_state  # changing the turn to the other player
                the_m = - inf  # check if it works
                # in case we are in level 0
                if (p_next == 0):
                    m = 0
                    res_t = (None, None)
                    for i in l:
                        self.state[i[0]][i[1]] = self.starting_player
                        m = self.minimax(p_next + 1)[0]
                        # we can implement random to choose any
                        if (the_m < m):
                            the_m = m
                            res_t = i
                        self.state[i[0]][i[1]] = 0
                    self.turn_state = not self.turn_state
                    return res_t
                else:
                    for i in l:
                        self.state[i[0]][i[1]] = self.starting_player
                        the_m = max(the_m, self.minimax(p_next + 1)[0])
                        self.state[i[0]][i[1]] = 0

            # it is the opposing player's turn:
            else:
                the_nums = 0
                the_m = inf
                self.turn_state = not self.turn_state
                m = 0
                for i in l:
                    self.state[i[0]][i[1]] = self.other_player
                    the_m = min(the_m, self.minimax(p_next + 1)[0])
                    if (the_m == -1 and the_nums == 0 and p_next == 1):
                        the_nums += 1
                    self.state[i[0]][i[1]] = 0
            self.turn_state = not self.turn_state
            return (the_m, None)

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
        # print("Mid Game", -self.heuristic_evaluation2(board, player))
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



    ###
    # Bilal heurisitic
    ###

    def graph_generator2(self, player: int):
        # we get the opposing player for later reference:
        if (player == 1):
            other_player = 2
        else:
            other_player = 1
        # we clear the graph from prior usage
        self.dijkstra_graph.clear()
        l_nei = list()  # this list will mostly used to parse through nodes' neighbors
        # we will add all the possible cells to the graph
        # 1-we will then remove all the nodes owned by the opposing player
        # 2-we will remove all the nodes owned by the player and connected their neighbors directly to each other
        self.dijkstra_graph.add_nodes_from([(i, j) for i in range(self.state.shape[0])
                                            for j in range(self.state.shape[1])])

        # naive edge creation
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                # getting the neighbors of each node:
                l_nei = self.logic.get_neighbours((i, j))
                # connecting each cell to its neighbors
                for k in l_nei:
                    if not self.dijkstra_graph.has_edge((i, j), k):
                        self.dijkstra_graph.add_edge((i, j), k)
        # now we are more free to work independently from the board structure and the self.logic structure
        #
        # parsing through the cells and removing the opposing player's cells
        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):
                if (self.state[i][j] == other_player):
                    # print(self.state[0][0])
                    # print(".")
                    self.dijkstra_graph.remove_node((i, j))
                # if it was owned by the player:
                elif self.state[i][j] == player:
                    # get all neighbors
                    l_nei = list(self.dijkstra_graph.neighbors((i, j)))
                    # print(l_nei)
                    for k in l_nei:
                        if self.state[k[0]][k[1]] == other_player:
                            l_nei.remove(k)
                    # for each neigbor connect it to all the other neighbors
                    # print("*******************************************")
                    # print("for cell ", (i, j))
                    new_additions = list()
                    for k in l_nei:
                        for m in l_nei:
                            # be aware of self loops so don't connect the node to itself
                            # be aware of neighbors that were already connected
                            if ((m[0] == k[0] and m[1] == k[1])):
                                continue
                            if (self.dijkstra_graph.has_edge(m, k)):
                                # print("the edge already exists: ", m, k)
                                continue
                            if ((k, m) in new_additions):
                                continue
                            # if all the conditions are satisfied connect the two neighbors
                            new_additions.append((m, k))
                            # self.dijkstra_graph.add_edge(m, k)
                            # print("m is ", m, "and k is : ", k)
                    for edges_to_add in new_additions:
                        self.dijkstra_graph.add_edge(
                            edges_to_add[0], edges_to_add[1])
                    # remove the node
                    # print("*****************************************")
                    self.dijkstra_graph.remove_node((i, j))

    def get_border_spes(self, player, cell, visited, border):
        if (player == 1):
            other_player = 2
        else:
            other_player = 1
        cell_nei = self.logic.get_neighbours(cell)
        for nei in cell_nei:
            if self.state[nei[0]][nei[1]] != other_player:
                if (nei not in visited):
                    visited.append(nei)
                    if (self.state[nei[0]][nei[1]] == 0):
                        border.append(nei)
                    elif (self.state[nei[0]][nei[1]] == player):
                        self.get_border_spes(player, nei, visited, border)

    def get_border_points2(self, player):
        borders_1 = list()
        borders_2 = list()
        visited = list()
        if player is self.ui.BLACK_PLAYER:
            for i in range(self.state.shape[0]):
                if (i, 0) in visited:
                    continue
                if (self.state[i][0] == 0):
                    visited.append((i, 0))
                    borders_1.append((i, 0))
                elif self.state[i][0] == player:
                    visited.append((i, 0))
                    self.get_border_spes(player, (i, 0), visited, borders_1)
            visited = list()
            for i in reversed(range(self.state.shape[0])):
                if ((i, self.state.shape[0] - 1) in visited):
                    continue
                if (self.state[i][self.state.shape[0] - 1] == 0):
                    visited.append((i, self.state.shape[0] - 1))
                    borders_2.append((i, self.state.shape[0] - 1))
                elif self.state[i][self.state.shape[0] - 1] == player:
                    visited.append((i, self.state.shape[0] - 1))
                    self.get_border_spes(
                        player, (i, self.state.shape[0] - 1), visited, borders_2)
        else:
            for i in range(self.state.shape[0]):
                if ((0, i) in visited):
                    continue
                if (self.state[0][i] == 0):
                    visited.append((0, i))
                    borders_1.append((0, i))
                elif self.state[0][i] == player:
                    visited.append((0, i))
                    self.get_border_spes(player, (0, i), visited, borders_1)
            visited = list()
            for i in reversed(range(self.state.shape[0])):
                if ((self.state.shape[0] - 1, i) in visited):
                    continue
                if (self.state[self.state.shape[0] - 1][i] == 0):
                    visited.append(
                        (self.state.shape[0] - 1, i))
                    borders_2.append((self.state.shape[0] - 1, i))
                elif self.state[self.state.shape[0] - 1][i] == player:
                    visited.append((self.state.shape[0] - 1, i))
                    self.get_border_spes(
                        player, (self.state.shape[0] - 1, i), visited, borders_2)
        return (borders_1, borders_2)

    def dijkstra(self, player):
        first_side = list()
        second_side = list()
        first_side, second_side = self.get_border_points2(player)
        # first_side = self.border_editor(first_side, player)
        # second_side = self.border_editor(second_side, player)
        (x, y) = (None, None)
        min_val = float('inf')
        for i in range(len(first_side)):
            for j in range(len(second_side)):
                try:
                    min_val = min(min_val, nx.shortest_path_length(
                        self.dijkstra_graph, first_side[i], second_side[j]) + 1)
                except Exception as e:
                    continue
        return min_val

    def search_with_heuristic(self, depth, max_depth):

        # if end game is reached return
        f = self.logic.is_game_over(self.starting_player, self.state)
        if f is None:
            # print("L")
            f = self.logic.is_game_over(self.other_player, self.state)
            # print(f)
        # possible stack overflow problem
        # print(f)
        # if game is over => we reached a leaf node
        if self.logic.GAME_OVER == True:
            # self.turn_state = not self.turn_state
            self.logic.GAME_OVER = False
            # we should try minimax for each player
            # if neither player won:
            if f is None:
                return (-0.5, None)  # less better than equal pathes
            # if starting player won:
            elif f == self.starting_player:
                return (5, None)
            # if the opposing player won:
            else:
                return (2, None)
        # if not make all choices
        elif depth == max_depth:
            # call graph generation: depending on which player's turn
            self.graph_generator2(self.starting_player)
            # call dijkstra for the first
            first = self.dijkstra(self.starting_player)
            self.graph_generator2(self.other_player)
            second = self.dijkstra(self.other_player)
            # call dijkstra for the second
            # print("************************")
            # print("first is: ", first)
            # print("the second is: ", second)
            # print("************************")
            # compare the two and return a score
            return (-first, None)
        else:
            l = self.logic.get_possible_moves(
                self.state)  # finds all the possible moves
            if self.turn_state == True:
                self.turn_state = not self.turn_state
                the_m = - inf  # check if it works
                # in case we are in level 0
                if (depth == 0):
                    m = 0
                    res_t = (None, None)
                    for i in l:
                        self.state[i[0]][i[1]] = self.starting_player
                        print("******************************************")
                        print("the other player chose the following path: ")
                        m = self.search_with_heuristic(depth + 1, max_depth)[0]
                        print("for : ", i, "we get the distance ", m)
                        print("*****************************************")
                        # if(m == -1):
                        #     print("we can go to ", i)
                        # we can implement random to choose any
                        # print("m is", m)
                        if (the_m < m):
                            the_m = m
                            res_t = i
                        self.state[i[0]][i[1]] = 0

                    self.turn_state = not self.turn_state
                    # print(res_t)
                    first_side, second_side = self.get_border_points2(
                        self.starting_player)
                    # print("*******************************************")
                    # print(first_side)
                    # print(second_side)
                    # print("*****************************************")

                    return res_t
                else:
                    for i in l:
                        # should be less than the optimal value
                        self.state[i[0]][i[1]] = self.starting_player
                        the_m = max(the_m, self.search_with_heuristic(
                            depth + 1, max_depth)[0])
                        self.state[i[0]][i[1]] = 0
                    print("the chosen m is : ", the_m)

            # it is the opposing player's turn:
            else:
                # to be removed#
                # *********************************************************#
                self.turn_state = not self.turn_state
                the_m = inf
                m = 0
                res_t = (None, None)
                for i in l:
                    self.state[i[0]][i[1]] = self.other_player
                    m = self.search_with_heuristic(depth + 1, max_depth)[0]
                    # if(m == -1):
                    #     print("we can go to ", i)
                    # we can implement random to choose any
                    # print("m is", m)
                    if (the_m > m):
                        the_m = m
                        res_t = i
                    self.state[i[0]][i[1]] = 0
                print("the chosen m is: ", res_t)

                # *********************************************************#
                # the_m = inf
                # self.turn_state = not self.turn_state
                # m = 0
                # # this won't be accessed most of the time
                # # I will remove the p_next == 0 since it is never used
                # # could be accessed at the end
                # for i in l:
                #     # print(the_m)
                #     self.state[i[0]][i[1]] = self.other_player
                #     # print(self.state[i[0]][i[1]])
                #     the_m = min(the_m, self.search_with_heuristic(
                #         depth + 1, max_depth)[0])
                #     self.state[i[0]][i[1]] = 0
            self.turn_state = not self.turn_state
            return (the_m, None)




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
            # print(eval)
            if eval > best_eval:
                best_eval = eval
                best_move = child.move

        return best_move
