import os
import pickle
import logging
from rich import print
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# Hide Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "show"
import pygame

from classes.game import Game

import pandas as pd


class Tournament:
    def __init__(self, args: list):
        """
        Initialises a tournament with:
           * the size of the board,
           * the playing mode, i.e., "ai_vs_ai", "man_vs_ai",
           * the game counter,
           * the number of games to play.
        """
        self.args = args
        self.BOARD_SIZE = args[0]
        self.MODE = args[1]
        self.GAME_COUNT = args[2]
        self.N_GAMES = args[3]
        self.game = None

    def single_game(self, black_starts: bool = True) -> int:
        """
        Runs a single game between two opponents.

        @return   The number of the winner, either 1 or 2, for black and white respectively.
        """
        pygame.init()
        pygame.display.set_caption("Polyline")

        game = Game(board_size=self.BOARD_SIZE, mode=self.MODE, black_starts=black_starts)
        self.game = game
        game.get_game_info([self.BOARD_SIZE, self.MODE, self.GAME_COUNT])
        while game.winner is None:
            game.play()
        return game.winner

    def championship(self):
        """
        Runs a number of games between the same two opponents.
        """
        # avg fields will be calculated after the end of N games
        statistics = {
            "number_games": 0,
            "black_home_wins": 0,
            "white_home_wins": 0,
            "black_away_wins": 0,
            "white_away_wins": 0,
            "black_number_moves": 0,
            "black_avg_number_moves": 0,
            "white_number_moves": 0,
            "white_avg_number_moves": 0,
            "black_time": 0,
            "black_avg_time": 0,
            "white_time": 0,
            "white_avg_time": 0,
            "black_peak_memory": 0,
            "white_peak_memory": 0
        }
        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _
            # First half of the tournament started by one player.
            # Remaining half started by other player (see "no pie rule")
            winner = self.single_game(black_starts=self.GAME_COUNT < self.N_GAMES / 2)
            # print(winner)
            print("Moves of player1: ", self.game.black_player_moves, "   Moves Player 2  :",
                  self.game.white_player_moves)
            # print(self.GAME_COUNT, self.N_GAMES)
            statistics["black_time"] += self.game.black_player_time
            statistics["white_time"] += self.game.white_player_time
            statistics["black_number_moves"] += self.game.black_player_moves
            statistics["white_number_moves"] += self.game.white_player_moves
            statistics["black_number_moves"] += self.game.black_player_moves
            if self.game.black_player_memory_peak > statistics["black_peak_memory"]:
                statistics["black_peak_memory"] = self.game.black_player_memory_peak
            if self.game.white_player_memory_peak > statistics["white_peak_memory"]:
                statistics["white_peak_memory"] = self.game.white_player_memory_peak

            if winner == self.game.ui.BLACK_PLAYER:
                if self.GAME_COUNT < self.N_GAMES / 2:
                    statistics["black_home_wins"] += 1
                else:
                    statistics["black_away_wins"] += 1

            else:
                if self.GAME_COUNT >= self.N_GAMES / 2:
                    statistics["white_home_wins"] += 1
                else:
                    statistics["black_home_wins"] += 1

        statistics["number_games"] = self.N_GAMES
        statistics["black_avg_number_moves"] = statistics["black_number_moves"] / statistics["number_games"]
        statistics["white_avg_number_moves"] = statistics["white_number_moves"] / statistics["number_games"]
        statistics["black_avg_time"] = statistics["black_time"] / statistics["number_games"]
        statistics["white_avg_time"] = statistics["white_time"] / statistics["number_games"]
        log = logging.getLogger("rich")
        log.debug(statistics)
        black_score = (45 / 100) * statistics["black_away_wins"] + (35 / 100) * statistics["black_home_wins"] + (
                    15 / 100) * statistics["black_avg_time"] + (5 / 100) * statistics["black_peak_memory"]
        white_score = (45 / 100) * statistics["white_away_wins"] + (35 / 100) * statistics["white_home_wins"] + (
                    15 / 100) * statistics["white_avg_time"] + (5 / 100) * statistics["white_peak_memory"]
        log.debug(black_score)
        log.debug(white_score)
        print("Design your own evaluation measure!")
