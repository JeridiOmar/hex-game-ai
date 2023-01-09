import sys

import pygame
from rich.console import Console
from rich.table import Table

from classes.logic import Logic
from classes.ui import UI


class Game:
    def __init__(self, board_size: int, mode: str, black_starts: bool = True):
        """
        Initialisation of a new game with:
            * the size of the board,
            * the playing mode, i.e., "ai_vs_ai", "man_vs_ai",
            * which player starts, i.e., black (by default) or white.

        Besides, the user interface is initialised and displayed.

        Also, public variables are set to their initial values:
            * there is no current node (set to None), which is an integer representing the 1D coordinates in a numpy array,
            * there is no current winner (set to None), which is to become eventually either 1 or 2, respectively for the black and white player.

        Finally, a dictionary-based "method/function" allows to retrieve the player based on the parity (even/odd) of the current step in the game.
        """
        # Select mode
        self.modes = { "ai_vs_ai":  0,
                       "man_vs_ai": 0 }
        self.modes[mode] = 1

        # Does BLACK player start?
        self.turn_state = black_starts

        # Instantiate classes
        self.ui = UI(board_size, mode)
        self.logic = Logic(self.ui)

        # Initialize public variables
        self.node = None
        self.winner = None

        # Initialize dict-based "function"
        self.turn = { True:  self.ui.BLACK_PLAYER, 
                      False: self.ui.WHITE_PLAYER }

    def get_game_info(self, args) -> None:
        """
        Prints on the console the parameters of the game:
           * the board size,
           * the playing mode, i.e., "ai_vs_ai", "man_vs_ai",
           * the number of the game when in competition mode.
        """
        console = Console()

        table = Table(title="Polyline", show_header=True, header_style="bold cyan")
        table.add_column("Parameters", justify="center")
        table.add_column("Value", justify="right")
        table.add_row("Board size", str(args[0]))
        table.add_row("Mode", str(args[1]))
        table.add_row("Game", str(args[2]))

        console.print(table)

    def handle_events(self) -> None:
        """
        Deals with one step of a game from either player taking into account the user interface and the fact that the human player can quit the game.
        """
        if self.modes["man_vs_ai"]:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP or self.modes["ai_vs_ai"]:
                    self.run_turn()
        elif self.modes["ai_vs_ai"]:
            self.run_turn()
        else: assert False, "SHOULD NOT HAPPEN UNLESS YOU IMPLEMENT THE man_vs_man VERSION"

    def run_turn(self) -> None:
        """
        Actually runs one step of a game.

        @bug   Progress is not guaranteed by this procedure.
        """
        if   self.modes["ai_vs_ai"]:    node = None
        elif self.modes["man_vs_ai"]:   node = self.node
        else: assert False, "SHOULD NOT HAPPEN UNLESS YOU IMPLEMENT THE man_vs_man VERSION"

        # BLACK player's turn
        if not self.check_move(node, self.turn[self.turn_state]):
            pass
        # WHITE player's turn (AI)
        elif not self.check_move(None, self.turn[self.turn_state]):
            pass

    def check_move(self, node, player) -> bool:
        """
        Forbids playing on an already busy node by *not* applying the move.
        Should the move be effective, then it also changes the turn of the game as a side-effect.

        @bug   Notice that this can lead to infinite loops if the player always plays an invalid node!

        @return   True iff there is a winner after the given (or rejected) move.
        """
        try:
            self.winner = self.logic.get_action(node, player)
        except AssertionError:
            return False

        # Next turn
        self.turn_state = not self.turn_state

        # If there is a winner, break the loop
        return not self.get_winner()

    def get_winner(self) -> bool:
        """
        @return  Either the actual winner, i.e., either 1 or 2 for black and white, or 0 when there is not yet a winner.
        """
        if self.winner is not None:
            print("Player {} wins!".format(self.winner))
            return True
        else:
            return False

    def play(self) -> None:
        """
        Runs a full game.
        """
        self.ui.draw_board()

        if self.modes["man_vs_ai"]:
            self.node = self.ui.get_node_hover()
        pygame.display.update()
        self.ui.clock.tick(30)
        self.handle_events()
