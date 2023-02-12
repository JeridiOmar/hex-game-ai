import logging

from rich import print
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

from classes.tournament import Tournament


def main(args):
    """
    Runs a tournament with a list of arguments that contain, in order:
       * the size of the board,
       * the playing mode, i.e., "ai_vs_ai", "man_vs_ai",
       * the game counter (why not? though it should be always zero),
       * the number of games to play.

    In the "ai_vs_ai" mode, there is a real competition.
    In contrast, in the "man_vs_ai" mode, there is a single match, i.e., the last parameter is ineffective.
    """
    arena = Tournament(args)

    if MODE == "ai_vs_ai":
        arena.championship()
    elif MODE == "man_vs_ai":
        arena.single_game(black_starts=True)
    else:
        assert False, "SHOULD NOT HAPPEN UNLESS YOU IMPLEMENT THE man_vs_man VERSION"


if __name__ == "__main__":
    log = logging.getLogger("rich")

    print("Do you want to play against AI (type 1) or let the AI play alone (type 2)?", end="\t")
    gamemode = int(input())
    if gamemode == 1:
        MODE = "man_vs_ai"
        print("You will be playing as the [bold]BLACK player[/bold]!")
    else:
        MODE = "ai_vs_ai"
        print("Ok, let the AI play alone.")

    BOARD_SIZE = 5
    GAME_COUNT = 0
    N_GAMES = 10

    main([BOARD_SIZE, MODE, GAME_COUNT, N_GAMES])
