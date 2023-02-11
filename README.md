# hex-game-ai
# Comparison procedure

To compare the different algorithms, several statistics were collected
and used to calculate a score.

The statistics collected include the number of games played, the number
of wins for each player when starting first and second, the average
number of moves during the game, the average time taken for each move,
and the peak memory usage. The statistics were collected and recorded in
the statistics dictionary.

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

The score calculation was performed using a weighted sum of the
different statistics. The score for the player was calculated as a
function of its wins as the first and second player, average computation
time for next move, average number of moves during the game, and peak
memory usage. The weights were chosen to reflect the relative importance
of each statistic in determining the overall performance of the
algorithm.

The calculation of the score is given by the equation:

*s**c**o**r**e* = *w*<sub>1</sub> * *p**l**a**y**e**r**A**w**a**y**W**i**n**s* + *w*<sub>2</sub> * *p**l**a**y**e**r**H**o**m**e**W**i**n**s* + *w*<sub>3</sub> * *p**l**a**y**e**r**A**v**g**T**i**m**e* + *w*<sub>4</sub> * *p**l**a**y**e**r**P**e**a**k**M**e**m**o**r**y*

where:

*w*<sub>1</sub> = 45%

*w*<sub>2</sub> = 35%

*w*<sub>3</sub> = 15%

*w*<sub>4</sub> = 5%

Finally, it is worth mentioning that there are other criteria that could
be used to evaluate the performance of algorithms in the Hex game. For
example, the ability to find win moves or winning positions in a timely
manner could also be considered. This is similar to the concept of
checkmate in chess, where a player is able to win the game by putting
the opponent in a position where they have no legal moves. Implementing
this criterion would provide a more complete evaluation of the
algorithms and could lead to a better understanding of their strengths
and weaknesses. However, due to limited time and resources, this
criterion was not considered in the current study. In conclusion, the
comparison procedure described in this section allowed for a systematic
evaluation of different algorithms in the hex game. The statistics
collected and the calculation of the score provided a quantitative
measure of the performance of each algorithm, enabling the
identification of the best-performing approach.

# Minimax Algorithm

............

# Minimax with *α* − *β* pruning 

In this version of the hex game, the minimax function uses alpha-beta
pruning to find the best move for the current player. In this
implementation, there is no heuristic and the depth of the search tree
is not limited. The function returns 1 if the game is won by the black
player, -1 if the game is won by the white player and 0 if the game is
drawn.

Alpha-beta pruning is a technique used in minimax algorithms to reduce
the number of nodes that need to be evaluated. The algorithm maintains
two values, alpha and beta, which represent the best possible score for
the maximizing player (alpha) and the best possible score for the
minimizing player (beta). The function starts by setting alpha to
negative infinity and beta to positive infinity. The value of alpha and
beta are then updated during the search as better scores are found. If
at any point, the value of beta becomes less than or equal to alpha, the
function stops the search and returns the current value as it is
guaranteed that this is the optimal value. This allows the algorithm to
significantly reduce the number of nodes that need to be evaluated and
thus, improve the efficiency of the search.

The normal minimax algorithm had a limitation of executing only for
boards with size *n* = 3. However, with the implementation of alpha-beta
pruning, it became possible to execute the algorithm in reasonable time
for boards of size *n* = 4 and slower for boards of size *n* = 5. This
demonstrates the efficiency improvement of using the alpha-beta pruning
technique in conjunction with the minimax algorithm.

# Heuristics

This section of our analysis focuses on incorporating heuristics into
our minimax algorithm to improve its performance. By using heuristics,
we can limit the depth of the minimax search tree and evaluate
unfinished boards more efficiently.

A heuristic is a function that assigns a score to a given board state,
allowing us to make an estimate of which move is likely to be the best.
By using heuristics, we can prune the minimax search tree and avoid
exploring branches that are unlikely to lead to a winning outcome.

Below, we will describe the various heuristics we used to enhance the
performance of our minimax algorithm. These heuristics will be defined
and explained in detail in subsequent subsections.

## distance to opposite side

The first heuristic function calculates the total distance of all pieces
of a specific player to the opposite side of the board. The player can
be either BLACK PLAYER or WHITE PLAYER. For BLACK PLAYER, the distance
of each piece to the bottom row (n-1) is computed. For WHITE PLAYER, the
distance of each piece to the rightmost column (m-1) is computed. The
distance is calculated as the absolute difference between the piece’s
position and the opposite side. The sum of all distances is then
returned as the result. This heuristic function helps to evaluate the
progress of a player’s pieces towards the opposite side of the board.

## number of paths to the other side

The second heuristic improves upon the first one by considering not only
the number of pieces the player has on the board, but also the number of
possible paths that the player has to their own side of the board. This
takes into account not just the current state of the game, but also the
potential future moves of both players. By evaluating the number of
paths the player has, the second heuristic provides a more comprehensive
evaluation of the player’s position and helps to determine a more
optimal move in the game.

The heuristic is based on the idea that a player who is closer to the
opposite side has a higher chance of forming a connection, and therefore
a higher score. The function "number_of_paths_other_side" takes in the
current board and the player as inputs, and returns the count of paths
the player has to their own side.

The function "check_path_left_right" and "check_path_top_bottom" check
if a player’s stone at a given position (row, col) is part of a path
from left to right or from top to bottom, respectively. These functions
use the depth-first search functions "dfs_left_right" and
"dfs_top_bottom" to traverse the board and find a path to the opposite
side. If a path is found, the count is incremented.

## longest player chain

The third heuristic is called "Longest Chain". It evaluates the length
of the longest chain of each player on the board. The idea is that a
player with a longer chain is closer to winning, so this player is
likely to have a higher score.

The code implements the longest_chain function, which takes the current
state of the board and the player as inputs. It first sets the
max_chain_length to 0, and then iterates through each cell in the board
to find the player’s pieces. If the cell contains the player’s piece, it
calls the "\_dfs_chain_length" function to calculate the length of the
chain starting from this piece. The \_dfs_chain_length function uses
depth-first search to traverse the connected pieces of the same player
and count the number of pieces in the chain. The result of the
longest_chain function is the maximum chain length found among all the
player’s pieces on the board.

Finally, the longest_player_chain function returns the result of the
longest_chain function as the score for the player.
