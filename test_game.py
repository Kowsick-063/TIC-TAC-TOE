
from game import TicTacToe, X, O

game = TicTacToe()

# X plays top-left, O plays middle-left, X plays top-middle,
# O plays middle, and X completes the top row.
moves = [0, 3, 1, 4, 2]

for position in moves:
    assert game.make_move(position)

assert game.done
assert game.get_result() == X

print("Test 1 passed: X wins.")

# Test that an occupied cell cannot be played.
game.reset()
assert game.make_move(0)
assert not game.make_move(0)

print("Test 2 passed: Invalid moves are rejected.")

# Test a draw.
game.reset()
draw_moves = [0, 1, 2, 4, 3, 5, 7, 6, 8]

for position in draw_moves:
    assert game.make_move(position)

assert game.get_result() == "draw"

print("Test 3 passed: Draw detected.")