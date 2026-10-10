
BOARD_SIZE = 3
EMPTY = 0
X = 1
O = -1


class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        """Start a new game."""
        self.board = [EMPTY] * 9
        self.current_player = X
        self.winner = None
        self.done = False

    def get_legal_actions(self):
        """Return the empty cell positions, numbered 0 to 8."""
        if self.done:
            return []

        return [
            position
            for position, cell in enumerate(self.board)
            if cell == EMPTY
        ]

    def make_move(self, position):
        """Place the current player's mark at a valid position."""
        if self.done:
            return False

        if position not in self.get_legal_actions():
            return False

        self.board[position] = self.current_player

        if self._has_won(self.current_player):
            self.winner = self.current_player
            self.done = True
        elif EMPTY not in self.board:
            self.done = True
        else:
            self.current_player *= -1

        return True

    def _has_won(self, player):
        winning_lines = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        return any(
            all(self.board[position] == player for position in line)
            for line in winning_lines
        )

    def get_state(self):
        """Return a board copy for the AI agent."""
        return tuple(self.board)

    def get_result(self):
        """Return X, O, 'draw', or None if the game is ongoing."""
        if not self.done:
            return None

        if self.winner is not None:
            return self.winner

        return "draw"