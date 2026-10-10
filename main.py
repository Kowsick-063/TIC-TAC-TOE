
import os
import random
import pygame
from game import TicTacToe, X, O
from agent import QLearningAgent

pygame.init()

# -------------------- WINDOW --------------------

WIDTH, HEIGHT = 1000, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe")
clock = pygame.time.Clock()

# -------------------- LIGHT THEME --------------------

BG = (248, 249, 252)
WHITE = (255, 255, 255)
TEXT = (35, 43, 58)
MUTED = (125, 135, 150)
BORDER = (229, 233, 240)
BLUE = (57, 112, 232)
BLUE_LIGHT = (237, 243, 255)
PINK = (222, 91, 113)
GREEN = (35, 155, 105)

# -------------------- FONTS --------------------

def font(size, bold=False):
    return pygame.font.SysFont("segoeui", size, bold=bold)


title_font = font(28, True)
heading = font(18, True)
body = font(15)
small = font(13)
symbol_font = font(76, True)

# -------------------- TRAINED AI --------------------

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.pkl")

if os.path.exists(MODEL_PATH):
    agent = QLearningAgent.load(MODEL_PATH)
    print(f"Loaded trained Q-Learning agent ({len(agent)} states).")
else:
    agent = None
    print("q_table.pkl not found - run 'python train.py' first. "
          "Falling back to a random opponent.")

# -------------------- GAME STATE --------------------

game = TicTacToe()
mode = "ai"

stats = {
    "you": 0,
    "ai": 0,
    "p2": 0,
    "draws": 0,
    "games": 0,
}

recorded = False

# -------------------- LAYOUT --------------------

board_rect = pygame.Rect(55, 170, 420, 420)
cell = 140

new_btn = pygame.Rect(55, 615, 130, 42)
mode_btn = pygame.Rect(195, 615, 175, 42)

# -------------------- DRAW HELPERS --------------------

def text(value, x, y, color=TEXT, f=body, center=False):
    image = f.render(str(value), True, color)

    if center:
        rect = image.get_rect(center=(x, y))
    else:
        rect = image.get_rect(topleft=(x, y))

    screen.blit(image, rect)


def card(rect):
    pygame.draw.rect(screen, WHITE, rect, border_radius=12)
    pygame.draw.rect(
        screen, BORDER, rect, width=1, border_radius=12
    )


def button(rect, label, active=False):
    fill = BLUE_LIGHT if active else WHITE
    outline = BLUE if active else BORDER

    pygame.draw.rect(screen, fill, rect, border_radius=9)
    pygame.draw.rect(
        screen, outline, rect, width=1, border_radius=9
    )

    text(
        label,
        rect.centerx,
        rect.centery,
        BLUE if active else TEXT,
        body,
        center=True,
    )


# -------------------- GAME FUNCTIONS --------------------

def new_game():
    global recorded
    game.reset()
    recorded = False


def ai_move():
    """The trained Q-Learning agent plays O (greedy, no learning)."""
    if mode == "ai" and not game.done and game.current_player == O:
        actions = game.get_legal_actions()

        if not actions:
            return

        if agent is not None:
            action = agent.choose_action(
                game.get_state(), actions, O, training=False
            )
        else:
            action = random.choice(actions)

        game.make_move(action)


def record_result():
    """Update statistics once when a match ends."""
    global recorded

    if not game.done or recorded:
        return

    recorded = True
    stats["games"] += 1

    result = game.get_result()

    if result == X:
        stats["you"] += 1
    elif result == O:
        if mode == "ai":
            stats["ai"] += 1
        else:
            stats["p2"] += 1
    else:
        stats["draws"] += 1


# -------------------- DRAW INTERFACE --------------------

def draw():
    screen.fill(BG)

    # Header
    text("Tic-Tac-Toe", 35, 25, TEXT, title_font)
    text(
        "Human vs a Q-Learning agent" if agent else "A simple game with reinforcement learning",
        37, 62, MUTED, small
    )

    text(
        "SESSION",
        960, 40, MUTED, small, center=True
    )

    # Left: game board card
    card(pygame.Rect(30, 105, 470, 525))

    text("GAME", 55, 125, MUTED, small)

    # Status message
    if game.done:
        result = game.get_result()

        if result == X:
            status = "You won!"
            status_color = GREEN
        elif result == O:
            status = "Player  won!" if mode == "human" else "AI won!"
            status_color = PINK
        else:
            status = "It's a draw!"
            status_color = BLUE

    elif mode == "ai" and game.current_player == O:
        status = "AI's turn"
        status_color = PINK

    else:
        status = (
            "Your turn"
            if game.current_player == X
            else "Player's turn"
        )
        status_color = BLUE

    text(
        status, 265, 148, status_color, heading, center=True
    )

    # Board background
    pygame.draw.rect(
        screen, WHITE, board_rect, border_radius=10
    )

    # Grid lines
    for i in (1, 2):
        pygame.draw.line(
            screen,
            BORDER,
            (board_rect.x + i * cell, board_rect.y),
            (board_rect.x + i * cell, board_rect.bottom),
            2,
        )

        pygame.draw.line(
            screen,
            BORDER,
            (board_rect.x, board_rect.y + i * cell),
            (board_rect.right, board_rect.y + i * cell),
            2,
        )

    # X and O symbols
    for position, value in enumerate(game.board):
        if value == 0:
            continue

        row, col = divmod(position, 3)

        symbol = "X" if value == X else "O"
        color = BLUE if value == X else PINK

        center_x = board_rect.x + col * cell + cell // 2
        center_y = board_rect.y + row * cell + cell // 2

        text(
            symbol,
            center_x,
            center_y,
            color,
            symbol_font,
            center=True,
        )

    # Controls
    button(new_btn, "New game")
    button(
        mode_btn,
        "Mode: Human vs AI" if mode == "ai"
        else "Mode: Two players",
        active=True,
    )

    # Right: statistics
    x = 525

    card(pygame.Rect(x, 105, 445, 145))
    text("YOUR STATISTICS", x + 20, 125, MUTED, small)

    stats_items = [
        ("Games", stats["games"]),
        ("Wins", stats["you"]),
        ("Draws", stats["draws"]),
    ]

    for i, (label, value) in enumerate(stats_items):
        cx = x + 78 + i * 140

        text(
            value,
            cx,
            172,
            BLUE if label == "Wins" else TEXT,
            title_font,
            center=True,
        )

        text(
            label,
            cx,
            207,
            MUTED,
            small,
            center=True,
        )

    # Leaderboard
    card(pygame.Rect(x, 270, 445, 205))

    text("LEADERBOARD", x + 20, 290, TEXT, heading)
    text("PLAYER", x + 25, 328, MUTED, small)
    text("WINS", x + 390, 328, MUTED, small, center=True)

    players = [
        ("You", stats["you"], BLUE),
        (
            "AI" if mode == "ai" else "Player",
            stats["ai"] if mode == "ai" else stats["p2"],
            PINK,
        ),
    ]

    for i, (name, wins, color) in enumerate(players):
        yy = 360 + i * 48

        pygame.draw.line(
            screen,
            BORDER,
            (x + 20, yy + 30),
            (x + 425, yy + 30),
            1,
        )

        pygame.draw.circle(
            screen, color, (x + 32, yy), 6
        )

        text(name, x + 50, yy - 10, TEXT, body)
        text(wins, x + 390, yy - 10, TEXT, body, center=True)

    # Simple footer — no Recent Games section
    text(
        "Tip: switch modes to play with a friend.",
        525, 520, MUTED, small
    )

    text(
        "Wins and statistics are tracked for this session.",
        525, 548, MUTED, small
    )

    pygame.display.flip()


# -------------------- MAIN LOOP --------------------

running = True

while running:
    draw()

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos

            if new_btn.collidepoint(pos):
                new_game()

            elif mode_btn.collidepoint(pos):
                mode = "human" if mode == "ai" else "ai"
                new_game()

            elif board_rect.collidepoint(pos) and not game.done:

                # AI plays O; the human controls X.
                if mode == "ai" and game.current_player == O:
                    continue

                col = (pos[0] - board_rect.x) // cell
                row = (pos[1] - board_rect.y) // cell
                position = row * 3 + col

                if game.make_move(position):
                    if not game.done:
                        ai_move()

                    record_result()

    clock.tick(60)

pygame.quit()