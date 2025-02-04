import random
import pickle

# Definición de constantes
EMPTY = 0
WHITE_NORMAL = 1
WHITE_KING = 2
BLACK_NORMAL = -1
BLACK_KING = -2
BOARD_SIZE = 4

# Variables globales
q_table = {}
epsilon = 1.0  # Se cargará desde el archivo si existe

def load_q_table(filename="oponente_q_table.pkl"):
    """Carga el Q-table y el valor de epsilon desde un archivo."""
    global q_table, epsilon
    try:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            q_table = data.get("q_table", {})
            epsilon = data.get("epsilon", 1.0)
        print(f"Q-table y epsilon cargados desde {filename}.")
    except FileNotFoundError:
        print("No se encontró un Q-table existente. Iniciando desde cero.")
        q_table = {}
        epsilon = 1.0

def get_possible_actions(board,player):
        """Devuelve una lista de movimientos posibles para un jugador."""
        actions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = board[i][j]
                if (player > 0 and piece > 0) or (player < 0 and piece < 0):
                    if abs(piece) == 1:  # Ficha normal
                        directions = [(1, -1), (1, 1)] if player == WHITE_NORMAL else [(-1, -1), (-1, 1)]
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                if board[ni][nj] == EMPTY:
                                    actions.append(((i, j), (ni, nj)))
                                elif board[ni][nj] * player < 0:  # Ficha del oponente
                                    ni2, nj2 = ni + di, nj + dj
                                    if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and board[ni2][nj2] == EMPTY:
                                        actions.append(((i, j), (ni2, nj2)))
                    elif abs(piece) == 2:  # Dama
                        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        for di, dj in directions:
                            for step in range(1, BOARD_SIZE):
                                ni, nj = i + di * step, j + dj * step
                                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                    if board[ni][nj] == EMPTY:
                                        actions.append(((i, j), (ni, nj)))
                                    elif board[ni][nj] * player < 0:
                                        ni2, nj2 = ni + di, nj + dj
                                        if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and board[ni2][nj2] == EMPTY:
                                            actions.append(((i, j), (ni2, nj2)))
                                        break
                                    else:
                                        break
                                else:
                                    break
        #print(f"Acciones disponibles para el jugador {player}: {actions}")
        return actions

def get_q_value(state, action):
    """Obtiene el valor Q para un estado y una acción."""
    return q_table.get((tuple(map(tuple, state)), action), 0)

def choose_action(state, possible_actions):
    """Elige una acción usando una política epsilon-greedy."""
    if random.random() < epsilon:
        return random.choice(possible_actions)
    q_values = [get_q_value(state, action) for action in possible_actions]
    max_q = max(q_values)
    return possible_actions[q_values.index(max_q)]

def get_best_move(board, player):
    """Retorna el mejor movimiento para el jugador dado el estado del tablero."""
    possible_actions = get_possible_actions(board, player)
    if possible_actions:
        return choose_action(board, possible_actions)
    return None  # No hay movimientos disponibles

# Cargar la Q-table antes de empezar
load_q_table()
