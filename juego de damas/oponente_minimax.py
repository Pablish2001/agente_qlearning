# agente_oponente.py
BOARD_SIZE = 4
BLACK_NORMAL = -1
WHITE_NORMAL = 1
BLACK_KING = -2
WHITE_KING = 2
EMPTY = 0
def generate_moves(board, player):
    """Devuelve una lista de movimientos posibles para un jugador."""
    actions = []
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if (player > 0 and piece > 0) or (player < 0 and piece < 0):
                if abs(piece) == 1:  # Ficha normal
                    # Direcciones de movimiento para fichas normales
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
    return actions



# Función para manejar movimientos en el tablero
def handle_move(board, move):
    """Realiza un movimiento en el tablero y devuelve el nuevo estado."""
    temp_board = [row[:] for row in board]  # Copia profunda del tablero
    (i, j), (ni, nj) = move
    piece = temp_board[i][j]
    temp_board[i][j] = EMPTY
    temp_board[ni][nj] = piece

    # Eliminar pieza capturada (si es un salto)
    if abs(ni - i) == 2:
        captured_row, captured_col = (i + ni) // 2, (j + nj) // 2
        temp_board[captured_row][captured_col] = EMPTY

    # Coronación de fichas
    if ni == 0 and piece == WHITE_NORMAL:  # Ficha blanca llega al extremo negro
        temp_board[ni][nj] = WHITE_KING
    elif ni == BOARD_SIZE - 1 and piece == BLACK_NORMAL:  # Ficha negra llega al extremo blanco
        temp_board[ni][nj] = BLACK_KING

    return temp_board

# Función para verificar si el juego ha terminado
def is_game_over(board):
    """Verifica si el juego ha terminado."""
    white_pieces = sum(row.count(WHITE_NORMAL) + row.count(WHITE_KING) for row in board)
    black_pieces = sum(row.count(BLACK_NORMAL) + row.count(BLACK_KING) for row in board)
    if white_pieces == 0:
        return True, BLACK_NORMAL  # Negras ganan
    if black_pieces == 0:
        return True, WHITE_NORMAL  # Blancas ganan
    if not generate_moves(board, WHITE_NORMAL) and not generate_moves(board, BLACK_NORMAL):
        return True, 0  # Empate
    return False, None

# Algoritmo Minimax con poda alfa-beta
def minimax(board, depth, alpha, beta, maximizing_player):
    """Algoritmo Minimax con poda alfa-beta."""
    if depth == 0 or is_game_over(board)[0]:
        return evaluate(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        possible_moves = generate_moves(board, WHITE_NORMAL)  # Movimientos para WHITE_NORMAL
        if not possible_moves:  # Si no hay movimientos disponibles
            return evaluate(board), None
        for move in possible_moves:
            new_board = handle_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        possible_moves = generate_moves(board, BLACK_NORMAL)  # Movimientos para BLACK_NORMAL
        if not possible_moves:  # Si no hay movimientos disponibles
            return evaluate(board), None
        for move in possible_moves:
            new_board = handle_move(board, move)
            eval, _ = minimax(new_board, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Función de evaluación mejorada
def evaluate(board):
    """Evalúa el tablero desde la perspectiva de las blancas (maximizando)."""
    score = 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            piece = board[i][j]
            if piece == WHITE_NORMAL:
                score += 10 + (BOARD_SIZE - 1 - i)  # Fichas blancas más cercanas a la coronación valen más
            elif piece == BLACK_NORMAL:
                score -= 10 + i  # Fichas negras más cercanas a la coronación valen menos
            elif piece == WHITE_KING:
                score += 20  # Las damas blancas valen más
            elif piece == BLACK_KING:
                score -= 20  # Las damas negras valen menos
    return score

# Función principal del agente oponente
def ai_move(board):
    """Realiza el mejor movimiento para el agente oponente."""
    # Accede al tablero interno de la clase Board
    board_matrix = board.board
    _, move = minimax(board_matrix, 5, float('-inf'), float('inf'), True)
    if move:
        print("blanco: ",move)
        return move  # Devuelve el movimiento para que el tablero lo procese
    else:
        print("No hay movimientos disponibles.")
        return None