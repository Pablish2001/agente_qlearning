import random
import pygame
import sys
import pickle
from oponente_minimax import ai_move
import oponente_qlearning
import copy

# Constantes
EMPTY = 0
WHITE_NORMAL = 1
WHITE_KING = 2
BLACK_NORMAL = -1
BLACK_KING = -2
BOARD_SIZE = 4

# Configuración del tablero
CELL_SIZE = 100
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GOLD = (255, 215, 0)
CAPTURA = None
# Parámetros de Q-Learning
alpha = 0.7  # Tasa de aprendizaje
gamma = 0.99  # Factor de descuento
epsilon = 1.0  # Probabilidad de exploración inicial
epsilon_decay = 0.999  # Tasa de decaimiento de epsilon
epsilon_min = 0.1  # Valor mínimo de epsilon
num_episodes = 1000

class Board:
    def __init__(self):
        self.board = [
            [WHITE_NORMAL, 0, WHITE_NORMAL, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, BLACK_NORMAL, 0, BLACK_NORMAL]
        ]
        self.movidas_realizadas = 0

    def reset(self):
        """Reinicia el tablero al estado inicial."""
        self.board = [
            [WHITE_NORMAL, 0, WHITE_NORMAL, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, BLACK_NORMAL, 0, BLACK_NORMAL]
        ]
        self.movidas_realizadas = 0

    def get_possible_actions(self, player):
        """Devuelve una lista de movimientos posibles para un jugador."""
        actions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                piece = self.board[i][j]
                if (player > 0 and piece > 0) or (player < 0 and piece < 0):
                    if abs(piece) == 1:  # Ficha normal
                        directions = [(1, -1), (1, 1)] if player == WHITE_NORMAL else [(-1, -1), (-1, 1)]
                        for di, dj in directions:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                if self.board[ni][nj] == EMPTY:
                                    actions.append(((i, j), (ni, nj)))
                                elif self.board[ni][nj] * player < 0:  # Ficha del oponente
                                    ni2, nj2 = ni + di, nj + dj
                                    if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and self.board[ni2][nj2] == EMPTY:
                                        actions.append(((i, j), (ni2, nj2)))
                    elif abs(piece) == 2:  # Dama
                        directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                        for di, dj in directions:
                            for step in range(1, BOARD_SIZE):
                                ni, nj = i + di * step, j + dj * step
                                if 0 <= ni < BOARD_SIZE and 0 <= nj < BOARD_SIZE:
                                    if self.board[ni][nj] == EMPTY:
                                        actions.append(((i, j), (ni, nj)))
                                    elif self.board[ni][nj] * player < 0:
                                        ni2, nj2 = ni + di, nj + dj
                                        if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and self.board[ni2][nj2] == EMPTY:
                                            actions.append(((i, j), (ni2, nj2)))
                                        break
                                    else:
                                        break
                                else:
                                    break
        #print(f"Acciones disponibles para el jugador {player}: {actions}")
        return actions

    def make_move(self, move):
        """Realiza un movimiento en el tablero."""
        (i, j), (ni, nj) = move
        piece = self.board[i][j]
        self.board[ni][nj] = piece
        self.board[i][j] = EMPTY

        # Eliminar pieza capturada (si es un salto)
        if abs(ni - i) >= 2:
            direction_row = ni - i
            direction_col = nj - j
            # Calcular la posición de la ficha capturada (un espacio detrás del destino)
            captured_row = ni - direction_row // abs(direction_row)
            captured_col = nj - direction_col // abs(direction_col)
            self.board[captured_row][captured_col] = EMPTY

        # Coronación de fichas
        if ni == BOARD_SIZE - 1 and piece == WHITE_NORMAL:  # Ficha blanca llega al extremo negro
            self.board[ni][nj] = WHITE_KING
        elif ni == 0 and piece == BLACK_NORMAL:  # Ficha negra llega al extremo blanco
            self.board[ni][nj] = BLACK_KING

    def is_game_over(self):
        """Verifica si el juego ha terminado."""
        white_pieces = sum(row.count(WHITE_NORMAL) + row.count(WHITE_KING) for row in self.board)
        black_pieces = sum(row.count(BLACK_NORMAL) + row.count(BLACK_KING) for row in self.board)
        if white_pieces == 0:
            return True, BLACK_NORMAL  # Negras ganan
        if black_pieces == 0:
            return True, WHITE_NORMAL  # Blancas ganan
        if self.movidas_realizadas >= 64:  # Empate después de 20 movimientos
            return True, 0
        return False, None

    def skip_turn(self, player):
        """Comprueba si un jugador no tiene movimientos y pasa el turno."""
        return len(self.get_possible_actions(player)) == 0


class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.5  # Se sobrescribirá si hay un archivo guardado
        self.epsilon_decay = 0.995  # Tasa de decaimiento
        self.epsilon_min = 0.01  # Valor mínimo de epsilon
        self.load_q_table()  # Cargar datos al iniciar

    def get_q_value(self, state, action):
        """Obtiene el valor Q para un estado y una acción."""
        return self.q_table.get((tuple(map(tuple, state)), action), 0)

    def update_q_value(self, state, action, reward, next_state):
        """Actualiza el valor Q para un estado y una acción."""
        max_next_q = max(
            [self.get_q_value(next_state, a) for a in board.get_possible_actions(BLACK_NORMAL)],
            default=0
        )
        current_q = self.get_q_value(state, action)
        self.q_table[(tuple(map(tuple, state)), action)] = current_q + alpha * (reward + gamma * max_next_q - current_q)

    def choose_action(self, state, possible_actions):
        """Elige una acción usando una política epsilon-greedy."""
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        q_values = [self.get_q_value(state, action) for action in possible_actions]
        max_q = max(q_values)
        return possible_actions[q_values.index(max_q)]

    def decay_epsilon(self):
        """Reduce el valor de epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_q_table(self, filename="q_table.pkl"):
        """Guarda el Q-table y el valor de epsilon en un archivo."""
        with open(filename, "wb") as f:
            pickle.dump({"q_table": self.q_table, "epsilon": self.epsilon}, f)
        print(f"Q-table y epsilon guardados en {filename}.")

    def load_q_table(self, filename="q_table.pkl"):
        """Carga el Q-table y el valor de epsilon desde un archivo."""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.q_table = data.get("q_table", {})
                self.epsilon = data.get("epsilon", 1.0)  # Recupera epsilon o usa el valor por defecto
            print(f"Q-table y epsilon cargados desde {filename}.")
        except FileNotFoundError:
            print("No se encontró un Q-table existente. Iniciando desde cero.")
            self.q_table = {}
            self.epsilon = 1.5  # Valor inicial si no hay archivo guardado


# Funciones auxiliares
def get_reward(board, move, player, previous_board,movidas,moveopponent=None):
    """Calcula la recompensa basada en el movimiento y el estado del tablero."""
    reward = 0
    global CAPTURA
    (i, j), (ni, nj) = move
    if moveopponent:
        (iop, jop), (niop, njop) = moveopponent
    opponent = WHITE_NORMAL if player == BLACK_NORMAL else BLACK_NORMAL
    fichas_oponente = sum(row.count(opponent) + row.count(opponent * 2) for row in board)
    fichas_player = sum(row.count(player) + row.count(player * 2) for row in board)
  # Recompensa por capturar una ficha
    if movidas >12:
        reward -= movidas-12
    captured = abs(ni - i) >= 2  # Movimiento de captura (salto)
    if captured:
        direction_row = ni - i
        direction_col = nj - j

        # Calcular la posición de la ficha capturada (un espacio detrás del destino)
        captured_row = ni - direction_row // abs(direction_row)
        captured_col = nj - direction_col // abs(direction_col)
        if previous_board[captured_row][captured_col] == opponent or previous_board[captured_row][captured_col] == opponent*2:
            if previous_board[captured_row][captured_col] == opponent*2:
                reward += 120  # Recompensa por captura
                print("Captura dama 120+")  # Mensaje de confirmación
            else:
                reward += 100  # Recompensa por captura
                print("Captura normal 100+")  # Mensaje de confirmación
            if fichas_oponente == 0:
                reward += 500
                print("Captura victoria 500+")  # Mensaje de confirmación
        else:
            captured = None

    if moveopponent:
        capturedopp = abs(niop - iop) >= 2  # Movimiento de captura (salto)
        if capturedopp and CAPTURA is not None:
            direction_row = niop - iop
            direction_col = njop - jop

            # Calcular la posición de la ficha capturada (un espacio detrás del destino)
            captured_row = niop - direction_row // abs(direction_row)
            captured_col = njop - direction_col // abs(direction_col)
            print("pieza: ",CAPTURA[captured_row][captured_col],captured_row, captured_col)
            if CAPTURA[captured_row][captured_col] == player or CAPTURA[captured_row][captured_col] == player*2:
                if fichas_player == 0:
                    reward -= 500
                    print("pierde ficha importante -500")  # Mensaje de confirmación
                elif CAPTURA[captured_row][captured_col] == player*2:
                    reward -= 120
                    print("pierde dama -120")  # Mensaje de confirmación
                else:
                    reward -= 100  # Recompensa por captura
                    print("pierde ficha -100")  # Mensaje de confirmación




    # Evaluar seguridad de las fichas (peligro y seguridad)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player or board[row][col] == player * 2:
                is_safe = True
                is_in_danger = False

                # Verificar si la ficha está en peligro de ser capturada
                for di, dj in [(-2, -2), (-2, 2), (2, -2), (2, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni_opp, nj_opp = row + di, col + dj
                    if 0 <= ni_opp < BOARD_SIZE and 0 <= nj_opp < BOARD_SIZE:
                        if board[ni_opp][nj_opp] == opponent or board[ni_opp][nj_opp] == opponent*2:
                            ni2 = -1 if di == -2 else (1 if di == 2 else di)
                            nj2= -1 if dj == -2 else (1 if dj == 2 else dj)
                            ni2, nj2 = row - ni2, col - nj2
                            if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and board[ni2][nj2] == EMPTY:
                                # Verificar la dirección del movimiento según el jugador
                                if abs(di) < 2 and abs(dj) < 2 and ((player == 1 and ni2 < row) or (player == -1 and ni2 > row)):
                                    is_safe = False
                                    is_in_danger = True
                                    break
                                elif board[ni_opp][nj_opp] == opponent * 2:
                                    is_safe = False
                                    is_in_danger = True
                                    break

                # Asignar recompensas o penalizaciones según la seguridad
                if is_safe:
                    if fichas_player == 1:
                        reward += 50
                        print("bien a salvo 50+")
                    else:
                        reward += 10  # Recompensa por estar en zona segura
                        print("a salvo 10+")
                elif is_in_danger:
                    if fichas_player == 1:
                        reward -= 300  # Castigo por estar en zona de peligro
                        print("peligro grave -300")
                    elif board[row][col]==player*2:
                        reward -= 35  # Castigo por estar en zona de peligro
                        print("peligro de perder dama -35")
                    else: 
                        reward -= 30  # Castigo por estar en zona de peligro
                        print("peligro -30")
                    CAPTURA = copy.deepcopy(board)
                

    # Evaluar si el jugador pudo haber capturado en el tablero anterior
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if previous_board[row][col] == player or previous_board[row][col] == player * 2:
                could_have_captured = False

                # Verificar si la ficha pudo capturar una ficha del oponente
                for di, dj in [(-2, -2), (-2, 2), (2, -2), (2, 2), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    ni_opp, nj_opp = row + di, col + dj
                    if 0 <= ni_opp < BOARD_SIZE and 0 <= nj_opp < BOARD_SIZE:
                        if previous_board[ni_opp][nj_opp] == opponent or previous_board[ni_opp][nj_opp] == opponent * 2:
                            ni2 = -1 if di == -2 else (1 if di == 2 else di)
                            nj2 = -1 if dj == -2 else (1 if dj == 2 else dj)
                            ni2, nj2 =  ni_opp + ni2, nj_opp + nj2
                            if 0 <= ni2 < BOARD_SIZE and 0 <= nj2 < BOARD_SIZE and previous_board[ni2][nj2] == EMPTY:
                                # Verificar la dirección del movimiento según el jugador
                                if abs(di)<2 and abs(dj)<2 and ((player == 1 and ni2 > row) or (player == -1 and ni2 < row)):
                                    could_have_captured = True
                                    break
                                elif previous_board[row][col] == player*2 :
                                    # Si es una KING, puede capturar desde más lejos
                                    could_have_captured = True
                                    break

                # Asignar recompensas o penalizaciones según si pudo capturar
                if could_have_captured and not captured:
                    if fichas_oponente == 1:
                        reward -= 300  # castigo fuerte por desperdiciar la victoria
                        print("Desaprovechó ganar -300")
                    else:  
                        reward -= 100 # Castigo fuerte por no capturar cuando pudo
                        print("Desaprovechó captura -100")

    # Recompensa por avanzar
    if (player == WHITE_NORMAL and ni > i and previous_board[i][j] == player) or (player == BLACK_NORMAL and ni < i and previous_board[i][j] == player and fichas_player>0):
        reward += 5  
        print("pasos 5+")

    # Recompensa por llegar a la última fila (convertirse en dama)
    if (player == WHITE_NORMAL and ni == BOARD_SIZE - 1 and previous_board[i][j] == player) or (player == BLACK_NORMAL and ni == 0 and previous_board[i][j] == player):
        reward += 50  
        print("coronacion 50+")

    return reward

def guardar_partidas(num_partidas, filename="cache.txt"):
    """Guarda el número de partidas jugadas en un archivo."""
    with open(filename, "w") as f:
        f.write(str(num_partidas))

def cargar_partidas(filename="cache.txt"):
    """Carga el número de partidas jugadas desde un archivo."""
    try:
        with open(filename, "r") as f:
            return int(f.read())
    except FileNotFoundError:
        return 0  # Si el archivo no existe, empezamos desde 0
# Inicialización de Pygame
def draw_board(screen, board,current_player,winner=None):
    """Dibuja el tablero y las piezas."""
    screen.fill((255, 255, 255))
    font = pygame.font.Font(None, 36)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = (139, 69, 19) if (i + j) % 2 == 0 else (245, 222, 179)
            pygame.draw.rect(screen, color, rect)
            piece = board[i][j]
            if piece != EMPTY:
                piece_color = WHITE if piece > 0 else BLACK
                pygame.draw.circle(screen, piece_color, rect.center, CELL_SIZE // 3)
                if abs(piece) == 2:  # Dama
                    pygame.draw.circle(screen, GOLD, rect.center, CELL_SIZE // 6)
    if winner is not None:
        msg = "Empate" if winner == 0 else "Ganador: BLANCO" if winner == 1 else "Ganador: NEGRO"
        winner_label = font.render(msg, True, BLACK)
        screen.blit(winner_label, (10, 10))
    else:
        turn_label = font.render(f"Turno: {'BLANCO' if current_player == 1 else 'NEGRO'}", True, BLACK)
        screen.blit(turn_label, (10, 10))

# Función para entrenar el agente
def entrenar_agente():
    empate = 0
    win = 0
    default = 0
    moveopponent = None
    num_partidas = cargar_partidas()  # Cargar el número de partidas jugadas
    for episode in range(num_episodes):
        board.reset()
        game_over = False
        current_player = WHITE_NORMAL
        while not game_over:
            if board.skip_turn(current_player):
                current_player = WHITE_NORMAL if current_player == BLACK_NORMAL else BLACK_NORMAL
                moveopponent = None
                continue

            if current_player == WHITE_NORMAL:
                moveopponent = ai_move(board)
                if moveopponent:
                    board.make_move(moveopponent)
                game_over, winner = board.is_game_over()
                board.movidas_realizadas += 1
                current_player = BLACK_NORMAL

            elif current_player == BLACK_NORMAL:
                state = [row[:] for row in board.board]
                possible_actions = board.get_possible_actions(current_player)
                if possible_actions:
                    action = agent.choose_action(state, possible_actions)
                    previus_board = board.board
                    board.make_move(action)
                    next_state = [row[:] for row in board.board]
                    game_over, winner = board.is_game_over()

                    # Calcular recompensa intermedia
                    movidas = copy.deepcopy(board.movidas_realizadas) 
                    reward = get_reward(board.board, action, BLACK_NORMAL,previus_board,movidas,moveopponent)

                    # Recompensa adicional si el juego termina
                    if game_over:
                        if winner == BLACK_NORMAL:
                            reward += 1000 
                            reward += 64-movidas
                        elif winner == 0:
                            reward -= 100  
                        else:
                            reward -= 1000
                    print("puntaje final: ", reward)
                    # Actualizar el Q-table
                    agent.update_q_value(state, action, reward, next_state)

                board.movidas_realizadas += 1
                current_player = WHITE_NORMAL

        num_partidas += 1  # Incrementar el número de partidas jugadas
        guardar_partidas(num_partidas)  # Guardar el número de partidas en el archivo de caché

        print(f"Episodio {episode + 1} completado. Ganador: {winner}")
        if winner == BLACK_NORMAL:
            print("¡El agente ganó este episodio!")
        if winner == BLACK_NORMAL:
            win+=1
        elif winner == WHITE_NORMAL:
            default += 1
        else:
            empate+=1
        
        # Reducir epsilon después de cada episodio
        agent.decay_epsilon()
    # Guardar el Q-table después del entrenamiento
    print("empate: ",empate,"win: ",win,"defauld: ", default)
    agent.save_q_table()

# Función para jugar contra el agente
def jugar_contra_agente():
    # Inicializar Pygame solo cuando se juega contra el agente
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
    pygame.display.set_caption("Damas 4x4")

    board.reset()
    current_player = WHITE_NORMAL  # Humano juega con las blancas
    selected_piece = None
    possible_moves = []
    move_count = 0  # Contador de movimientos para el empate
    moveopponent = None
    
    # Dibujar el tablero
    draw_board(screen, board.board,current_player)
    pygame.display.flip()
    game = True
    while game:
        if board.skip_turn(current_player):
                current_player = WHITE_NORMAL if current_player == BLACK_NORMAL else BLACK_NORMAL
                moveopponent = None
                continue
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Turno del humano (fichas blancas)
            if current_player == WHITE_NORMAL:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = pygame.mouse.get_pos()
                    col = x // CELL_SIZE
                    row = y // CELL_SIZE
                    if selected_piece is None:
                        # Seleccionar una ficha blanca
                        if board.board[row][col] == WHITE_NORMAL or board.board[row][col] == WHITE_KING:
                            selected_piece = (row, col)
                            possible_moves = [move for move in board.get_possible_actions(WHITE_NORMAL) if move[0] == (row, col)]
                    else:
                        # Mover la ficha seleccionada
                        moveopponent = (selected_piece, (row, col))
                        if moveopponent in possible_moves:
                            board.make_move(moveopponent)
                            selected_piece = None
                            possible_moves = []
                            
                            # Dibujar el tablero
                            current_player = BLACK_NORMAL  # Cambia al turno del agente Q-learning
                            draw_board(screen, board.board,current_player)
                            pygame.display.flip()
                            pygame.time.delay(1000)  # Retraso de 1 segundo para ver el movimiento
                            move_count += 1

        # Turno del agente Q-learning (fichas negras)
        if current_player == BLACK_NORMAL:
            reward = 0
            state = [row[:] for row in board.board]
            possible_actions = board.get_possible_actions(current_player)
            if possible_actions:
                action = agent.choose_action(state, possible_actions)
                if action:
                    previous_board = copy.deepcopy(board.board)
                    board.make_move(action)
                    for fila in previous_board:
                        print(fila)
                    
                    # Dibujar el tablero
                    current_player = WHITE_NORMAL  # Cambia al turno del humano
                    draw_board(screen, board.board,current_player)
                    pygame.display.flip()
                    move_count += 1

            movidas = copy.deepcopy(board.movidas_realizadas)      
            reward = get_reward(board.board, action, BLACK_NORMAL,previous_board,movidas,moveopponent)
            print("next: ",reward)
            agent.update_q_value(state, action, reward, previous_board)

        # Verificar si el juego ha terminado
        game_over, winner = board.is_game_over()
        if game_over :  # Empate después de 20 movimientos
            print(f"El juego ha terminado. Ganador: {winner}")
            draw_board(screen, board.board,current_player,winner)
            pygame.display.flip()
            pygame.time.delay(3000)  # Espera 2 segundos antes de reiniciar
            board.reset()
            move_count = 0
            current_player = WHITE_NORMAL
        draw_board(screen, board.board,current_player)
        pygame.display.flip()

def jugar_contra_agente_qlearning():
    empate = 0
    win = 0
    default = 0
     # Inicializar Pygame solo cuando se juega contra el agente qlearning 
    pygame.init()
    running = 0
    screen = pygame.display.set_mode((BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE))
    pygame.display.set_caption("Damas 4x4")

    board.reset()
    current_player = WHITE_NORMAL  # Humano juega con las blancas
    print("turno de: BLANCO")
    while running<1000:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = 1000
                pygame.quit()
                return
            
        if board.skip_turn(current_player):
            current_player = WHITE_NORMAL if current_player == BLACK_NORMAL else BLACK_NORMAL
            print("turno de: ",current_player)
            continue
            

        if current_player == WHITE_NORMAL:
            state=copy.deepcopy(board.board)
            best_move = oponente_qlearning.get_best_move(state, current_player)
            print(best_move)
            if best_move:
                board.make_move(best_move)
                 # Dibujar el tablero
                draw_board(screen, board.board,current_player)
                pygame.display.flip()
                #pygame.time.delay(2000)  # Retraso de 1 segundo para ver el movimiento
                current_player = BLACK_NORMAL  # Cambia al turno del humano
                print("turno de: NEGRO")
        # Turno del agente Q-learning (fichas negras)
        if current_player == BLACK_NORMAL:
            state=copy.deepcopy(board.board) 
            possible_actions = board.get_possible_actions(current_player)
            if possible_actions:
                action = agent.choose_action(state, possible_actions)
                if action:
                    board.make_move(action)
                    # Dibujar el tablero
                    draw_board(screen, board.board,current_player)
                    pygame.display.flip()
                    #pygame.time.delay(2000)  # Retraso de 1 segundo para ver el movimiento
                    current_player = WHITE_NORMAL  # Cambia al turno del humano
                    print("turno de: BLANCO")


        # Verificar si el juego ha terminado
        game_over, winner = board.is_game_over()
        if game_over :  # Empate después de 20 movimientos
            print(f"El juego ha terminado. Ganador: {winner}")
            #pygame.time.delay(2000)  # Espera 2 segundos antes de reiniciar
            board.reset()
            current_player = WHITE_NORMAL
            if winner == BLACK_NORMAL:
                win+=1
            elif winner == WHITE_NORMAL:
                default += 1
            else:
                empate+=1
            running +=1
    print("empate: ",empate,"win: ",win,"defauld: ", default)




# Definir botones
buttons = [
    ("Entrenar agente contra minimax (1000 partidas)", (400//2 - 100, 100, 420, 50)),
    ("Jugar contra el agente", (600//2 - 100, 170, 200, 50)),
    ("Agente vs Agente (no terminado)", (500//2 - 100, 240, 290, 50)),
    ("Salir", (750//2 - 100, 310, 60, 50))
]

def draw_menu():
    pygame.init()
    font = pygame.font.Font(None, 25)
    screen = pygame.display.set_mode((600, 400))
    screen.fill(WHITE)
    for text, rect in buttons:
        pygame.draw.rect(screen, GRAY, rect)
        label = font.render(text, True, BLACK)
        screen.blit(label, (rect[0] + 10, rect[1] + 10))
    pygame.display.flip()


# Menú principal
def main():
    global board, agent
    board = Board()
    agent = QLearningAgent()
    draw_menu()
    while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    for i, (_, rect) in enumerate(buttons):
                        if pygame.Rect(rect).collidepoint(x, y):
                            if i == 0:
                                print("Entrenando agente...")
                                entrenar_agente()
                            elif i == 1:
                                print("Jugando contra el agente...")
                                pygame.quit()
                                jugar_contra_agente()
                            elif i == 2:
                                print("Error, opción no terminada")
                            elif i == 3:
                                print("Saliendo...")
                                pygame.quit()
                                sys.exit()

if __name__ == "__main__":
    main()