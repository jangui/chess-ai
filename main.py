import chess
import numpy as np

BOARD = chess.Board()

WHITE = [
    ('k', 'e1'),
    ('q', 'd1'),
    ('b1', 'c1'),
    ('b2', 'f1'),
    ('n1', 'b1'),
    ('n2', 'g1'),
    ('r1', 'a1'),
    ('r2', 'h1'),
    ('p1', 'a2'),
    ('p2', 'b2'),
    ('p3', 'c2'),
    ('p4', 'd2'),
    ('p5', 'e2'),
    ('p6', 'f2'),
    ('p7', 'g2'),
    ('p8', 'h2'),
]

BLACK = [
    ('k', 'e8'),
    ('q', 'd8'),
    ('b1', 'c8'),
    ('b2', 'f8'),
    ('n1', 'b8'),
    ('n2', 'g8'),
    ('r1', 'a8'),
    ('r2', 'h8'),
    ('p1', 'a7'),
    ('p2', 'b7'),
    ('p3', 'c7'),
    ('p4', 'd7'),
    ('p5', 'e7'),
    ('p6', 'f7'),
    ('p7', 'g7'),
    ('p8', 'h7'),
]

def init_state():
    state = np.zeros((16,2,64))
    for i in range(len(WHITE)):
        white_piece_ind = chess.parse_square(WHITE[i][1])
        black_piece_ind = chess.parse_square(BLACK[i][1])
        state[i][0][white_piece_ind] = 1
        state[i][1][black_piece_ind] = 1
    return state

def main():
    state = init_state()
    print(state[0])

    """
    move = chess.Move(ind, ind)
    valid = move in board.legal_moves()
    board.is_game_over(claim_draw=True)
    """


if __name__ == "__main__":
    main()
