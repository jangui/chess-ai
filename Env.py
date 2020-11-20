import numpy as np
import chess

class Env:
    def __init__(self):
        self.board = chess.Board()
        self.output_dimensions = 64*64
        self.reset()

    def step(self, action):
        reward = 0

        # move piece
        # if moving pawn to backrank, promote
        piece = str(self.board.piece_at(action[0])).lower()
        if (action[1] == 0 or action[1] == 7) and piece == 'p':
            move = chess.Move(action[0], action[1], promotion=chess.QUEEN)
        else:
            move = chess.Move(action[0], action[1])
        turn = self.board.turn

        # check if move legal
        valid = move in self.board.legal_moves

        # if invalid, end game and negative reward
        if not valid:
            print("ILLEGAL MOVE", chess.SQUARE_NAMES[action[0]], chess.SQUARE_NAMES[action[1]])
            print(action[0],action[1])
            if turn:
                reward = -9999
            else:
                reward = 0
            self.done = True
        else:
            # make move
            self.board.push(move)

        # game over
        if self.board.is_game_over(claim_draw=True):
            if turn: # white won
                reward = 9999
            else: # black won
                reward = -9999
            self.done = True

        # too many moves
        if self.board.fullmove_number > 45:
            print("TOO MANY MOVES")
            reward = 0
            self.done = True

        self.update_state(action)
        return self.state, reward, self.done

    def update_state(self, move):
        piece = str(self.board.piece_at(move[0])).lower()

        # if not moving a piece, don't update state
        if piece == 'none':
            return self.state

        color = self.board.piece_at(move[0]).color
        if color:
            color = 1
        else:
            # color is black
            color = -1

        ind1, ind2 = self.num_to_inds(move[0])
        ind3, ind4 = self.num_to_inds(move[1])

        if piece == 'p':
            ind0 = 0
        elif piece == 'n':
            ind0 = 1
        elif piece == 'b':
            ind0 = 2
        elif piece == 'r':
            ind0 = 3
        elif piece == 'q':
            ind0 = 4
        elif piece == 'k':
            ind0 = 5

        # update state to reflect move
        self.state[ind0][ind1][ind2] = 0
        self.state[ind0][ind3][ind4] = color
        return self.state


    def num_to_inds(self, num):
        ind1 = num // 8
        ind2 = num % 8
        return ind1, ind2

    def reset(self):
        self.done = False
        self.board.reset()
        self.state = np.array((
            # pawns
            [[0,0,0,0,0,0,0,0],
            [1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [0,0,0,0,0,0,0,0]],

            # knights
            [[0,1,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,-1,0,0,0,0,-1,0]],

            # bishops
            [[0,0,1,0,0,1,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,-1,0,0,-1,0,0]],

            # rooks
            [[1,0,0,0,0,0,0,1],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [-1,0,0,0,0,0,0,-1]],

            # queens
            [[0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,-1,0,0,0]],

            # kings
            [[0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0],
            [0,0,0,-1,0,0,0,0]],
            ))
        return self.state


