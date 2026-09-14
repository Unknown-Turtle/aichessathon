"""AI Chessathon agent.
Mwenya Sikazwe

A classical alpha-beta engine on top of python-chess. Everything is written from
ordinary chess principles; no third-party engine, network, or tuned tables are used.

Search
    Fail-soft negamax with alpha-beta, iterative deepening, a transposition table that
    persists for the whole game, null-move pruning, check extension, late-move
    reductions, and a quiescence search over captures filtered by static exchange
    evaluation (SEE).

Move ordering
    Hash move, then winning/equal captures by MVV-LVA, then killer moves, then quiet
    moves by history heuristic, then losing captures last.

Evaluation
    Material, hand-written piece-square tables tapered between midgame and endgame,
    passed pawns scaled by rank, connected passers, doubled/isolated pawns, bishop pair,
    rooks on open files, a pawn shield in front of the king, and light mobility.

Time
    A soft budget stops new iterations, a hard budget aborts mid-search and returns the
    last completed iteration. A forced mate returns immediately. The whole of get_move
    is wrapped so any exception falls through to a legal move rather than a crash.
"""

from __future__ import annotations

import math
import sys
import time

import chess

# Search depth plus quiescence never approaches this, but a RecursionError is a lost game.
sys.setrecursionlimit(10_000)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)
WHITE, BLACK = chess.WHITE, chess.BLACK

# Material in centipawns, indexed by piece type (index 0 unused).
VALUE = [0, 100, 320, 330, 500, 900, 20000]
SEE_VALUE = [0, 100, 320, 330, 500, 900, 20000]

MATE = 100_000
MATE_BOUND = MATE - 1_000  # scores beyond this are mate-in-N
INF = MATE + 1
DRAW_CONTEMPT = 15  # a draw is scored slightly against the root side

# Game-phase weights: 24 at the start, 0 when only pawns remain.
PHASE_WEIGHT = [0, 0, 1, 1, 2, 4, 0]
TOTAL_PHASE = 24

# Transposition table bound types.
EXACT, LOWER, UPPER = 0, 1, 2
TT_MAX_ENTRIES = 600_000  # roughly 300 MB of tuple keys and entries, well inside 2 GB

MAX_PLY = 64
NODES_PER_CLOCK_CHECK = 512

# Pruning margins in centipawns, indexed by remaining depth.
REVERSE_FUTILITY_MARGIN = [0, 120, 240, 360]
FUTILITY_MARGIN = [0, 150, 300]
ASPIRATION_WINDOW = 40
# Late move pruning: at low depth, quiet moves beyond this many are skipped outright.
LMP_LIMIT = [0, 5, 9, 15]


def _build_lmr() -> list[list[int]]:
    """LMR[depth][move_index]: how much shallower a late quiet move is searched first.

    Grows with the log of both depth and move index, so early moves at low depth are
    barely reduced and the tail of a long move list at high depth is reduced by several
    plies. A reduced move that beats alpha is re-searched at full depth.
    """
    table = [[0] * 64 for _ in range(MAX_PLY + 1)]
    for depth in range(3, MAX_PLY + 1):
        for index in range(3, 64):
            table[depth][index] = int(0.75 + math.log(depth) * math.log(index) / 2.25)
    return table


LMR = _build_lmr()

# Flag pressure: when the opponent is short of time and we are not, keeping material on
# the board keeps their moves hard. This is a bonus per 100cp of non-pawn material still on
# the board, from our side's point of view, switched on by get_move.
KEEP_MATERIAL_PER_100 = 4
_keep_material_bonus = 0  # set by get_move before each search

# ---------------------------------------------------------------------------
# Piece-square tables. Written visually, rank 8 on the top row, from White's
# point of view. Ordinary principles: knights to the centre, rooks to the
# seventh, king tucked away in the midgame and central in the endgame.
# ---------------------------------------------------------------------------

# fmt: off
PAWN_MG = [
     0,   0,   0,   0,   0,   0,   0,   0,
    60,  60,  60,  60,  60,  60,  60,  60,
    15,  20,  30,  40,  40,  30,  20,  15,
     5,  10,  15,  30,  30,  15,  10,   5,
     0,   0,  10,  25,  25,  10,   0,   0,
     5,   0,   0,   5,   5,   0,   0,   5,
     5,  10,  10, -20, -20,  10,  10,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
]
PAWN_EG = [
     0,   0,   0,   0,   0,   0,   0,   0,
    90,  90,  90,  90,  90,  90,  90,  90,
    50,  50,  50,  50,  50,  50,  50,  50,
    30,  30,  30,  30,  30,  30,  30,  30,
    15,  15,  15,  15,  15,  15,  15,  15,
     5,   5,   5,   5,   5,   5,   5,   5,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
]
KNIGHT_MG = [
   -50, -40, -30, -30, -30, -30, -40, -50,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -30,   5,  20,  25,  25,  20,   5, -30,
   -30,   0,  20,  25,  25,  20,   0, -30,
   -30,   5,  15,  20,  20,  15,   5, -30,
   -40, -20,   0,   5,   5,   0, -20, -40,
   -50, -40, -30, -30, -30, -30, -40, -50,
]
KNIGHT_EG = [
   -40, -30, -20, -20, -20, -20, -30, -40,
   -30, -15,   0,   0,   0,   0, -15, -30,
   -20,   0,  10,  15,  15,  10,   0, -20,
   -20,   0,  15,  20,  20,  15,   0, -20,
   -20,   0,  15,  20,  20,  15,   0, -20,
   -20,   0,  10,  15,  15,  10,   0, -20,
   -30, -15,   0,   0,   0,   0, -15, -30,
   -40, -30, -20, -20, -20, -20, -30, -40,
]
BISHOP_MG = [
   -20, -10, -10, -10, -10, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   5,   5,  10,  10,   5,   5, -10,
   -10,   0,  10,  10,  10,  10,   0, -10,
   -10,  10,  10,  10,  10,  10,  10, -10,
   -10,   5,   0,   0,   0,   0,   5, -10,
   -20, -10, -10, -10, -10, -10, -10, -20,
]
BISHOP_EG = [
   -15, -10, -10, -10, -10, -10, -10, -15,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   5,  10,  10,   5,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -15, -10, -10, -10, -10, -10, -10, -15,
]
ROOK_MG = [
     0,   0,   0,   0,   0,   0,   0,   0,
    10,  15,  15,  15,  15,  15,  15,  10,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
     0,   0,   0,   5,   5,   0,   0,   0,
]
ROOK_EG = [
     5,   5,   5,   5,   5,   5,   5,   5,
    10,  10,  10,  10,  10,  10,  10,  10,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,
]
QUEEN_MG = [
   -20, -10, -10,  -5,  -5, -10, -10, -20,
   -10,   0,   0,   0,   0,   0,   0, -10,
   -10,   0,   5,   5,   5,   5,   0, -10,
    -5,   0,   5,   5,   5,   5,   0,  -5,
     0,   0,   5,   5,   5,   5,   0,  -5,
   -10,   5,   5,   5,   5,   5,   0, -10,
   -10,   0,   5,   0,   0,   0,   0, -10,
   -20, -10, -10,  -5,  -5, -10, -10, -20,
]
QUEEN_EG = [
   -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
    -5,   0,   0,   0,   0,   0,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   5,  10,  10,   5,   0,  -5,
    -5,   0,   5,  10,  10,   5,   0,  -5,
    -5,   0,   5,   5,   5,   5,   0,  -5,
    -5,   0,   0,   0,   0,   0,   0,  -5,
   -10,  -5,  -5,  -5,  -5,  -5,  -5, -10,
]
KING_MG = [
   -60, -60, -60, -60, -60, -60, -60, -60,
   -50, -50, -50, -50, -50, -50, -50, -50,
   -40, -40, -40, -40, -40, -40, -40, -40,
   -30, -30, -30, -40, -40, -30, -30, -30,
   -20, -20, -20, -30, -30, -20, -20, -20,
   -10, -10, -10, -20, -20, -10, -10, -10,
    10,  10,   0, -10, -10,   0,  10,  10,
    20,  30,  10,   0,   0,  10,  30,  20,
]
KING_EG = [
   -50, -30, -20, -20, -20, -20, -30, -50,
   -30, -10,   0,   0,   0,   0, -10, -30,
   -20,   0,  15,  20,  20,  15,   0, -20,
   -20,   0,  20,  30,  30,  20,   0, -20,
   -20,   0,  20,  30,  30,  20,   0, -20,
   -20,   0,  15,  20,  20,  15,   0, -20,
   -30, -10,   0,   0,   0,   0, -10, -30,
   -50, -30, -20, -20, -20, -20, -30, -50,
]
# fmt: on

# Passed pawn bonus by relative rank (index 0 = own back rank, 7 = promotion rank).
PASSED_MG = [0, 5, 10, 20, 35, 60, 100, 0]
PASSED_EG = [0, 10, 25, 45, 75, 120, 190, 0]
CONNECTED_PASSER_MG, CONNECTED_PASSER_EG = 15, 30
DOUBLED_MG, DOUBLED_EG = -10, -20
ISOLATED_MG, ISOLATED_EG = -12, -15
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 30, 50
ROOK_OPEN_MG, ROOK_SEMI_MG = 20, 10
ROOK_OPEN_EG, ROOK_SEMI_EG = 10, 5
SHIELD_MG = 10
MOBILITY_MG = [0, 0, 4, 4, 2, 1, 0]
MOBILITY_EG = [0, 0, 3, 4, 3, 2, 0]
TEMPO = 10

# King safety: each piece attacking the squares around the enemy king adds attack units;
# the penalty grows with the square of the total so a single attacker is nearly harmless
# and a coordinated attack is not. Midgame only, halved when the attacker has no queen.
ATTACK_UNITS = [0, 0, 2, 2, 3, 5, 0]
KING_ATTACK_CAP = 400

# The side that is ahead wants open lines. Locked pawn pairs (a pawn blocked head-on by
# an enemy pawn) and opposite-coloured bishops make a material edge harder to convert,
# which is how round 105 of the ladder was drawn.
LOCKED_PAWN_PENALTY = 8
OCB_PENALTY = 30
OCB_PURE_ENDGAME_PENALTY = 60
AHEAD_THRESHOLD = 150

# ---------------------------------------------------------------------------
# Tables built once at import (inside the free init budget).
# ---------------------------------------------------------------------------


def _build_pst() -> tuple[list[list[list[int]]], list[list[list[int]]]]:
    """Return MG[color][piece][square] and EG[...] with material folded in.

    White values are positive and black values negative so the evaluation is a plain sum.
    The visual tables above have a8 at index 0, so a white piece on square s reads
    table[s ^ 56] (mirror the rank) and a black piece reads table[s] directly.
    """
    visual = {
        PAWN: (PAWN_MG, PAWN_EG),
        KNIGHT: (KNIGHT_MG, KNIGHT_EG),
        BISHOP: (BISHOP_MG, BISHOP_EG),
        ROOK: (ROOK_MG, ROOK_EG),
        QUEEN: (QUEEN_MG, QUEEN_EG),
        KING: (KING_MG, KING_EG),
    }
    mg: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
    eg: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
    for piece, (tmg, teg) in visual.items():
        for sq in range(64):
            mg[WHITE][piece][sq] = VALUE[piece] + tmg[sq ^ 56]
            eg[WHITE][piece][sq] = VALUE[piece] + teg[sq ^ 56]
            mg[BLACK][piece][sq] = -(VALUE[piece] + tmg[sq])
            eg[BLACK][piece][sq] = -(VALUE[piece] + teg[sq])
    return mg, eg


MG_TABLE, EG_TABLE = _build_pst()

FILE_MASKS = [chess.BB_FILES[f] for f in range(8)]
ADJACENT_FILES = [
    (chess.BB_FILES[f - 1] if f > 0 else 0) | (chess.BB_FILES[f + 1] if f < 7 else 0)
    for f in range(8)
]


def _build_passed_masks() -> list[list[int]]:
    """PASSED[color][sq]: squares an enemy pawn would need to occupy to stop this pawn."""
    masks: list[list[int]] = [[0] * 64, [0] * 64]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        span = FILE_MASKS[f] | ADJACENT_FILES[f]
        ahead_white = 0
        ahead_black = 0
        for rr in range(r + 1, 8):
            ahead_white |= chess.BB_RANKS[rr]
        for rr in range(r):
            ahead_black |= chess.BB_RANKS[rr]
        masks[WHITE][sq] = span & ahead_white
        masks[BLACK][sq] = span & ahead_black
    return masks


PASSED_MASK = _build_passed_masks()


def _build_connected_masks() -> list[list[int]]:
    """CONNECTED[color][sq]: adjacent-file squares beside or diagonally behind a pawn."""
    masks: list[list[int]] = [[0] * 64, [0] * 64]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        adjacent = ADJACENT_FILES[f]
        white = chess.BB_RANKS[r] | (chess.BB_RANKS[r - 1] if r > 0 else 0)
        black = chess.BB_RANKS[r] | (chess.BB_RANKS[r + 1] if r < 7 else 0)
        masks[WHITE][sq] = adjacent & white
        masks[BLACK][sq] = adjacent & black
    return masks


CONNECTED_MASK = _build_connected_masks()
PROMOTION_SOURCE = [chess.BB_RANK_2, chess.BB_RANK_7]  # indexed by colour: BLACK=0, WHITE=1


def _build_shield_masks() -> list[list[int]]:
    """SHIELD[color][king_sq]: the three squares directly in front of the king."""
    masks: list[list[int]] = [[0] * 64, [0] * 64]
    for sq in range(64):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        files = FILE_MASKS[f] | ADJACENT_FILES[f]
        if r < 7:
            masks[WHITE][sq] = files & chess.BB_RANKS[r + 1]
        if r > 0:
            masks[BLACK][sq] = files & chess.BB_RANKS[r - 1]
    return masks


SHIELD_MASK = _build_shield_masks()
KING_ZONE = [chess.BB_KING_ATTACKS[sq] | chess.BB_SQUARES[sq] for sq in range(64)]

BB_SQUARES = chess.BB_SQUARES
BB_LIGHT = chess.BB_LIGHT_SQUARES
popcount = chess.popcount

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


EVAL_CACHE_MAX = 200_000
_eval_cache: dict[object, int] = {}


def evaluate(board: chess.Board) -> int:
    """Static evaluation in centipawns from the side to move's point of view, cached."""
    key = board._transposition_key()
    cached = _eval_cache.get(key)
    if cached is not None:
        return cached
    if len(_eval_cache) >= EVAL_CACHE_MAX:
        _eval_cache.clear()
    score = _evaluate(board)
    _eval_cache[key] = score
    return score


def _evaluate(board: chess.Board) -> int:
    occ_w = board.occupied_co[WHITE]
    occ_b = board.occupied_co[BLACK]
    pawns, knights, bishops = board.pawns, board.knights, board.bishops
    rooks, queens, kings = board.rooks, board.queens, board.kings
    all_pawns = pawns

    # Bare kings, or a lone minor piece against a bare king, cannot win.
    if not pawns and not rooks and not queens and popcount(knights | bishops) <= 1:
        return 0

    mg = 0
    eg = 0
    phase = 0
    # Material balance from White's view, for the "side that is ahead" terms.
    material_balance = 0

    for color, occ in ((WHITE, occ_w), (BLACK, occ_b)):
        sign = 1 if color == WHITE else -1
        mg_t = MG_TABLE[color]
        eg_t = EG_TABLE[color]
        own_pawns = pawns & occ
        enemy_pawns = pawns & ~occ
        not_own = ~occ
        enemy_king = kings & ~occ
        enemy_zone = KING_ZONE[enemy_king.bit_length() - 1] if enemy_king else 0
        attack_units = 0
        material_balance += sign * (
            popcount(own_pawns) * 100
            + popcount(knights & occ) * 320
            + popcount(bishops & occ) * 330
            + popcount(rooks & occ) * 500
            + popcount(queens & occ) * 900
        )

        # Pawns: PST, passed, doubled, isolated.
        bb = own_pawns
        pmg = mg_t[PAWN]
        peg = eg_t[PAWN]
        passed_masks = PASSED_MASK[color]
        connected_masks = CONNECTED_MASK[color]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += pmg[sq]
            eg += peg[sq]
            f = sq & 7
            if not (enemy_pawns & passed_masks[sq]):
                rel_rank = (sq >> 3) if color == WHITE else 7 - (sq >> 3)
                mg += sign * PASSED_MG[rel_rank]
                eg += sign * PASSED_EG[rel_rank]
                # Connected passer: a friendly pawn beside or diagonally behind it.
                if own_pawns & connected_masks[sq]:
                    mg += sign * CONNECTED_PASSER_MG
                    eg += sign * CONNECTED_PASSER_EG
            if not (own_pawns & ADJACENT_FILES[f]):
                mg += sign * ISOLATED_MG
                eg += sign * ISOLATED_EG
        for f in range(8):
            n = popcount(own_pawns & FILE_MASKS[f])
            if n > 1:
                mg += sign * DOUBLED_MG * (n - 1)
                eg += sign * DOUBLED_EG * (n - 1)

        # Knights.
        bb = knights & occ
        nmg, neg = mg_t[KNIGHT], eg_t[KNIGHT]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += nmg[sq]
            eg += neg[sq]
            phase += 1
            attacks = board.attacks_mask(sq)
            mob = popcount(attacks & not_own)
            mg += sign * MOBILITY_MG[KNIGHT] * mob
            eg += sign * MOBILITY_EG[KNIGHT] * mob
            if attacks & enemy_zone:
                attack_units += ATTACK_UNITS[KNIGHT]

        # Bishops.
        bb = bishops & occ
        if popcount(bb) >= 2:
            mg += sign * BISHOP_PAIR_MG
            eg += sign * BISHOP_PAIR_EG
        bmg, beg = mg_t[BISHOP], eg_t[BISHOP]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += bmg[sq]
            eg += beg[sq]
            phase += 1
            attacks = board.attacks_mask(sq)
            mob = popcount(attacks & not_own)
            mg += sign * MOBILITY_MG[BISHOP] * mob
            eg += sign * MOBILITY_EG[BISHOP] * mob
            if attacks & enemy_zone:
                attack_units += ATTACK_UNITS[BISHOP]

        # Rooks: PST, open files, mobility.
        bb = rooks & occ
        rmg, reg = mg_t[ROOK], eg_t[ROOK]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += rmg[sq]
            eg += reg[sq]
            phase += 2
            fmask = FILE_MASKS[sq & 7]
            if not (own_pawns & fmask):
                if not (all_pawns & fmask):
                    mg += sign * ROOK_OPEN_MG
                    eg += sign * ROOK_OPEN_EG
                else:
                    mg += sign * ROOK_SEMI_MG
                    eg += sign * ROOK_SEMI_EG
            attacks = board.attacks_mask(sq)
            mob = popcount(attacks & not_own)
            mg += sign * MOBILITY_MG[ROOK] * mob
            eg += sign * MOBILITY_EG[ROOK] * mob
            if attacks & enemy_zone:
                attack_units += ATTACK_UNITS[ROOK]

        # Queens.
        bb = queens & occ
        qmg, qeg = mg_t[QUEEN], eg_t[QUEEN]
        while bb:
            sq = (bb & -bb).bit_length() - 1
            bb &= bb - 1
            mg += qmg[sq]
            eg += qeg[sq]
            phase += 4
            attacks = board.attacks_mask(sq)
            mob = popcount(attacks & not_own)
            mg += sign * MOBILITY_MG[QUEEN] * mob
            eg += sign * MOBILITY_EG[QUEEN] * mob
            if attacks & enemy_zone:
                attack_units += ATTACK_UNITS[QUEEN]

        # King safety of the enemy king: our attackers on its zone.
        if attack_units:
            if not (queens & occ):
                attack_units //= 2
            penalty = attack_units * attack_units
            if penalty > KING_ATTACK_CAP:
                penalty = KING_ATTACK_CAP
            mg += sign * penalty

        # King: PST plus a pawn shield in the midgame.
        kbb = kings & occ
        if kbb:
            sq = kbb.bit_length() - 1
            mg += mg_t[KING][sq]
            eg += eg_t[KING][sq]
            mg += sign * SHIELD_MG * popcount(own_pawns & SHIELD_MASK[color][sq])

    # The side that is ahead dislikes locked pawns and opposite-coloured bishops.
    if material_balance >= AHEAD_THRESHOLD or material_balance <= -AHEAD_THRESHOLD:
        ahead = 1 if material_balance > 0 else -1
        locked = popcount(((pawns & occ_w) << 8) & pawns & occ_b)
        penalty = LOCKED_PAWN_PENALTY * locked
        white_bishops = bishops & occ_w
        black_bishops = bishops & occ_b
        if (
            popcount(white_bishops) == 1
            and popcount(black_bishops) == 1
            and bool(white_bishops & BB_LIGHT) != bool(black_bishops & BB_LIGHT)
            and not knights
        ):
            penalty += OCB_PURE_ENDGAME_PENALTY if not (rooks | queens) else OCB_PENALTY
        mg -= ahead * penalty
        eg -= ahead * penalty

    if phase > TOTAL_PHASE:
        phase = TOTAL_PHASE
    # Truncate toward zero so a mirrored position scores exactly the negation.
    score = int((mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE)
    score += TEMPO if board.turn == WHITE else -TEMPO
    if _keep_material_bonus:
        # Flag pressure: material on the board is good for the side named by the sign.
        non_pawn = (
            popcount(knights) * 320
            + popcount(bishops) * 330
            + popcount(rooks) * 500
            + popcount(queens) * 900
        )
        score += _keep_material_bonus * non_pawn // 100
    return score if board.turn == WHITE else -score


# ---------------------------------------------------------------------------
# Static exchange evaluation
# ---------------------------------------------------------------------------


def see(board: chess.Board, move: chess.Move) -> int:
    """Material the side to move expects to gain by starting the exchange on move.to_square.

    Plays out the capture sequence with least-valuable-attacker-first on both sides,
    recomputing attackers as pieces leave the board so x-rays through the vacated squares
    are found. Positive means the exchange wins material, negative means it loses it.
    """
    to_sq = move.to_square
    from_sq = move.from_square
    attacker = board.piece_type_at(from_sq)
    if attacker is None:
        return 0

    if board.is_en_passant(move):
        gain0 = SEE_VALUE[PAWN]
    else:
        victim = board.piece_type_at(to_sq)
        gain0 = SEE_VALUE[victim] if victim else 0

    attacker_value = SEE_VALUE[attacker]
    if move.promotion:
        attacker_value = SEE_VALUE[move.promotion]
        gain0 += SEE_VALUE[move.promotion] - SEE_VALUE[PAWN]

    gain = [gain0]
    occupied = board.occupied & ~BB_SQUARES[from_sq]
    side = not board.turn
    pieces_by_type = (
        (PAWN, board.pawns),
        (KNIGHT, board.knights),
        (BISHOP, board.bishops),
        (ROOK, board.rooks),
        (QUEEN, board.queens),
        (KING, board.kings),
    )

    while True:
        attackers = board.attackers_mask(side, to_sq, occupied) & occupied
        if not attackers:
            break
        # Least valuable attacker for this side.
        chosen_sq = -1
        chosen_value = 0
        for piece_type, bb in pieces_by_type:
            sub = attackers & bb
            if sub:
                chosen_sq = (sub & -sub).bit_length() - 1
                chosen_value = SEE_VALUE[piece_type]
                break
        if chosen_sq < 0:
            break
        gain.append(attacker_value - gain[-1])
        # If even winning the piece back cannot make this side's capture profitable,
        # neither side will continue, so stop early.
        if max(-gain[-2], gain[-1]) < 0:
            break
        attacker_value = chosen_value
        occupied &= ~BB_SQUARES[chosen_sq]
        side = not side

    while len(gain) > 1:
        last = gain.pop()
        gain[-1] = -max(-gain[-1], last)
    return gain[0]


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TimeUp(Exception):
    """Raised inside the search when the hard deadline passes."""


class Searcher:
    """Holds per-game state: the transposition table, killers, history, and game history."""

    def __init__(self) -> None:
        self.tt: dict[object, tuple[int, int, int, chess.Move | None]] = {}
        self.killers: list[list[chess.Move | None]] = [[None, None] for _ in range(MAX_PLY + 2)]
        # history[color][piece_type][to_square]
        self.history: list[list[list[int]]] = [[[0] * 64 for _ in range(7)] for _ in range(2)]
        self.game_keys: list[object] = []  # positions that have occurred in the game
        self.root_color = WHITE
        # Per-search state. path_keys is the game history followed by the current search
        # line, so repetition detection sees both.
        self.nodes = 0
        self.deadline = 0.0
        self.abort_allowed = False
        self.path_keys: list[object] = []
        self.seldepth = 0

    # -- helpers -------------------------------------------------------------

    def draw_score(self, board: chess.Board) -> int:
        """A draw is slightly bad for the root side, so we do not drift into repetitions."""
        return -DRAW_CONTEMPT if board.turn == self.root_color else DRAW_CONTEMPT

    def age_history(self) -> None:
        for color in (WHITE, BLACK):
            for piece in range(7):
                self.history[color][piece] = [h // 2 for h in self.history[color][piece]]

    def check_time(self) -> None:
        if self.abort_allowed and time.monotonic() >= self.deadline:
            raise TimeUp

    def is_repetition(self, board: chess.Board, key: object) -> bool:
        """Has this position occurred on the current line or earlier in the game?

        Only positions since the last irreversible move can repeat, so just the last
        halfmove_clock entries are scanned. A single earlier occurrence is treated as a
        draw, which is the usual engine convention.
        """
        clock = board.halfmove_clock
        if clock < 4:
            return False
        path = self.path_keys
        start = len(path) - clock
        if start < 0:
            start = 0
        return any(path[i] == key for i in range(len(path) - 2, start - 1, -2))

    def store(
        self, key: object, depth: int, score: int, flag: int, move: chess.Move | None, ply: int
    ) -> None:
        if score > MATE_BOUND:
            score += ply
        elif score < -MATE_BOUND:
            score -= ply
        if len(self.tt) >= TT_MAX_ENTRIES:
            self.tt.clear()
        self.tt[key] = (depth, score, flag, move)

    def order_moves(
        self,
        board: chess.Board,
        moves: list[chess.Move],
        hash_move: chess.Move | None,
        ply: int,
    ) -> list[chess.Move]:
        """Hash move, good captures (MVV-LVA), killers, quiets by history, bad captures."""
        scored: list[tuple[int, chess.Move]] = []
        killers = self.killers[ply]
        hist = self.history[board.turn]
        piece_at = board.piece_type_at
        for move in moves:
            if move == hash_move:
                scored.append((10_000_000, move))
                continue
            if board.is_capture(move):
                if board.is_en_passant(move):
                    victim_value = VALUE[PAWN]
                else:
                    victim_value = VALUE[board.piece_type_at(move.to_square) or PAWN]
                attacker_value = VALUE[board.piece_type_at(move.from_square) or PAWN]
                if move.promotion:
                    scored.append((9_000_000 + VALUE[move.promotion], move))
                elif attacker_value <= victim_value or see(board, move) >= 0:
                    scored.append((8_000_000 + victim_value * 10 - attacker_value // 10, move))
                else:
                    scored.append((victim_value * 10 - attacker_value // 10, move))
            elif move.promotion:
                scored.append((9_000_000 + VALUE[move.promotion], move))
            elif move == killers[0]:
                scored.append((7_000_000, move))
            elif move == killers[1]:
                scored.append((6_999_999, move))
            else:
                piece = piece_at(move.from_square) or PAWN
                scored.append((1_000_000 + hist[piece][move.to_square], move))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [m for _, m in scored]

    # -- quiescence -----------------------------------------------------------

    def quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        if self.nodes % NODES_PER_CLOCK_CHECK == 0:
            self.check_time()
        if ply > self.seldepth:
            self.seldepth = ply

        in_check = board.is_check()
        if in_check and ply < MAX_PLY:
            # No stand-pat when in check: every evasion has to be searched.
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
            for move in self.order_moves(board, moves, None, min(ply, MAX_PLY)):
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha, ply + 1)
                board.pop()
                if score > best:
                    best = score
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            break
            return best

        stand_pat = evaluate(board)
        if stand_pat >= beta:
            return stand_pat
        if ply >= MAX_PLY:
            return stand_pat
        best = stand_pat
        if stand_pat > alpha:
            alpha = stand_pat

        captures: list[tuple[int, chess.Move]] = []
        for move in board.generate_legal_captures():
            if move.promotion and move.promotion != QUEEN:
                continue
            if board.is_en_passant(move):
                victim_value = VALUE[PAWN]
            else:
                victim_value = VALUE[board.piece_type_at(move.to_square) or PAWN]
            # Delta pruning: this capture cannot lift the score to alpha.
            if stand_pat + victim_value + 200 < alpha and not move.promotion:
                continue
            attacker_value = VALUE[board.piece_type_at(move.from_square) or PAWN]
            if attacker_value > victim_value and not move.promotion and see(board, move) < 0:
                continue
            captures.append((victim_value * 10 - attacker_value // 10, move))
        # Queen promotions without capture are worth searching too.
        promoters = board.pawns & board.occupied_co[board.turn] & PROMOTION_SOURCE[board.turn]
        if promoters:
            for move in board.generate_legal_moves(promoters):
                if move.promotion == QUEEN and not board.is_capture(move):
                    captures.append((VALUE[QUEEN] * 10, move))
        captures.sort(key=lambda t: t[0], reverse=True)

        for _, move in captures:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    # -- main search -------------------------------------------------------

    def negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int, allow_null: bool
    ) -> int:
        self.nodes += 1
        if self.nodes % NODES_PER_CLOCK_CHECK == 0:
            self.check_time()

        if ply > 0:
            # Draw detection: repetition or fifty-move rule.
            if board.halfmove_clock >= 100:
                return self.draw_score(board)
            key = board._transposition_key()
            if self.is_repetition(board, key):
                return self.draw_score(board)
            # Mate distance pruning: no point searching for a slower mate.
            alpha = max(alpha, -MATE + ply)
            beta = min(beta, MATE - ply - 1)
            if alpha >= beta:
                return alpha
        else:
            key = board._transposition_key()

        in_check = board.is_check()
        if in_check:
            depth += 1  # check extension

        if depth <= 0 or ply >= MAX_PLY:
            return self.quiescence(board, alpha, beta, ply)

        # Transposition table probe.
        hash_move: chess.Move | None = None
        entry = self.tt.get(key)
        if entry is not None:
            tt_depth, tt_score, tt_flag, hash_move = entry
            if tt_depth >= depth and ply > 0:
                if tt_score > MATE_BOUND:
                    tt_score -= ply
                elif tt_score < -MATE_BOUND:
                    tt_score += ply
                if tt_flag == EXACT:
                    return tt_score
                if tt_flag == LOWER and tt_score >= beta:
                    return tt_score
                if tt_flag == UPPER and tt_score <= alpha:
                    return tt_score

        static_eval: int | None = None
        pv_node = beta - alpha > 1
        # Reverse futility: the position is so far above beta that a quiet move will not
        # bring it back down within the remaining depth.
        if depth <= 3 and not in_check and not pv_node and ply > 0 and beta < MATE_BOUND:
            static_eval = evaluate(board)
            if static_eval - REVERSE_FUTILITY_MARGIN[depth] >= beta:
                return static_eval

        # Null-move pruning: if passing still beats beta, the position is too good to matter.
        if (
            allow_null
            and not in_check
            and depth >= 3
            and ply > 0
            and beta < MATE_BOUND
            and (board.occupied_co[board.turn] & ~board.pawns & ~board.kings)
        ):
            reduction = 3 if depth >= 6 else 2
            board.push(chess.Move.null())
            self.path_keys.append(key)
            try:
                score = -self.negamax(
                    board, depth - 1 - reduction, -beta, -beta + 1, ply + 1, False
                )
            finally:
                self.path_keys.pop()
                board.pop()
            if score >= beta and score < MATE_BOUND:
                return score

        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if in_check else self.draw_score(board)

        moves = self.order_moves(board, moves, hash_move, ply)

        # Futility: at the last plies, quiet moves cannot recover a static deficit this large.
        futile = False
        if depth <= 2 and not in_check and not pv_node and alpha > -MATE_BOUND:
            if static_eval is None:
                static_eval = evaluate(board)
            futile = static_eval + FUTILITY_MARGIN[depth] <= alpha

        best_score = -INF
        best_move: chess.Move | None = None
        original_alpha = alpha
        self.path_keys.append(key)
        try:
            for index, move in enumerate(moves):
                is_capture = board.is_capture(move)
                is_quiet = not is_capture and not move.promotion
                if is_quiet and best_move is not None:
                    if futile and not board.gives_check(move):
                        continue
                    # Late move pruning: deep in the move list at low depth, quiet moves
                    # that ordering ranked this low almost never matter.
                    if (
                        depth <= 3
                        and not in_check
                        and not pv_node
                        and index >= LMP_LIMIT[depth]
                        and alpha > -MATE_BOUND
                    ):
                        continue
                board.push(move)

                # Late-move reduction: quiet moves late in the list get a shallower look first.
                new_depth = depth - 1
                if (
                    index >= 3
                    and depth >= 3
                    and is_quiet
                    and not in_check
                    and not board.is_check()  # a checking move is not reduced
                ):
                    reduction = LMR[depth][index if index < 64 else 63]
                    if pv_node and reduction > 0:
                        reduction -= 1
                    score = -self.negamax(
                        board, new_depth - reduction, -alpha - 1, -alpha, ply + 1, True
                    )
                    if score > alpha:
                        score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, True)
                elif index == 0:
                    score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, True)
                else:
                    # Principal variation search: null window first, full window on success.
                    score = -self.negamax(board, new_depth, -alpha - 1, -alpha, ply + 1, True)
                    if alpha < score < beta:
                        score = -self.negamax(board, new_depth, -beta, -alpha, ply + 1, True)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move
                    if score > alpha:
                        alpha = score
                        if alpha >= beta:
                            if is_quiet:
                                killers = self.killers[ply]
                                if killers[0] != move:
                                    killers[1] = killers[0]
                                    killers[0] = move
                                piece = board.piece_type_at(move.from_square) or PAWN
                                hist = self.history[board.turn][piece]
                                hist[move.to_square] += depth * depth
                                if hist[move.to_square] > 1_000_000:
                                    self.age_history()
                            break
        finally:
            self.path_keys.pop()

        if best_score <= original_alpha:
            flag = UPPER
        elif best_score >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.store(key, depth, best_score, flag, best_move, ply)
        return best_score

    # -- root ----------------------------------------------------------------

    def search_root(
        self,
        board: chess.Board,
        soft_ms: float,
        hard_ms: float,
        max_depth: int = MAX_PLY - 1,
    ) -> tuple[chess.Move, int, int]:
        """Iterative deepening. Returns (move, score, depth) of the last completed iteration."""
        start = time.monotonic()
        self.deadline = start + hard_ms / 1000.0
        self.nodes = 0
        self.seldepth = 0
        self.root_color = board.turn
        for k in self.killers:
            k[0] = k[1] = None
        self.age_history()

        moves = list(board.legal_moves)
        best_move = moves[0]
        best_score = 0
        completed_depth = 0
        root_key = board._transposition_key()
        stack_len = len(board.move_stack)
        # The game history ends with the root position (appended by the caller), so the
        # search line starts right after it.
        base_path = list(self.game_keys)
        if not base_path or base_path[-1] != root_key:
            base_path.append(root_key)

        for depth in range(1, max_depth + 1):
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if depth > 1 and elapsed_ms > soft_ms:
                break
            # Deeper iterations cost several times the previous one; do not start one that
            # will predictably blow through the soft budget.
            if depth > 3 and elapsed_ms * 2.0 > soft_ms:
                break
            self.abort_allowed = depth > 1

            hash_move = best_move
            ordered = self.order_moves(board, moves, hash_move, 0)
            # Aspiration window around the previous score; fall back to a full window if the
            # result lands outside it.
            if depth >= 4 and abs(best_score) < MATE_BOUND:
                windows = [
                    (best_score - ASPIRATION_WINDOW, best_score + ASPIRATION_WINDOW),
                    (-INF, INF),
                ]
            else:
                windows = [(-INF, INF)]
            iter_best: chess.Move | None = None
            iter_score = -INF
            self.path_keys = base_path
            try:
                for alpha0, beta0 in windows:
                    alpha, beta = alpha0, beta0
                    iter_best = None
                    iter_score = -INF
                    for index, move in enumerate(ordered):
                        board.push(move)
                        if index == 0:
                            score = -self.negamax(board, depth - 1, -beta, -alpha, 1, True)
                        else:
                            score = -self.negamax(board, depth - 1, -alpha - 1, -alpha, 1, True)
                            if score > alpha:
                                score = -self.negamax(board, depth - 1, -beta, -alpha, 1, True)
                        board.pop()
                        if score > iter_score:
                            iter_score = score
                            iter_best = move
                            if score > alpha:
                                alpha = score
                                if alpha >= beta:
                                    break
                    if alpha0 < iter_score < beta0:
                        break  # inside the window, result is exact
                    # Otherwise re-search with the next (full) window.
            except TimeUp:
                # Unwind the board to the root; keep the last completed iteration's move.
                while len(board.move_stack) > stack_len:
                    board.pop()
                # The previous best move was searched first with a full window, so if a
                # later move beat it before the abort, that move is better at this depth.
                if iter_best is not None and iter_best != hash_move:
                    best_move, best_score = iter_best, iter_score
                break
            finally:
                self.path_keys = base_path

            if iter_best is not None:
                best_move, best_score, completed_depth = iter_best, iter_score, depth
                self.store(root_key, depth, best_score, EXACT, best_move, 0)

            # Mate short circuit: a forced mate is found, no need to look deeper.
            if abs(best_score) >= MATE_BOUND:
                break
            if len(moves) == 1:
                break

        return best_move, best_score, completed_depth


# ---------------------------------------------------------------------------
# Time management
# ---------------------------------------------------------------------------

INCREMENT_MS = 500
SAFETY_MS = 60  # runner overhead and clock jitter
FAST_OPENING_MOVES = 4  # our first few moves of a game run on a reduced budget
START_CLOCK_MS = 120_000
OPPONENT_OVERHEAD_MS = 10  # runner overhead included in the gap between our moves


def time_budget(time_left_ms: int, board: chess.Board, moves_played: int) -> tuple[float, float]:
    """Return (soft_ms, hard_ms) for this move.

    Spend roughly a 1/18 share of the clock plus most of the increment each move, less in
    the first few moves of the game where the curated positions are level and deep thought
    buys little. Under pressure play on the increment alone. The hard limit is a multiple
    of the soft limit, capped so a single move never eats a large fraction of what is left.
    """
    remaining = max(0, time_left_ms - SAFETY_MS)
    if remaining < 1_500:
        return 80.0, 150.0
    if remaining < 6_000:
        soft = INCREMENT_MS * 0.7
        hard = INCREMENT_MS * 0.95
        return soft, hard
    # Fewer pieces means a cheaper search, and the game is likely closer to its end.
    pieces = popcount(board.occupied)
    divisor = 18 if pieces > 20 else 15
    soft = remaining / divisor + INCREMENT_MS * 0.8
    if moves_played < FAST_OPENING_MOVES:
        soft *= 0.5
    hard = min(soft * 2.5, remaining * 0.25)
    return soft, hard


class OpponentClock:
    """Estimate the opponent's remaining clock from the wall time between our moves.

    Our process is suspended while the opponent thinks, but time.monotonic() keeps running,
    so the gap between the end of one get_move and the start of the next is their think
    time plus runner overhead. Both sides start at the same clock with the same increment.
    """

    def __init__(self) -> None:
        self.remaining_ms = float(START_CLOCK_MS)
        self.last_move_end: float | None = None
        self.last_think_ms = 0.0
        self.think_history: list[float] = []

    def on_move_start(self, now: float) -> None:
        if self.last_move_end is not None:
            gap_ms = (now - self.last_move_end) * 1000.0
            self.last_think_ms = max(0.0, gap_ms - OPPONENT_OVERHEAD_MS)
            self.remaining_ms -= self.last_think_ms
            self.remaining_ms += INCREMENT_MS  # their increment lands after their move
            self.remaining_ms = max(0.0, self.remaining_ms)
            self.think_history.append(self.last_think_ms)

    def spiked(self) -> bool:
        """Did the opponent just think far longer than they usually do?

        A sudden long think means their search found the position critical. That is a
        hint the position is sharper than our own evaluation suggests, so we give this
        move more time. It only ever changes how long we think, never what we play.
        """
        history = self.think_history
        if len(history) < 4 or self.last_think_ms < 1_000:
            return False
        recent = sorted(history[-9:-1])
        median = recent[len(recent) // 2]
        return self.last_think_ms >= 3.0 * max(median, 50.0)

    def on_move_end(self, now: float) -> None:
        self.last_move_end = now


_opponent = OpponentClock()
_moves_played = 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_searcher = Searcher()


def _choose(fen: str, time_left_ms: int) -> str:
    global _keep_material_bonus, _moves_played
    board = chess.Board(fen)
    started = time.monotonic()
    _opponent.on_move_start(started)

    key = board._transposition_key()
    _searcher.game_keys.append(key)

    # Flag pressure: opponent short of time while we are comfortable.
    opp_ms = _opponent.remaining_ms
    previous_bonus = _keep_material_bonus
    if _moves_played > 0 and opp_ms < 15_000 and time_left_ms > opp_ms * 1.5:
        _keep_material_bonus = KEEP_MATERIAL_PER_100
        if board.turn == BLACK:
            _keep_material_bonus = -KEEP_MATERIAL_PER_100
    else:
        _keep_material_bonus = 0
    if _keep_material_bonus != previous_bonus:
        _eval_cache.clear()  # cached scores were computed under the old bonus

    legal = list(board.legal_moves)
    if len(legal) == 1:
        move = legal[0]
        print(
            f"move {board.fullmove_number}: {move.uci()} forced, {time_left_ms}ms left",
            flush=True,
        )
    else:
        soft_ms, hard_ms = time_budget(time_left_ms, board, _moves_played)
        if time_left_ms > 20_000 and _opponent.spiked():
            soft_ms *= 1.3
            hard_ms = min(hard_ms * 1.3, time_left_ms * 0.25)
        move, score, depth = _searcher.search_root(board, soft_ms, hard_ms)
        elapsed_ms = (time.monotonic() - started) * 1000.0
        nps = int(_searcher.nodes / max(elapsed_ms / 1000.0, 1e-3))
        print(
            f"m{board.fullmove_number} {move.uci()} d{depth}/{_searcher.seldepth} s{score} "
            f"n{_searcher.nodes} nps{nps} t{elapsed_ms:.0f}/{soft_ms:.0f}/{hard_ms:.0f} "
            f"left{time_left_ms} opp{opp_ms:.0f}({_opponent.last_think_ms:.0f}"
            f"{'!' if _opponent.spiked() else ''}) "
            f"p{_keep_material_bonus} tt{len(_searcher.tt)}",
            flush=True,
        )

    # The one place every search bug is caught: never return an illegal move.
    if move not in board.legal_moves:
        print(f"illegal {move.uci()} from search, substituting", flush=True)
        move = legal[0]

    # Remember the position after our move too, so repetitions are tracked on both sides.
    board.push(move)
    _searcher.game_keys.append(board._transposition_key())
    _moves_played += 1
    _opponent.on_move_end(time.monotonic())
    return move.uci()


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI for the side to move in fen."""
    try:
        return _choose(fen, time_left_ms)
    except Exception as exc:
        # Any failure must still produce a legal move; a crash loses the game outright.
        print(f"fallback: {exc!r}", file=sys.stderr, flush=True)
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        # Prefer a capture or check over a random move when we are in fallback.
        for move in moves:
            if board.is_capture(move):
                return move.uci()
        return moves[0].uci()


# Warm up: run a short search at import so any lazy initialisation in python-chess and
# our tables lands in the init budget rather than on the clock.
_warm = chess.Board()
Searcher().search_root(_warm, 300.0, 400.0)
