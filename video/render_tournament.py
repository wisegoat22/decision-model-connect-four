"""Silent Connect Four film from completed, audited tournament JSON.

Layout mode produces watermarked PNGs only. A final MP4 requires actual,
complete results and consistency checks against every recorded match.
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import hashlib
import json
import math
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W, H, FPS, DURATION = 1080, 1350, 30, 64
BG, INK, MUTED, LINE = '#F5F3ED', '#152B38', '#5B6A72', '#D8DDD9'
WHITE, NAVY, PALE = '#FFFFFF', '#112C3B', '#E8ECE7'
COLORS = ['#087F8C', '#B9512A', '#7563A8', '#28644B']
FONT = '/System/Library/Fonts/Avenir Next.ttc'
TIMES = [2, 9, 15, 23, 34, 37, 44, 55, 61]


@functools.lru_cache(maxsize=200)
def font(size, weight='regular'):
    return ImageFont.truetype(FONT, round(size), index={
        'bold': 0, 'demi': 2, 'medium': 5, 'regular': 7}[weight])


def text(d, xy, value, size=30, color=INK, weight='regular', anchor=None):
    d.text(xy, str(value), fill=color, font=font(size, weight), anchor=anchor)


def fit(d, value, maxwidth, size=38, minimum=20, weight='demi'):
    while size > minimum and d.textlength(str(value), font=font(size, weight)) > maxwidth:
        size -= 1
    if d.textlength(str(value), font=font(size, weight)) > maxwidth:
        raise ValueError(f'Text cannot fit without truncation: {value!r}')
    return size


def wrap(d, value, width, size=30, weight='regular'):
    lines, line = [], ''
    for word in str(value).split():
        candidate = f'{line} {word}'.strip()
        if line and d.textlength(candidate, font=font(size, weight)) > width:
            lines.append(line)
            line = word
        else:
            line = candidate
    return lines + ([line] if line else [])


def paragraph(d, xy, value, width, size=30, color=INK, weight='regular', leading=1.3):
    x, y = xy
    lines = wrap(d, value, width, size, weight)
    for i, line in enumerate(lines):
        text(d, (x, y + i * size * leading), line, size, color, weight)
    return y + len(lines) * size * leading


def title(d, lines, y=148, size=66):
    for i, line in enumerate(lines):
        text(d, (72, y + i * size * 1.12), line, size, INK, 'bold')


def base(t, chapter, layout=False):
    im = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 13), fill=LINE)
    d.rectangle((0, 0, round(W * t / DURATION), 13), fill=COLORS[0])
    text(d, (72, 55), 'DECISION MODEL LAB', 24, COLORS[0], 'bold')
    text(d, (1008, 55), chapter.upper(), 21, MUTED, 'demi', 'ra')
    d.line((72, 1266, 1008, 1266), fill=LINE, width=2)
    text(d, (72, 1283), 'TARUN DAGAR', 22, INK, 'demi')
    text(d, (1008, 1283), 'CONNECT FOUR  /  SILENT FILM', 20, MUTED, 'medium', 'ra')
    if layout:
        d.rounded_rectangle((72, 1168, 1008, 1248), radius=14, fill='#9B3225')
        text(d, (540, 1191), 'LAYOUT ONLY · RESULTS PENDING', 29, WHITE, 'bold', 'ma')
    return im, d


def model_id(value, match):
    if value in (1, '1'):
        return match['first_player']
    if value in (2, '2'):
        return match['second_player']
    return value


def player_number(value, match):
    return 1 if model_id(value, match) == match['first_player'] else 2


def winning_cells(board):
    for r in range(6):
        for c in range(7):
            token = board[r * 7 + c]
            if not token:
                continue
            for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                cells = [(r + i * dr) * 7 + c + i * dc for i in range(4)]
                if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                    if all(board[k] == token for k in cells):
                        return cells
    return []


def winning_colors(board):
    winners = set()
    for r in range(6):
        for c in range(7):
            token = board[r * 7 + c]
            if token:
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                        if all(board[(r + i * dr) * 7 + c + i * dc] == token for i in range(4)):
                            winners.add(token)
    return winners


def forfeit_request(board, to_move):
    """Independently reconstruct the frozen common request for forfeit evidence."""
    legal = [c + 1 for c in range(7) if board[c] == 0]
    symbols = '.XO'
    rows = [''.join(symbols[v] for v in board[r * 7:(r + 1) * 7]) for r in range(6)]
    state = ('Connect Four. Drop a piece into a column; gravity applies. Connect 4 horizontally, vertically or diagonally to win. '
             f'X moves first. You play {symbols[to_move]}. . is empty.\n'
             'Rows top to bottom; columns left to right.\nColumns: 1234567\n'
             + '\n'.join(rows) + '\nLegal columns: ' + ','.join(map(str, legal)))
    return {'state': state, 'questions': {'move': {'type': 'choice',
            'instructions': 'Choose the legal column that gives you the best chance to win.',
            'criteria': {f'c{c}': f'Drop in column {c}.' for c in legal}}}}


def validate_match(match):
    """Audit terminal outcomes independently of reported totals or the runner."""
    a, b = match['first_player'], match['second_player']
    if a == b or match.get('status', 'complete') != 'complete':
        raise ValueError('Match must be complete and have two distinct players.')
    board = list(match['initial_board'])
    if len(board) != 42 or any(type(v) is not int or v != 0 for v in board):
        raise ValueError('Initial board must be42 integer zeros before the included opening plies.')
    opening = match['opening_columns']
    if len(match['moves']) < len(opening):
        raise ValueError('Missing fixed opening moves.')
    for i, move in enumerate(match['moves'], 1):
        if winning_colors(board) or all(board):
            raise ValueError('Recorded moves continue after a terminal board.')
        token = player_number(move['player'], match)
        if model_id(move['player'], match) not in (a, b):
            raise ValueError('Recorded move has unknown player.')
        if move.get('ply') != i or type(move.get('ply')) is not int:
            raise ValueError('Recorded ply numbers are not consecutive.')
        if token != 1 + (i - 1) % 2:
            raise ValueError('Recorded move violates player alternation.')
        is_opening = i <= len(opening)
        if move.get('opening') is not is_opening or (is_opening and move['column'] != opening[i - 1]):
            raise ValueError('Recorded opening differs from the fixed opening.')
        column = move['column']
        if type(column) is not int or not 1 <= column <= 7:
            raise ValueError('Column must be an integer1..7.')
        col = column - 1
        row = next((r for r in range(5, -1, -1) if not board[r * 7 + col]), None)
        if row is None:
            raise ValueError('Recorded move drops into a full column.')
        board[row * 7 + col] = token
        recorded = move['board']
        if len(recorded) != 42 or any(type(v) is not int or v not in (0, 1, 2) for v in recorded):
            raise ValueError('Recorded board must contain42 integers0/1/2.')
        if board != recorded:
            raise ValueError(f'Recorded board differs from legal move in {match["id"]}, move{i}.')
    winners = winning_colors(board)
    if len(winners) > 1:
        raise ValueError('Both colors cannot win a completed legal game.')
    board_winner = next(iter(winners), None)
    full = all(board)
    winner = model_id(match.get('winner'), match)
    draw = match.get('draw')
    if type(draw) is not bool:
        raise ValueError('Draw flag must be a boolean.')
    reason = match.get('termination')
    if reason == 'invalid_choice':
        if board_winner is not None or full:
            raise ValueError('An invalid-choice forfeit cannot replace a terminal board result.')
        to_move = 1 + len(match['moves']) % 2
        decision = match.get('invalid_decision')
        if not isinstance(decision, dict) or 'response' not in decision:
            raise ValueError('Invalid-choice forfeit requires the recorded invalid decision evidence.')
        request = forfeit_request(board, to_move)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':'),
                                          ensure_ascii=True, allow_nan=False).encode()).hexdigest()
        expected_model = a if to_move == 1 else b
        if (decision.get('model_id') != expected_model or decision.get('request') != request
                or decision.get('request_sha256') != digest):
            raise ValueError('Forfeit decision evidence does not match the acting model and current board.')
        response = decision['response']
        if not isinstance(response, dict) or response.get('error') is not None:
            raise ValueError('Infrastructure errors cannot count as model forfeits.')
        if isinstance(response.get('metadata'), dict) and response['metadata'].get('truncated') is True:
            raise ValueError('Truncated requests cannot count as model forfeits.')
        choice = response.get('choice')
        if isinstance(choice, str) and len(choice) == 1 and choice in '1234567':
            choice = 'c' + choice
        if isinstance(choice, str) and choice in request['questions']['move']['criteria']:
            raise ValueError('A legal model response cannot be scored as an invalid-choice forfeit.')
        expected_color = 3 - to_move
        expected_winner = a if expected_color == 1 else b
        if draw or winner != expected_winner:
            raise ValueError('Forfeit winner must be the opponent of the model with the invalid choice.')
    elif reason == 'connect_four':
        if board_winner is None:
            raise ValueError('A connect_four result requires a winning line on the final board.')
        expected_color = board_winner
        expected_winner = a if expected_color == 1 else b
        if draw or winner != expected_winner:
            raise ValueError('Reported winner does not match the final board’s winning color.')
    elif reason == 'board_full':
        expected_color = None
        if not full or board_winner is not None or not draw or winner is not None:
            raise ValueError('A draw requires a full board with no winning color and no reported winner.')
    else:
        raise ValueError('Unknown or missing completed-match termination reason.')
    if 'winner_color' in match and match['winner_color'] != expected_color:
        raise ValueError('Reported winner_color does not match the verified outcome.')
    return board, winner


def validate(data):
    if not data.get('complete'):
        raise ValueError('Final output requires complete=true; partial results cannot be ranked.')
    if data.get('game') != 'connect_four':
        raise ValueError('This renderer supports Connect Four only.')
    models = data['models']
    ids = {m['id'] for m in models}
    if len(models) != 4 or len(ids) != 4:
        raise ValueError('Exactly four distinct verified models are required.')
    matches = data['matches']
    if not matches:
        raise ValueError('No recorded matches.')
    replay_ids = data.get('video_match_ids', [])
    if len(replay_ids) != 2 or len(set(replay_ids)) != 2:
        raise ValueError('Two prespecified replay IDs are required to show all four models playing.')
    selected = [next((m for m in matches if m['id'] == key), None) for key in replay_ids]
    if any(m is None for m in selected):
        raise ValueError('A selected replay is missing from the tournament.')
    if {mid for m in selected for mid in (m['first_player'], m['second_player'])} != ids:
        raise ValueError('The two replays must collectively show all four participants.')
    for chosen in selected:
        pair = {chosen['first_player'], chosen['second_player']}
        first = next(m for m in matches if {m['first_player'], m['second_player']} == pair
                     and not m['opening_columns'])
        if chosen['id'] != first['id'] or chosen['opening_columns']:
            raise ValueError('Each replay must be its pair’s first scheduled empty-opening match.')
    tally = {mid: Counter(wins=0, draws=0, losses=0, games=0) for mid in ids}
    balanced = Counter()
    pairs = Counter()
    match_ids = set()
    for match in matches:
        if match['id'] in match_ids:
            raise ValueError('Duplicate match ID.')
        match_ids.add(match['id'])
        a, b = match['first_player'], match['second_player']
        if a not in ids or b not in ids or a == b:
            raise ValueError('Match has invalid participants.')
        board, winner = validate_match(match)
        for mid in (a, b):
            tally[mid]['games'] += 1
            tally[mid]['draws' if match.get('draw') else 'wins' if mid == winner else 'losses'] += 1
        balanced[(a, b, str(match['opening_id']))] += 1
        pairs[frozenset((a, b))] += 1
    standings = data['standings']
    if len(standings) != 4 or {r['model_id'] for r in standings} != ids:
        raise ValueError('Standings must include each of the four models.')
    ordered = sorted(standings, key=lambda r: (-r['wins'], r['losses'], -r['draws']))
    prior, rank = None, 0
    for pos, row in enumerate(ordered, 1):
        if any(row[k] != tally[row['model_id']][k] for k in ('wins', 'draws', 'losses', 'games')):
            raise ValueError('Leaderboard does not match recorded games.')
        key = (row['wins'], row['losses'], row['draws'])
        if key != prior:
            rank = pos
        if row['rank'] != rank:
            raise ValueError('Rank does not follow wins descending, losses ascending, shared exact ties.')
        prior = key
    data['_ordered'] = ordered
    data['_balanced'] = all(n == balanced[(b, a, opening)] for (a, b, opening), n in balanced.items())
    if (len(matches) != 60 or {r['games'] for r in ordered} != {30}
            or not data['_balanced'] or len(pairs) != 6 or set(pairs.values()) != {10}
            or set(balanced.values()) != {1}):
        raise ValueError('This film requires the agreed60-match tournament,30 games per model, swapped seats.')
    data['_replays'] = selected
    data['_forfeits'] = sum(m['termination'] == 'invalid_choice' for m in matches)
    return data


def context(data):
    models = {m['id']: m for m in data['models']}
    colors = {m['id']: COLORS[i] for i, m in enumerate(data['models'])}
    return models, colors


def intro(t, data, layout):
    im, d = base(t, 'The lineup', layout)
    title(d, ['Which decision', 'model wins?'])
    paragraph(d, (76, 335), 'Connect Four · head-to-head decisions', 928, 31, MUTED, 'medium')
    for i, model in enumerate(data['models']):
        x, y = 72 + (i % 2) * 480, 446 + (i // 2) * 260
        d.rounded_rectangle((x, y, x + 456, y + 226), radius=24, fill=WHITE, outline=LINE, width=2)
        d.ellipse((x + 26, y + 27, x + 64, y + 65), fill=COLORS[i])
        text(d, (x + 26, y + 86), model['name'], fit(d, model['name'], 405, 39), INK, 'demi')
        paragraph(d, (x + 26, y + 148), model.get('subtitle', 'External decision model'), 400, 24, MUTED)
    paragraph(d, (76, 1050), 'Planned: 60 matches · 30 games per model' if layout else '60 matches. 30 games each. Ranked by wins.', 924, 33, INK, 'demi')
    return im


def protocol(t, data, layout):
    im, d = base(t, 'The tournament', layout)
    title(d, ['Same rules.', 'Both sides of the board.'], size=60)
    count = data.get('planned_matches', len(data.get('matches', [])))
    d.rounded_rectangle((72, 395, 1008, 870), radius=28, fill=NAVY)
    text(d, (110, 430), str(count) if count else '—', 113, WHITE, 'bold')
    text(d, (302, 488), 'MATCHES' if not layout else 'PLANNED MATCHES', 34, '#A8DCD8', 'demi')
    paragraph(d, (112, 603), 'Every pair meets. Starting seats swap.', 820, 35, WHITE, 'demi')
    paragraph(d, (112, 689), 'Fixed openings · same legal moves', 820, 30, '#D5E0E3')
    paragraph(d, (112, 752), 'Four in a row wins. A full board can draw.', 820, 30, '#D5E0E3')
    paragraph(d, (76, 945), 'Rank by wins. Fewer losses break a tie; exact ties share a rank.', 926, 34, INK, 'demi')
    paragraph(d, (76, 1080), 'Two replays show all four models. Match selection was fixed before results.', 926, 29, MUTED)
    return im


def board(d, values, colors, last_cell=None, moving=None, highlight=()):
    x, y, cell, pad = 132, 483, 112, 16
    d.rounded_rectangle((x, y, x + 7 * cell + 2 * pad, y + 6 * cell + 2 * pad), radius=28, fill=NAVY)
    for c in range(7):
        text(d, (x + pad + (c + .5) * cell, y - 31), str(c + 1), 23, MUTED, 'demi', 'mm')
    for k, token in enumerate(values):
        r, c = divmod(k, 7)
        cx, cy = x + pad + (c + .5) * cell, y + pad + (r + .5) * cell
        color = BG if token == 0 else colors[token - 1]
        d.ellipse((cx - 45, cy - 45, cx + 45, cy + 45), fill=color)
        if token:
            text(d, (cx, cy), str(token), 27, WHITE, 'bold', 'mm')
        if k == last_cell:
            d.ellipse((cx - 49, cy - 49, cx + 49, cy + 49), outline=WHITE, width=3)
        if k in highlight:
            d.ellipse((cx - 49, cy - 49, cx + 49, cy + 49), outline='#F2C765', width=6)
    if moving:
        index, token, fraction = moving
        r, c = divmod(index, 7)
        cx = x + pad + (c + .5) * cell
        final_y = y + pad + (r + .5) * cell
        cy = y + 28 + (final_y - y - 28) * fraction * fraction
        d.ellipse((cx - 45, cy - 45, cx + 45, cy + 45), fill=colors[token - 1])
        text(d, (cx, cy), str(token), 27, WHITE, 'bold', 'mm')


def compact_board(d, values, xy, colors, last_cell=None, moving=None, highlight=()):
    x, y = xy
    cell, pad = 58, 13
    d.rounded_rectangle((x, y, x + 7 * cell + 2 * pad, y + 6 * cell + 2 * pad), radius=22, fill=NAVY)
    for k, token in enumerate(values):
        r, c = divmod(k, 7)
        cx, cy = x + pad + (c + .5) * cell, y + pad + (r + .5) * cell
        d.ellipse((cx - 23, cy - 23, cx + 23, cy + 23), fill=BG if not token else colors[token - 1])
        if token:
            text(d, (cx, cy), str(token), 20, WHITE, 'bold', 'mm')
        if k == last_cell:
            d.ellipse((cx - 26, cy - 26, cx + 26, cy + 26), outline=WHITE, width=2)
        if k in highlight:
            d.ellipse((cx - 26, cy - 26, cx + 26, cy + 26), outline='#F2C765', width=4)
    if moving:
        k, token, fraction = moving
        r, c = divmod(k, 7)
        cx = x + pad + (c + .5) * cell
        target = y + pad + (r + .5) * cell
        cy = y + 20 + (target - y - 20) * fraction * fraction
        d.ellipse((cx - 23, cy - 23, cx + 23, cy + 23), fill=colors[token - 1])
        text(d, (cx, cy), str(token), 20, WHITE, 'bold', 'mm')


def replay_state(t, match):
    seq = match['moves']
    progress = max(0, (t - 14.75) / 20.5) * max(1, len(seq))
    n = min(len(seq), max(0, math.floor(progress)))
    fraction = min(1, (progress - n) / .45) if n < len(seq) else 1
    current = match['initial_board'] if n == 0 else seq[n - 1]['board']
    last_cell, moving = None, None
    if n:
        prev = match['initial_board'] if n == 1 else seq[n - 2]['board']
        last_cell = next(k for k, (p, q) in enumerate(zip(prev, current)) if p != q)
    if n < len(seq) and fraction > 0:
        nxt = seq[n]
        changed = next(k for k, (p, q) in enumerate(zip(current, nxt['board'])) if p != q)
        moving = (changed, player_number(nxt['player'], match), fraction)
    return n, current, last_cell, moving


def replay(t, data, layout):
    im, d = base(t, 'Real game replay' if not layout else 'Replay layout', layout)
    models, colors = context(data)
    title(d, ['Four models in play.', 'Every move recorded.'], size=57)
    text(d, (76, 303), 'Two recorded games · both start from an empty board', 28, MUTED, 'medium')
    selected = data.get('_replays', data['matches'][:2])
    for i, match in enumerate(selected):
        y = 368 + i * 407
        d.rounded_rectangle((72, y - 14, 1008, y + 388), radius=22, fill=WHITE, outline=LINE, width=2)
        a, b = match['first_player'], match['second_player']
        match_label = f'MATCH {i + 1} · EMPTY BOARD'
        text(d, (99, y + 9), match_label, fit(d, match_label, 419, 21, 16), MUTED, 'demi')
        for j, mid in enumerate((a, b)):
            label_y = y + 72 + j * 83
            d.ellipse((99, label_y + 5, 135, label_y + 41), fill=colors[mid])
            text(d, (117, label_y + 23), str(j + 1), 20, WHITE, 'bold', 'mm')
            text(d, (153, label_y), models[mid]['name'], fit(d, models[mid]['name'], 339, 36), INK, 'demi')
        seq = match['moves']
        n, current, last_cell, moving = replay_state(t, match)
        shown_n = n + 1 if moving else n
        done = n == len(seq) and not layout
        compact_board(d, current, (553, y), [colors[a], colors[b]], last_cell, moving,
                      winning_cells(current) if done else ())
        text(d, (100, y + 251), 'Moves pending' if layout else f'MOVE {shown_n} / {len(seq)}', 25, MUTED, 'demi')
        if done:
            winner = model_id(match.get('winner'), match)
            label = 'Draw' if match.get('draw') else f'{models[winner]["name"]} wins'
            text(d, (100, y + 301), label, fit(d, label, 416, 28), INK, 'demi')
            reason = {'invalid_choice': 'Forfeit: invalid choice', 'connect_four': 'Four in a row',
                      'board_full': 'Full board'}.get(match['termination'], match['termination'].replace('_', ' '))
            text(d, (100, y + 340), reason, fit(d, reason, 412, 22, 17, 'regular'), MUTED)
        elif shown_n and seq[shown_n - 1].get('opening'):
            text(d, (100, y + 309), 'Fixed opening', 27, MUTED, 'medium')
        elif not layout:
            text(d, (100, y + 309), 'Recorded choices', 27, MUTED, 'medium')
    if not layout:
        text(d, (540, 1209), 'Moves replayed for clarity · move counts shown · not a speed test', 25, MUTED, 'medium', 'ma')
    return im


def leaderboard(t, data, layout):
    im, d = base(t, 'Complete results', layout)
    title(d, ['Ranked by wins.', 'Every game counted.'], size=61)
    text(d, (76, 326), 'Completed tournament' if not layout else 'Results will appear after all matches finish', 29, MUTED, 'medium')
    models, colors = context(data)
    rows = data.get('_ordered', data.get('standings', []))
    rank_counts = Counter(r['rank'] for r in rows)
    for i, row in enumerate(rows):
        x, y, mid = 72, 423 + i * 166, row['model_id']
        d.rounded_rectangle((x, y, 1008, y + 150), radius=22, fill=WHITE, outline=LINE, width=2)
        d.ellipse((98, y + 28, 154, y + 84), fill=colors[mid])
        rank = '—' if layout else ('=' if rank_counts[row['rank']] > 1 else '') + str(row['rank'])
        text(d, (126, y + 56), rank, 28, WHITE, 'bold', 'mm')
        name = models[mid]['name']
        text(d, (177, y + 20), name, fit(d, name, 565, 37), INK, 'demi')
        text(d, (966, y + 15), '—' if layout else row['wins'], 53, colors[mid], 'bold', 'ra')
        text(d, (966, y + 81), 'WINS', 20, MUTED, 'demi', 'ra')
        record = 'No scores yet' if layout else f'{row["games"]} games · {row["draws"]} draws · {row["losses"]} losses'
        text(d, (177, y + 77), record, 24, MUTED, 'medium')
        d.rounded_rectangle((177, y + 124, 966, y + 132), radius=4, fill=PALE)
        if not layout and row['wins']:
            width = 789 * row['wins'] / max(1, row['games'])
            d.rounded_rectangle((177, y + 124, 177 + width, y + 132), radius=4, fill=colors[mid])
    if not layout:
        footnote = (f'Includes {data["_forfeits"]} invalid-choice forfeits. Exact ties share a rank.'
                    if data.get('_forfeits') else f'All {len(data["matches"])} matches included. Exact ties share a rank.')
        text(d, (76, 1126), footnote, fit(d, footnote, 929, 28), MUTED, 'medium')
        text(d, (76, 1184), 'One tournament sample · results can change on new games.', 27, MUTED)
    return im


def ending(t, data, layout):
    im, d = base(t, 'Evidence + sources', layout)
    title(d, ['The Connect Four', 'result'], size=64)
    rows = data.get('_ordered', data.get('standings', []))
    models, colors = context(data)
    leaders = [models[r['model_id']]['name'] for r in rows if r['rank'] == 1]
    verdict = 'Results pending' if layout else ('Most wins: ' + ' & '.join(leaders))
    d.rounded_rectangle((72, 383, 1008, 622), radius=24, fill=NAVY)
    paragraph(d, (108, 414), verdict, 864, 38, WHITE, 'demi')
    result = 'Tournament not yet completed.' if layout else f'{rows[0]["wins"]} wins from {rows[0]["games"]} games' + (' each' if len(leaders) > 1 else '')
    paragraph(d, (108, 535), result, 864, 31, '#CBD9DC', 'medium')
    text(d, (76, 658), 'THE FOUR PARTICIPANTS', 24, COLORS[0], 'bold')
    for i, model in enumerate(data['models']):
        y = 716 + 92 * i
        d.ellipse((78, y + 8, 99, y + 29), fill=colors[model['id']])
        text(d, (115, y), model['name'], fit(d, model['name'], 410, 29), INK, 'demi')
        runtime = model.get('runtime', {})
        sdk = runtime.get('von_sdk') or runtime.get('laya')
        revision = model.get('weights_revision')
        version = ('SDK ' + sdk + ' · ' if sdk else '') + ('revision ' + revision[:7] if revision else '')
        if version:
            text(d, (115, y + 38), version, 19, MUTED, 'regular')
        source = model.get('source_label') or model.get('source') or 'Source verification pending'
        if isinstance(source, dict):
            source = source.get('url', source.get('name', 'Source recorded in evidence'))
        source = str(source).removeprefix('https://').removeprefix('http://').rstrip('/')
        text(d, (1006, y + 4), source, fit(d, source, 467, 23, 16, 'regular'), MUTED, anchor='ra')
    if not layout:
        paragraph(d, (76, 1140), 'Results apply to this tournament.', 927, 32, INK, 'medium')
    return im


def frame(t, data, layout=False):
    if t < 6:
        return intro(t, data, layout)
    if t < 14:
        return protocol(t, data, layout)
    if t < 38:
        return replay(t, data, layout)
    if t < 52:
        return leaderboard(t, data, layout)
    return ending(t, data, layout)


def layout_data(confirmed=None):
    models = confirmed['models'] if confirmed else [{'id': f'placeholder_{i}', 'name': f'Model {chr(65 + i)}', 'subtitle': 'Verified lineup pending',
               'source_label': 'Source pending'} for i in range(4)]
    if len(models) != 4 or len({m['id'] for m in models}) != 4:
        raise ValueError('Lineup preview requires four distinct model identities.')
    return {'complete': False, 'game': 'connect_four', 'models': models,
            'planned_matches': 60, 'matches': [{'id': str(i + 1), 'opening_id': '1',
                'first_player': models[i * 2]['id'], 'second_player': models[i * 2 + 1]['id'],
                'initial_board': [0] * 42, 'moves': []} for i in range(2)],
            'standings': [{'rank': i + 1, 'model_id': model['id']} for i, model in enumerate(models)]}


def previews(data, out, layout=False):
    thumbs = []
    for t in TIMES:
        im = frame(t, data, layout)
        im.save(out / f'frame-{t:04.1f}.png')
        thumbs.append(im.resize((360, 450), Image.Resampling.LANCZOS))
    contact = Image.new('RGB', (1080, 1350), BG)
    for i, im in enumerate(thumbs):
        contact.paste(im, ((i % 3) * 360, (i // 3) * 450))
    contact.save(out / 'contact.png')


def render(data, out):
    target = out / 'Decision_Model_Tournament_Silent.mp4'
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'warning', '-f', 'rawvideo',
               '-vcodec', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
               '-i', '-', '-an', '-c:v', 'libx264', '-preset', 'medium',
               '-b:v', '1000k', '-minrate', '1000k', '-maxrate', '1000k', '-bufsize', '2000k',
               '-x264-params', 'nal-hrd=cbr:filler=1',
               '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        for index in range(DURATION * FPS):
            process.stdin.write(frame(index / FPS, data).tobytes())
        process.stdin.close()
        code = process.wait()
        if code:
            raise RuntimeError(f'Encoder exited with {code}')
    except BaseException:
        process.kill()
        process.wait()
        raise
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--layout', action='store_true', help='Watermarked layout PNGs only; no scores, no movie.')
    parser.add_argument('--lineup-only', action='store_true', help='With --layout, save only a confirmed lineup PNG.')
    parser.add_argument('--render', action='store_true', help='Encode a silent64-second movie after validation.')
    args = parser.parse_args()
    if args.layout and args.render:
        parser.error('Layout placeholders cannot be encoded as a tournament movie.')
    if args.lineup_only and not args.layout:
        parser.error('--lineup-only requires --layout so results remain explicitly pending.')
    if not args.layout and not args.data:
        parser.error('--data is required for real tournament output.')
    data = (layout_data(json.loads(args.data.read_text()) if args.data else None) if args.layout
            else validate(json.loads(args.data.read_text())))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.lineup_only:
        target = args.out_dir / 'lineup-preview.png'
        intro(2, data, True).save(target)
        print(target)
        return
    previews(data, args.out_dir, args.layout)
    if args.render:
        target = render(data, args.out_dir)
        report = {'video': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                  'width': W, 'height': H, 'fps': FPS, 'seconds': DURATION, 'audio': False,
                  'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
                  'video_match_ids': data['video_match_ids'], 'matches': len(data['matches']),
                  'complete': True, 'validation': 'Ranks and records match all legal recorded moves.'}
        (args.out_dir / 'render-manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        print(target)
    else:
        print(args.out_dir / 'contact.png')


if __name__ == '__main__':
    main()
