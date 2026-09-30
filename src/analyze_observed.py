"""Descriptive statistics from saved tournament moves; never runs a model."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

DEFAULT_EVIDENCE = Path(__file__).resolve().parents[1] / 'evidence'
IDS = ['dev', 'von', 'laya', 'kev']


def legal(board):
    return [c for c in range(7) if board[c] == 0]


def put(board, column, player):
    result = list(board)
    row = next(r for r in range(5, -1, -1) if result[r * 7 + column] == 0)
    result[row * 7 + column] = player
    return result


def won(board, player):
    for r in range(6):
        for c in range(7):
            for dr, dc in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                if 0 <= r + 3 * dr < 6 and 0 <= c + 3 * dc < 7:
                    if all(board[(r + k * dr) * 7 + c + k * dc] == player for k in range(4)):
                        return True
    return False


def winning_columns(board, player):
    return {c for c in legal(board) if won(put(board, c, player), player)}


def descriptive(values):
    ordered = sorted(values)
    return {'n': len(values), 'mean': round(mean(values), 3), 'median': round(median(values), 3),
            'p95_nearest_rank': round(ordered[math.ceil(.95 * len(values)) - 1], 3),
            'min': round(min(values), 3), 'max': round(max(values), 3)}


def analyze(evidence):
    """Recompute statistics from recorded moves; return a JSON-compatible object."""
    evidence = Path(evidence)
    matches = [json.loads(p.read_text()) for p in sorted((evidence / 'matches').glob('*.json'))]
    assert len(matches) == 60
    totals = {m: Counter() for m in IDS}
    seats = {m: {'first': Counter(), 'second': Counter()} for m in IDS}
    pairs = defaultdict(Counter)
    opening = defaultdict(Counter)
    response_ms, native_ms = defaultdict(list), defaultdict(list)
    requests = {m: defaultdict(list) for m in IDS}
    tactical_events = []
    columns = {m: Counter() for m in IDS}
    whole_lengths, model_lengths = [], []
    for game in matches:
        assert game['status'] == 'complete'
        pair = tuple(sorted(game['pair']))
        pairs[pair][game['winner'] or 'draw'] += 1
        for seat, mid in [('first', game['first_player']), ('second', game['second_player'])]:
            outcome = 'draws' if game['draw'] else 'wins' if game['winner'] == mid else 'losses'
            totals[mid][outcome] += 1
            seats[mid][seat][outcome] += 1
            opening[game['opening_id']][mid + '_' + outcome] += 1
        board = [0] * 42
        count_decisions = 0
        for index, move in enumerate(game['moves']):
            player = move['player']
            col = move['column'] - 1
            assert not won(board, 1) and not won(board, 2)
            assert col in legal(board)
            next_board = put(board, col, player)
            assert next_board == move['board']
            if not move['opening']:
                d = move['decision']
                mid = d['model_id']
                assert mid == (game['first_player'] if player == 1 else game['second_player'])
                assert d['source'] == 'model'
                totals[mid]['legal_decisions'] += 1
                count_decisions += 1
                columns[mid][move['column']] += 1
                response_ms[mid].append(d['elapsed_ns'] / 1e6)
                meta = d['response']['metadata']
                assert not meta.get('truncated') and not meta.get('state_truncated')
                native_ms[mid].append(meta.get('inference_ms', meta.get('elapsed_ms')))
                requests[mid][d['request_sha256']].append(col + 1)
                own_wins = winning_columns(board, player)
                opponent_wins = winning_columns(board, 3 - player)
                # An immediate own win ends the game and takes priority over defense.
                safe = {c for c in legal(board)
                        if won(put(board, c, player), player)
                        or not winning_columns(put(board, c, player), 3 - player)}
                event = {'match_id': game['id'], 'ply': index + 1, 'model': mid,
                         'chosen_column': col + 1, 'own_winning_columns': sorted(c + 1 for c in own_wins),
                         'opponent_threat_columns_before': sorted(c + 1 for c in opponent_wins),
                         'safe_columns': sorted(c + 1 for c in safe)}
                if own_wins:
                    totals[mid]['immediate_win_opportunities'] += 1
                    totals[mid]['immediate_wins_taken'] += int(col in own_wins)
                    event['category'] = 'immediate_win'
                    event['success'] = col in own_wins
                    tactical_events.append(event)
                elif opponent_wins and safe:
                    totals[mid]['blockable_threat_turns'] += 1
                    totals[mid]['safe_blocks_played'] += int(col in safe)
                    event['category'] = 'blockable_threat'
                    event['success'] = col in safe
                    tactical_events.append(event)
                elif opponent_wins and not safe:
                    totals[mid]['unavoidable_immediate_loss_turns'] += 1
                if not own_wins and not opponent_wins and safe:
                    totals[mid]['no_current_threat_turns_with_safe_move'] += 1
                    totals[mid]['created_avoidable_immediate_threat'] += int(col not in safe)
            board = next_board
        assert won(board, game['winner_color']) and not game['draw']
        whole_lengths.append(len(game['moves']))
        model_lengths.append(count_decisions)
    assert sum(t['wins'] for t in totals.values()) == 60
    assert sum(t['legal_decisions'] for t in totals.values()) == 581
    output = {
        'scope': 'Only the 60 saved local Connect Four games; no external benchmark scores.',
        'analysis': 'Post hoc descriptive analysis; tactical metrics were not prespecified tournament ranking criteria.',
        'source_results_sha256': hashlib.sha256((evidence / 'results.json').read_bytes()).hexdigest(),
        'models': {mid: {'counts': dict(totals[mid]), 'wins_percent': round(100 * totals[mid]['wins'] / 30, 2),
                         'seats': seats[mid], 'harness_response_ms': descriptive(response_ms[mid]),
                         'adapter_native_timer_ms': descriptive(native_ms[mid]),
                         'unique_request_hashes': len(requests[mid]),
                         'request_hashes_with_different_choices': sum(len(set(v)) > 1 for v in requests[mid].values()),
                         'chosen_columns': dict(sorted(columns[mid].items()))} for mid in IDS},
        'pairwise': [{'models': list(k), 'wins': dict(v)} for k, v in pairs.items()],
        'by_opening': dict(opening),
        'game_total_plies': descriptive(whole_lengths),
        'game_model_decisions': descriptive(model_lengths),
        'all_recorded_plies': sum(whole_lengths), 'preset_opening_plies': sum(whole_lengths) - sum(model_lengths),
        'tactical_events': tactical_events,
        'tactical_definitions': {
            'immediate_win': 'A gravity-legal placement completing four of the moving player; success chooses any such column.',
            'blockable_threat': 'No immediate own win, at least one opponent immediate winning column before the move, and at least one legal own move leaving zero opponent immediate wins; success chooses such a move.',
            'created_avoidable_immediate_threat': 'Neither player has an immediate win before the move, a safe own move exists, but chosen move leaves an opponent immediate win.',
            'unit': 'A model decision occurrence; repeated positions count repeatedly and exposure differs across models.'},
        'timing_limitations': [
            'Harness times use the same request/response boundary for all models, including IPC, validation and any startup/loading wait.',
            'Models use different backends; games have different positions, were run in stages, and no controlled latency benchmark was prespecified.',
            'Native timers have different boundaries: Kev excludes encoding, Dev includes answer encoding, Von includes evaluation, Laya includes extra truncation checks.',
            'Do not compare native timer values as like-for-like model speed or confuse either timing with video pacing.'],
        'uncertainty': 'Fixed openings and deterministic choices produce repeated states; no confidence intervals or global ranking claims are justified by these counts alone.'}
    # Match JSON key types used by the archived analysis (column keys are strings).
    return json.loads(json.dumps(output, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE,
                        help='Recorded evidence directory (default: repository evidence/).')
    parser.add_argument('--output', type=Path,
                        help='Destination JSON (default: EVIDENCE/OBSERVED_STATISTICS.json).')
    args = parser.parse_args(argv)
    output = analyze(args.evidence)
    destination = args.output or args.evidence / 'OBSERVED_STATISTICS.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': 'recomputed', 'games': 60,
                      'model_decisions': sum(m['counts']['legal_decisions'] for m in output['models'].values()),
                      'statistics': destination.name}, indent=2))


if __name__ == '__main__':
    main()
