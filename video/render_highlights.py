"""Render curated real Connect Four highlights; never call or rerun a model.

Usage: python render_highlights.py --data evidence/results.json \
  --selection highlights/selection.json --output highlights --review-dir review
Requires Pillow, imageio-ffmpeg, and sibling render_tournament.py.
--preview-only renders review PNGs without exporting movies.
"""
from pathlib import Path
import argparse
import functools
import hashlib
import json
import math
import re
import struct
import subprocess

from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
from render_tournament import context, validate, winning_cells, winning_colors

W, H, FPS = 1080, 1350, 30
BG, INK, MUTED, LINE, WHITE, NAVY = '#F5F3ED', '#152B38', '#5B6A72', '#D8DDD9', '#FFFFFF', '#112C3B'
GOLD = '#F2C765'
INITIAL_HOLD, MOVE_SECONDS, DROP_SECONDS, FINAL_HOLD = 1.5, 1.0, .3, 2.2
FONT_PATH = None


@functools.lru_cache(maxsize=150)
def font(size, bold=False):
    candidates = [FONT_PATH] if FONT_PATH else [
        '/System/Library/Fonts/Avenir Next.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        'Arial.ttf']
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, round(size), index=(0 if bold else 7) if str(candidate).endswith('Avenir Next.ttc') else 0)
        except (OSError, TypeError):
            pass
    raise RuntimeError('Install DejaVu Sans or supply --font with a readable TrueType font.')


def text(draw, xy, value, size=30, color=INK, bold=False, anchor=None):
    draw.text(xy, str(value), font=font(size, bold), fill=color, anchor=anchor)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def drop(board, color, column):
    result = list(board)
    for row in range(5, -1, -1):
        index = row * 7 + column - 1
        if not result[index]:
            result[index] = color
            return result
    raise ValueError('Illegal full-column move.')


def immediate_wins(board, color):
    return [column for column in range(1, 8) if not board[column - 1]
            and color in winning_colors(drop(board, color, column))]


def prepare(data, manifest, results_path):
    validate(data)
    if sha(results_path) != manifest['results_sha256']:
        raise ValueError('Results hash differs from the verified selection manifest.')
    matches = {m['id']: m for m in data['matches']}
    clips, audits = [], []
    for item in manifest['clips']:
        match = matches[item['match_id']]
        source_path = results_path.parent / 'matches' / (match['id'] + '.json')
        if sha(source_path) != item['source_match_sha256']:
            raise ValueError('Selected match file hash differs from audited source.')
        if json.loads(source_path.read_text()) != match:
            raise ValueError('Individual source match differs from aggregate results.')
        start_ply = item['start_ply']
        opening_count = len(match['opening_columns'])
        if not opening_count <= start_ply < len(match['moves']):
            raise ValueError('Start must follow all preset moves and precede the final move.')
        initial = match['initial_board'] if start_ply == 0 else match['moves'][start_ply - 1]['board']
        sequence = match['moves'][start_ply:]
        if any(m['opening'] for m in sequence):
            raise ValueError('Preset opening plies must never be animated as model decisions.')
        if any(m.get('decision', {}).get('source') == 'forced_by_rules' for m in sequence):
            raise ValueError('This model-decision label requires recorded model choices.')
        block_audits = []
        for event in item['blocks']:
            ply = event['ply']
            if not start_ply < ply <= len(match['moves']):
                raise ValueError('Block annotation is outside the replayed sequence.')
            move = match['moves'][ply - 1]
            board = match['initial_board'] if ply == 1 else match['moves'][ply - 2]['board']
            color = move['player']
            mid = match['first_player'] if color == 1 else match['second_player']
            own_wins = immediate_wins(board, color)
            threats = immediate_wins(board, 3 - color)
            safe = [c for c in range(1, 8) if not board[c - 1]
                    and not immediate_wins(drop(board, color, c), 3 - color)]
            if (mid != event['model'] or move['column'] != event['column'] or not threats
                    or own_wins or move['column'] not in safe
                    or immediate_wins(move['board'], 3 - color)):
                raise ValueError('Claimed safe block is not supported by the recorded board.')
            changed = [i for i, (a, b) in enumerate(zip(board, move['board'])) if a != b]
            assert len(changed) == 1
            block_audits.append({**event, 'piece_cell_index': changed[0],
                'own_immediate_winning_columns': own_wins,
                'opponent_immediate_winning_columns_before': threats,
                'safe_columns': safe, 'verified': True})
        length = INITIAL_HOLD + (len(sequence) - 1) * MOVE_SECONDS + DROP_SECONDS + FINAL_HOLD
        clip = {**item, 'match': match, 'initial_board': initial, 'sequence': sequence,
                'blocks': {event['ply']: event for event in block_audits}, 'seconds': length,
                'opening_count': opening_count}
        clips.append(clip)
        audits.append({'filename': item['filename'], 'match_id': match['id'],
            'source_match_sha256': item['source_match_sha256'],
            'source_match_canonical_json_sha256': object_sha(match),
            'start_ply': start_ply, 'end_ply': len(match['moves']),
            'initial_board': initial, 'final_board': sequence[-1]['board'],
            'animated_plies': [m['ply'] for m in sequence],
            'animated_columns': [m['column'] for m in sequence],
            'animated_model_decisions': len(sequence),
            'preset_opening_plies': opening_count,
            'no_opening_plies_animated': True, 'all_source_boards_and_terminal_result_validated': True,
            'no_skipped_internal_plies': True, 'safe_block_annotations': block_audits,
            'winner': match['winner'], 'winning_cells': winning_cells(sequence[-1]['board']),
            'seconds': length})
    return clips, audits


def state_at(t, clip):
    seq = clip['sequence']
    elapsed = t - INITIAL_HOLD
    if elapsed < 0:
        return {'board': clip['initial_board'], 'decision': 0, 'ply': clip['start_ply'],
            'moving': None, 'last_cell': None, 'done': False, 'next': seq[0]['player'], 'event': None}
    index = math.floor(elapsed / MOVE_SECONDS + 1e-8)
    fraction = 1 if index >= len(seq) else min(1, (elapsed - index * MOVE_SECONDS) / DROP_SECONDS)
    index = min(index, len(seq) - 1)
    before = clip['initial_board'] if index == 0 else seq[index - 1]['board']
    move = seq[index]
    changed = next(k for k, (a, b) in enumerate(zip(before, move['board'])) if a != b)
    landed = fraction >= 1 - 1e-8
    visible_ply = move['ply'] if landed else move['ply'] - 1
    past_blocks = [v for k, v in clip['blocks'].items() if k <= visible_ply]
    return {'board': move['board'] if landed else before, 'decision': index + 1,
        'ply': move['ply'], 'moving': None if landed else (changed, move['player'], fraction),
        'last_cell': changed if landed else None, 'done': landed and index == len(seq) - 1,
        'next': 3 - move['player'] if landed else move['player'],
        'event': past_blocks[-1] if past_blocks else None}


def draw_board(draw, state, colors):
    x, y, cell, pad, radius = 81, 327, 126, 18, 49
    draw.rounded_rectangle((x, y, x + 7 * cell + 2 * pad, y + 6 * cell + 2 * pad), radius=28, fill=NAVY)
    for col in range(7):
        text(draw, (x + pad + (col + .5) * cell, y - 24), col + 1, 27, MUTED, True, 'mm')
    wins = winning_cells(state['board']) if state['done'] else []
    event_cell = state['event']['piece_cell_index'] if state['event'] and not state['done'] else None
    for k, token in enumerate(state['board']):
        row, column = divmod(k, 7)
        cx, cy = x + pad + (column + .5) * cell, y + pad + (row + .5) * cell
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=colors[token-1] if token else BG)
        if token:
            text(draw, (cx, cy), 'X' if token == 1 else 'O', 43, WHITE, True, 'mm')
        if k == state['last_cell']:
            draw.ellipse((cx-radius-4, cy-radius-4, cx+radius+4, cy+radius+4), outline=WHITE, width=3)
        if k == event_cell or k in wins:
            draw.ellipse((cx-radius-6, cy-radius-6, cx+radius+6, cy+radius+6), outline=GOLD, width=6)
    if wins:
        centers = [(x+pad+(k%7+.5)*cell, y+pad+(k//7+.5)*cell) for k in wins]
        draw.line((centers[0], centers[-1]), fill=GOLD, width=8)
    if state['moving']:
        k, token, progress = state['moving']
        row, column = divmod(k, 7)
        cx = x + pad + (column + .5) * cell
        target = y + pad + (row + .5) * cell
        cy = y + 23 + (target - y - 23) * progress * progress
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=colors[token-1])
        text(draw, (cx, cy), 'X' if token == 1 else 'O', 43, WHITE, True, 'mm')


def frame(t, data, clip):
    image = Image.new('RGB', (W, H), BG)
    draw = ImageDraw.Draw(image)
    models, colors = context(data)
    match = clip['match']
    state = state_at(t, clip)
    for j, mid in enumerate((match['first_player'], match['second_player'])):
        x = 72 + j * 480
        draw.rounded_rectangle((x, 48, x + 456, 131), radius=18, fill=colors[mid])
        text(draw, (x + 25, 89), 'X' if j == 0 else 'O', 36, WHITE, True, 'lm')
        text(draw, (x + 78, 89), models[mid]['name'], 38, WHITE, True, 'lm')
    text(draw, (72, 174), f'REPLAY MOVE {state["decision"]} / {len(clip["sequence"])}', 28, MUTED, True, 'lm')
    text(draw, (1008, 174), f'GAME MOVE {state["ply"]} / {len(match["moves"])}', 28, MUTED, True, 'rm')
    actor = match['first_player'] if state['next'] == 1 else match['second_player']
    label = (models[match['winner']]['name'] + ' wins · four in a row') if state['done'] else ('Turn: ' if state['moving'] or not state['decision'] else 'Next: ') + models[actor]['name']
    text(draw, (72, 222), label, 33, INK, True, 'lm')
    if clip['start_ply'] > clip['opening_count']:
        context_label = f'Excerpt · moves {clip["start_ply"] + 1}–{len(match["moves"])} · starts after {clip["start_ply"]} recorded moves'
    elif clip['opening_count']:
        context_label = f'{clip["opening_count"]} preset opening moves · shown before model play'
    else:
        context_label = 'Full game · empty-board start'
    text(draw, (72, 266), context_label, 25, MUTED, False, 'lm')
    draw_board(draw, state, [colors[match['first_player']], colors[match['second_player']]])
    draw.rounded_rectangle((72, 1150, 1008, 1237), radius=17, fill=WHITE, outline=LINE, width=2)
    if state['done']:
        badge = 'FINAL'
        message = models[match['winner']]['name'] + ' connects four'
    elif state['event']:
        event = state['event']
        badge = f'MOVE {event["ply"]}'
        name = 'Laya' if event['model'] == 'laya' else models[event['model']]['name']
        message = f'{name} blocks column {event["column"]}'
    elif not state['decision']:
        badge = 'START'
        message = f'After {clip["start_ply"]} recorded moves' if clip['start_ply'] > clip['opening_count'] else (f'{clip["opening_count"]} preset opening moves' if clip['opening_count'] else 'Empty board · Von plays first')
    else:
        badge = 'REPLAY'
        message = 'Recorded moves · one second per decision'
    draw.rounded_rectangle((92, 1169, 241, 1218), radius=11, fill=NAVY)
    text(draw, (166.5, 1193.5), badge, 25, WHITE, True, 'mm')
    if draw.textlength(message, font=font(32, True)) > 725:
        raise ValueError(f'Annotation does not fit: {message}')
    text(draw, (264, 1193.5), message, 32, INK, True, 'lm')
    text(draw, (540, 1290), 'Selected highlight · replay pace, not model speed', 25, MUTED, False, 'mm')
    return image


def sample_times(clip):
    times = [.5, 2.0]
    times += [INITIAL_HOLD + (ply - clip['start_ply'] - 1) * MOVE_SECONDS + .5 for ply in clip['blocks']]
    times += [clip['seconds'] - 1]
    return sorted(set(times))


def contact(paths, target):
    thumbs = []
    for path in paths:
        thumb = Image.open(path).convert('RGB').resize((360, 450), Image.Resampling.LANCZOS)
        thumbs.append(thumb)
    image = Image.new('RGB', (360 * 3, 450 * math.ceil(len(thumbs) / 3)), LINE)
    for i, thumb in enumerate(thumbs):
        image.paste(thumb, ((i % 3) * 360, (i // 3) * 450))
    image.save(target)


def render(data, clip, path):
    if path.exists():
        raise ValueError(f'Refusing to overwrite existing movie {path.name}')
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-v', 'warning', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
        '-preset', 'medium', '-b:v', '1000k', '-minrate', '1000k', '-maxrate', '1000k', '-bufsize', '2000k',
        '-x264-params', 'nal-hrd=cbr:filler=1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for index in range(round(clip['seconds'] * FPS)):
            process.stdin.write(frame(index / FPS, data, clip).tobytes())
        process.stdin.close()
        if process.wait():
            raise RuntimeError('Video encoding failed.')
    except BaseException:
        process.kill()
        process.wait()
        raise


def inspect_media(path, clip, review_dir):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    probe = subprocess.run([ffmpeg, '-hide_banner', '-xerror', '-i', str(path), '-map', '0:v:0', '-f', 'null', '-'], capture_output=True, text=True, check=True)
    stream = probe.stderr.split('Stream mapping:')[0]
    assert 'Audio:' not in stream and stream.count('Video:') == 1
    assert 'h264' in stream and '1080x1350' in stream and '30 fps' in stream and 'yuv420p' in stream
    bitrate = int(re.search(r',\s*(\d+) kb/s,', next(line for line in stream.splitlines() if 'Video:' in line)).group(1))
    assert bitrate >= 192
    decoded_frames = int(re.findall(r'frame=\s*(\d+)', probe.stderr)[-1])
    assert decoded_frames == round(clip['seconds'] * FPS)
    expected_duration = f'{int(clip["seconds"]) // 60:02d}:{int(clip["seconds"]) % 60:02d}.00'
    assert f'Duration: 00:{expected_duration}' in stream
    raw = path.read_bytes()
    atoms, offset = [], 0
    while offset + 8 <= len(raw):
        size, kind = struct.unpack('>I4s', raw[offset:offset+8])
        if size == 1:
            size = struct.unpack('>Q', raw[offset+8:offset+16])[0]
        if not size:
            size = len(raw) - offset
        atoms.append(kind.decode('ascii'))
        offset += size
    assert atoms.index('moov') < atoms.index('mdat')
    times = sample_times(clip)
    selection = '+'.join(f'eq(n\\,{round(t * FPS)})' for t in times)
    subprocess.run([ffmpeg, '-hide_banner', '-v', 'error', '-i', str(path), '-vf', f'select={selection}', '-fps_mode', 'vfr', str(review_dir / 'decoded-%02d.png')], check=True)
    paths = sorted(review_dir.glob('decoded-*.png'))
    assert len(paths) == len(times)
    contact(paths, review_dir / 'decoded-contact.png')
    return {'video_sha256': sha(path), 'bytes': path.stat().st_size, 'width': W, 'height': H,
        'fps': FPS, 'seconds': clip['seconds'], 'video_codec': 'H.264', 'pixel_format': 'yuv420p',
        'audio_streams': 0, 'video_bitrate_kbps': bitrate, 'minimum_192kbps_pass': True,
        'full_decode_exit_code': probe.returncode, 'decoded_frames': decoded_frames,
        'faststart': True, 'sampled_decoded_frame_seconds': times,
        'visual_review': 'pending'}


def main():
    global FONT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='Output directory, not a movie filename.')
    parser.add_argument('--review-dir', required=True, type=Path)
    parser.add_argument('--font', type=str)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    FONT_PATH = args.font
    data = json.loads(args.data.read_text())
    manifest = json.loads(args.selection.read_text())
    clips, audits = prepare(data, manifest, args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    args.review_dir.mkdir(parents=True, exist_ok=True)
    for clip, audit in zip(clips, audits):
        directory = args.review_dir / Path(clip['filename']).stem
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(sample_times(clip)):
            path = directory / f'source-{i:02d}.png'
            frame(t, data, clip).save(path)
            paths.append(path)
        contact(paths, directory / 'source-contact.png')
        if not args.preview_only:
            movie = args.output / clip['filename']
            render(data, clip, movie)
            audit['media'] = inspect_media(movie, clip, directory)
            print(json.dumps({'filename': clip['filename'], 'media': audit['media']}), flush=True)
    report = {'status': 'preview_only' if args.preview_only else 'media_pass_visual_review_pending',
        'selection': 'Curated after results; these are highlights, not prespecified or representative matches.',
        'results_sha256': sha(args.data), 'selection_manifest_sha256': sha(args.selection),
        'renderer_sha256': sha(__file__), 'validator_sha256': sha(Path(__file__).with_name('render_tournament.py')),
        'all_60_source_games_independently_validated': True,
        'constant_seconds_between_model_move_starts': MOVE_SECONDS,
        'drop_animation_seconds': DROP_SECONDS, 'initial_board_hold_seconds': INITIAL_HOLD,
        'final_board_hold_seconds': FINAL_HOLD, 'audio': 'none',
        'model_calls': 0, 'source_game_evidence_modified': False,
        'annotations': 'Immediate-threat blocks verified from legal one-move outcomes; no inferred model intent.',
        'clips': audits}
    target = args.review_dir / 'preview-audit.json' if args.preview_only else args.output / 'QA.json'
    target.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
