# What our Connect Four experiment actually showed

Analysis date: 30 September 2026. Games recorded: 29 September 2026.

Every result below comes from our 60 saved games, their request/response logs, our recorded runtime settings, or direct inspection of our video renderer. No published model scores, vendor claims or external benchmark results are used to infer performance.

Dev-0.4B won the tournament. The more useful finding is that its 86.7% win rate coexisted with an almost fixed centre-column choice, poor observed blocking, and a large change between preset and empty-board starts. This sample supports a narrow result for these conditions; it does not establish general game-playing competence.

## Test conditions and coverage

Four fixed public checkpoints played six pairings × five starting positions × both player assignments = 60 games. Each entrant played 30 games, with 15 as X and 15 as O. The five starts were the empty board and two mirrored pairs of four-move openings, fixed before outcomes. They were not checked by a game solver for equal advantage.

All models received the same plain-text rules, board, own piece and legal column descriptions. Each chose its highest-probability legal option using its recorded native input format. We performed no additional training, prompt tuning, tactical search or stronger-model assistance. Von's chains were off. Kev used MLX with a bf16 backbone and float32 head; the others used CPU float32. Thus this is one recorded configuration per model, not an exhaustive test of every configuration.

The logs contain 581 actual model decisions plus 192 preset opening placements, for 773 recorded piece placements. All 581 decisions were legal; no recorded truncation, timeout, infrastructure error, automatic mandatory move or forfeit occurred. All 60 games ended in a four-in-a-row win; there were no draws. These legality results describe selection among supplied legal choices, not the models' ability to discover game rules unaided.

## Tournament outcomes

| Model | Wins–losses | Win rate | Wins going first | Wins going second |
|---|---:|---:|---:|---:|
| Dev-0.4B | 26–4 | 86.7% | 14/15 | 12/15 |
| Von | 17–13 | 56.7% | 7/15 | 10/15 |
| Laya English | 9–21 | 30.0% | 5/15 | 4/15 |
| Kev-4B | 8–22 | 26.7% | 3/15 | 5/15 |

There were 29 first-player wins and 31 second-player wins across the tournament. No first-player advantage was observed in this sample; this does not settle whether an advantage exists in the game.

| Pairing | Observed result |
|---|---|
| Dev vs Kev | Dev 8–2 |
| Dev vs Von | Dev 9–1 |
| Dev vs Laya | Dev 9–1 |
| Von vs Kev | Von 7–3 |
| Laya vs Kev | Laya 7–3 |
| Von vs Laya | Von 9–1 |

## Opening sensitivity

| Model | Empty-board wins | Preset-opening wins |
|---|---:|---:|
| Dev-0.4B | 3/6 (50.0%) | 23/24 (95.8%) |
| Von | 3/6 (50.0%) | 14/24 (58.3%) |
| Laya English | 4/6 (66.7%) | 5/24 (20.8%) |
| Kev-4B | 2/6 (33.3%) | 6/24 (25.0%) |

The overall ranking changed on the six empty-board games per model: Laya had the most wins there. Dev's large aggregate lead came primarily from games with our preset openings. Six empty-board games per entrant is a very small subset; neither table should be generalized to arbitrary starts.

## Move patterns and immediate tactics

Dev chose column 4 on 104 of 105 turns (99.0%) and column 1 once. Von chose column 1 or 3 on 139 of 164 turns (84.8%). Laya chose column 1 on 102 of 165 turns (61.8%). Kev chose column 1 on 66 of 147 turns (44.9%). These are observed concentration patterns; they do not establish why the models chose those columns or what would happen after reordering the legal choices.

We then inspected each recorded model turn with an independent game-rule calculation. This was a post-hoc analysis, not part of the original ranking rule, and it did not alter any moves.

| Model | Took an available immediate win | Played a safe block when one existed |
|---|---:|---:|
| Dev-0.4B | 26/27 (96.3%) | 0/10 (0.0%) |
| Von | 17/39 (43.6%) | 3/18 (16.7%) |
| Laya English | 9/24 (37.5%) | 4/34 (11.8%) |
| Kev-4B | 8/22 (36.4%) | 0/30 (0.0%) |

An **immediate win opportunity** means at least one legal placement would complete four of the moving player's pieces. A successful choice selects any such column.

A **blockable threat** means the moving player has no immediate win of its own, the opponent already has an immediate winning placement, and at least one legal move would leave the opponent with zero immediate winning placements. A successful block selects such a move. Unavoidable one-move losses are excluded from this denominator.

Each denominator counts encountered turns, including repeated positions. Models faced different positions and different numbers of opportunities, so these rates are descriptive, not matched-position tactical accuracy scores. Missed wins may be counted on successive turns if an opponent does not finish the game.

All four models missed at least one available immediate win. All four missed most of the blockable threats they encountered. Dev's 26/27 immediate-win record occurred alongside its nearly constant centre-column choice and 0/10 safe blocks; these logs alone cannot establish whether it recognized the wins strategically. A follow-up comparison against a simple centre-column policy would be needed to test that explanation.

## Game lengths

Games lasted 7–31 total placements, with a median of 11 and mean of 12.88. Those totals include any preset opening. After excluding opening placements, there were 6–27 model decisions per game, median 7 and mean 9.68. Long games are not scored as stronger play; wins alone determined the primary ranking.

## What we measured about response time

| Model | Calls | Median request–response | 95th percentile | Maximum |
|---|---:|---:|---:|---:|
| Dev-0.4B | 105 | 131.6 ms | 141.1 ms | 20653.9 ms |
| Von | 164 | 102.4 ms | 109.3 ms | 4765.6 ms |
| Laya English | 165 | 124.3 ms | 129.6 ms | 381.5 ms |
| Kev-4B | 147 | 75.3 ms | 78.1 ms | 754.3 ms |

These are the elapsed times around the same harness request/response boundary, calculated across every recorded call with no outliers removed. The 95th percentile uses the nearest-rank method. They include communication, input processing, checks and any wait for worker startup/loading; the large Dev maximum includes a startup wait. They are local observations, not an intrinsic speed ranking: CPU and MLX backends differed, models received different positions, and runs occurred in stages without a dedicated controlled latency protocol.

The native timer inside each adapter covers a different scope. Kev excludes input encoding; Dev includes its answer call; Von includes evaluation; Laya includes additional truncation checks. Those native timers are retained in the evidence but are not presented as comparable inference speeds.

## Gameplay video

The 27-second silent video shows two recorded games selected before outcomes, at the same one-second interval per move. Dev beats Kev in the first. Laya beats Von in the second, although Von won nine of their ten games overall. Replay timing is for readability, not inference speed.

## What remains untested

There is no random-policy, centre-column-policy, human or game-solver baseline in this experiment. We did not test whether these models beat such baselines, make optimal moves, understand the board, or transfer to other decision tasks. We also did not test alternative prompts, alternative column orders, fine-tuning, or every runtime configuration.

There were only 63 unique request hashes for Dev's 105 decisions, 129 for Von's 164, 150 for Laya's 165 and 132 for Kev's 147. Every repeated exact request within a model produced the same chosen column. This reflects repeated positions and deterministic choices in this run; the 60 games should not be treated as independent random trials for broad confidence claims.

## Evidence

`evidence/OBSERVED_STATISTICS.json` holds the counts, timing summaries and every tactical event with its match, move number, winning columns, safe columns and actual choice. It is computed solely from the 60 files in `evidence/matches/`. `evidence/AUDIT.json` records the original full-tournament checks. The portable analysis is in `src/analyze_observed.py`.

Frozen protocol hash: `6db078738bd32e9dbb3ddca68a3017bc8784b4874661dc7006812e4539f57921`.
