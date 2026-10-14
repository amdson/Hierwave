
**`ckpop_hard_pen` cr2 arm seed 0 (bench 3/4 seeds)** (`runs/pen_s0.json`; lam = [4.3, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 3/4/4/0/0 | 11 | fail | 3/5/9/0/0 | 0.0 | 11/16 | lm 0.552 dirt 0.434 rough 0.94 float 1.8 (all agree) |
| repair | 3/2/3/0/0 | 8 | pass | 3/3/6/0/0 | 0.6 | 13/16 | lm 0.552 dirt 0.434 rough 0.94 float 1.8 (all agree) |
| jitter1 | 2/1/3/0/0 | 6 | pass | 2/2/5/0/0 | 2.9 | 15/16 | lm 0.553 dirt 0.434 rough 0.94 float 1.5 (all agree) |
| jitter2 | 2/1/3/0/0 | 6 | pass | 2/2/5/0/0 | 3.0 | 15/16 | lm 0.553 dirt 0.434 rough 0.94 float 1.5 (all agree) |
| guided1 | 4/8/2/0/0 | 14 | pass | 4/10/6/0/0 | 30.4 | 13/16 | lm 0.553 dirt 0.435 rough 0.93 float 0.9 (all agree) |
| guided2 | 6/10/4/0/0 | 20 | fail | 6/15/11/0/0 | 37.1 | 15/16 | lm 0.554 dirt 0.435 rough 0.94 float 1.9 (all agree) |
| guided4 | 2/6/2/0/0 | 10 | pass | 2/7/4/0/0 | 34.1 | 15/16 | lm 0.552 dirt 0.435 rough 0.94 float 0.6 (all agree) |
| guided8 | 4/2/2/1/0 | 9 | fail | 4/4/8/8/0 | 32.6 | 15/16 | lm 0.552 dirt 0.434 rough 0.94 float 2.0 (all agree) |
| guided4+repair | 1/1/4/0/0 | 6 | fail | 1/2/6/0/0 | 33.9 | 15/16 | lm 0.552 dirt 0.435 rough 0.94 float 0.5 (all agree) |
| guided8+repair | 4/2/2/1/0 | 9 | fail | 4/4/6/8/0 | 32.6 | 15/16 | lm 0.552 dirt 0.434 rough 0.94 float 2.0 (all agree) |

cells changed per guided step (mean over worlds): [30.4, 15.6, 25.9, 18.7, 17.2, 22.7, 16.9, 19.2]; time per world (GPU 1 unless noted): pass 5.6 ms, repair 1.9 ms, jitter R=1 8.0 ms, R=2 14.1 ms, 8 guided steps 44.0 ms (5.5 ms per step)

**`ckpop_hard_rkl` wave4 W2b, 300 s** (`runs/rkl_w2b.json`; lam = [20.0, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 29/10/6/2/0 | 47 | fail | 30/13/19/7/0 | 0.0 | 2/16 | lm 0.599 dirt 0.495 rough 1.60 float 6.2 (all agree) |
| repair | 23/6/7/0/0 | 36 | fail | 23/10/23/0/0 | 1.3 | 2/16 | lm 0.599 dirt 0.495 rough 1.60 float 5.7 (all agree) |
| jitter1 | 13/2/6/0/0 | 21 | fail | 13/4/23/0/0 | 15.9 | 5/16 | lm 0.599 dirt 0.495 rough 1.62 float 4.1 (all agree) |
| jitter2 | 9/2/6/0/0 | 17 | fail | 9/4/25/0/0 | 19.1 | 7/16 | lm 0.599 dirt 0.495 rough 1.62 float 3.4 (all agree) |
| guided1 | 22/10/7/2/0 | 41 | fail | 22/17/25/15/0 | 148.6 | 4/16 | lm 0.600 dirt 0.496 rough 1.60 float 4.9 (all agree) |
| guided2 | 18/5/4/3/0 | 30 | fail | 18/7/14/23/0 | 164.3 | 4/16 | lm 0.601 dirt 0.496 rough 1.63 float 4.2 (all agree) |
| guided4 | 18/3/7/2/0 | 30 | fail | 18/6/21/16/0 | 175.0 | 5/16 | lm 0.601 dirt 0.496 rough 1.63 float 3.8 (all agree) |
| guided8 | 12/3/0/0/0 | 15 | pass | 12/5/0/0/0 | 193.4 | 7/16 | lm 0.600 dirt 0.495 rough 1.66 float 2.7 (all agree) |
| guided4+repair | 15/3/2/2/0 | 22 | fail | 15/4/6/10/0 | 174.8 | 5/16 | lm 0.601 dirt 0.496 rough 1.63 float 3.5 (all agree) |
| guided8+repair | 11/1/2/0/0 | 14 | pass | 11/2/3/0/0 | 193.6 | 7/16 | lm 0.600 dirt 0.495 rough 1.66 float 2.2 (all agree) |

cells changed per guided step (mean over worlds): [148.6, 59.6, 54.3, 38.2, 43.3, 55.8, 51.7, 7.3]; time per world (GPU 1 unless noted): pass 5.4 ms, repair 1.9 ms, jitter R=1 7.8 ms, R=2 13.7 ms, 8 guided steps 42.8 ms (5.3 ms per step)

**`ckpop_rkl123x2` wave6 C** (`runs/rkl123x2_C.json`; lam = [19.6, 20.0, 20.0, 20.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 23/37/18/3/0 | 81 | fail | 23/46/38/12/0 | 0.0 | 3/16 | lm 0.571 dirt 0.507 rough 0.71 float 4.2 (all agree) |
| repair | 15/15/9/5/0 | 44 | fail | 15/18/25/18/0 | 3.4 | 6/16 | lm 0.571 dirt 0.507 rough 0.71 float 4.1 (all agree) |
| jitter1 | 6/4/5/3/0 | 18 | fail | 6/6/15/15/0 | 22.4 | 8/16 | lm 0.570 dirt 0.507 rough 0.70 float 1.2 (all agree) |
| jitter2 | 5/4/5/3/0 | 17 | fail | 5/5/16/15/0 | 23.4 | 9/16 | lm 0.570 dirt 0.507 rough 0.71 float 1.1 (all agree) |
| guided1 | 21/29/9/3/0 | 62 | fail | 21/45/23/13/0 | 158.4 | 6/16 | lm 0.572 dirt 0.506 rough 0.73 float 4.9 (all agree) |
| guided2 | 14/16/4/2/0 | 36 | fail | 14/22/5/2/0 | 180.1 | 10/16 | lm 0.575 dirt 0.507 rough 0.70 float 2.2 (all agree) |
| guided4 | 5/3/7/2/0 | 17 | fail | 5/4/20/3/0 | 188.3 | 12/16 | lm 0.573 dirt 0.508 rough 0.69 float 2.0 (all agree) |
| guided8 | 11/10/5/2/0 | 28 | fail | 11/16/12/6/0 | 190.6 | 12/16 | lm 0.573 dirt 0.507 rough 0.69 float 7.6 (all agree) |
| guided4+repair | 5/1/5/3/0 | 14 | fail | 5/1/14/6/0 | 188.4 | 12/16 | lm 0.573 dirt 0.508 rough 0.69 float 2.0 (all agree) |
| guided8+repair | 10/6/5/1/0 | 22 | fail | 10/10/14/5/0 | 190.4 | 12/16 | lm 0.573 dirt 0.507 rough 0.69 float 7.6 (all agree) |

cells changed per guided step (mean over worlds): [158.4, 97.8, 46.8, 44.4, 36.1, 29.1, 26.6, 28.1]; time per world (GPU 1 unless noted): pass 5.5 ms, repair 1.9 ms, jitter R=1 7.7 ms, R=2 13.7 ms, 8 guided steps 43.5 ms (5.4 ms per step)

**`ckpop_hard` wave4 W1 (no violation term)** (`runs/hard_w1.json`; lam = [0.0, 6.3, 6.1, 0.0, 0.0])

| state | violated L0/L1/L2/L3/L4 (16 worlds) | total | budget | graded mass L0..L4 | cells changed / world | valid worlds | stats vs raw (bench rule) |
|---|---|---|---|---|---|---|---|
| raw | 52/46/12/3/0 | 113 | fail | 52/58/23/20/0 | 0.0 | 0/16 | lm 0.628 dirt 0.513 rough 1.86 float 11.4 (all agree) |
| repair | 38/18/5/5/0 | 66 | fail | 38/26/10/21/0 | 4.6 | 2/16 | lm 0.628 dirt 0.513 rough 1.85 float 8.8 (all agree) |
| jitter1 | 18/13/3/5/0 | 39 | fail | 18/20/10/21/0 | 31.3 | 4/16 | lm 0.629 dirt 0.513 rough 1.87 float 5.1 (all agree) |
| jitter2 | 13/12/3/4/0 | 32 | fail | 13/17/10/15/0 | 35.8 | 6/16 | lm 0.629 dirt 0.513 rough 1.87 float 4.2 (all agree) |
| guided1 | 46/26/9/0/0 | 81 | fail | 46/34/21/0/0 | 323.6 | 4/16 | lm 0.627 dirt 0.519 rough 1.88 float 10.0 (all agree) |
| guided2 | 27/21/7/1/0 | 56 | fail | 27/26/16/6/0 | 347.6 | 5/16 | lm 0.626 dirt 0.520 rough 1.93 float 6.1 (all agree) |
| guided4 | 18/6/2/0/0 | 26 | pass | 18/8/5/0/0 | 369.4 | 9/16 | lm 0.622 dirt 0.520 rough 1.92 float 3.9 (all agree) |
| guided8 | 15/4/1/0/0 | 20 | pass | 15/6/3/0/0 | 374.1 | 9/16 | lm 0.624 dirt 0.520 rough 1.94 float 4.1 (all agree) |
| guided4+repair | 14/3/2/0/0 | 19 | pass | 14/4/5/0/0 | 369.3 | 9/16 | lm 0.622 dirt 0.520 rough 1.93 float 3.4 (all agree) |
| guided8+repair | 15/4/0/0/0 | 19 | pass | 15/7/0/0/0 | 373.9 | 9/16 | lm 0.624 dirt 0.520 rough 1.94 float 4.1 (all agree) |

cells changed per guided step (mean over worlds): [323.6, 105.8, 78.6, 39.8, 18.8, 31.6, 12.1, 14.6]; time per world (GPU 1 unless noted): pass 5.7 ms, repair 1.9 ms, jitter R=1 8.1 ms, R=2 14.5 ms, 8 guided steps 45.3 ms (5.7 ms per step)
