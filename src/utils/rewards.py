
reward_dict = {
    'esc': {
        'worse': -1.0,
        'same': -0.5,
        'better': 0.1,
        'solved': 1.0,
    },
    'cima': {
        'incorrect': -1.0,
        'did not': -0.5,
        'part': 0.5,
        'whole': 1.0,
    },
    # values follow GDPZero's P4G heuristic (PersuasionGame user dialog acts):
    # no donation -> -1.0, negative reaction -> -0.5, neutral -> 0.0,
    # positive reaction -> 0.5, donate -> 1.0
    'p4g': {
        'no donation': -1.0,
        'negative reaction': -0.5,
        'neutral': 0.0,
        'positive reaction': 0.5,
        'donate': 1.0,
    },
    # CraigslistBargain (CBGame user dialog acts). Mirrors CBSystemPlanner.heuristic;
    # the chat planner additionally scales successful deals by the deal price.
    'cb': {
        'no deal': -1.0,
        'deal': 1.0,
    },
}