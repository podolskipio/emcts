"""Corpus mining for the emotion channel.

Fits the quantities EmoMCTS reads at plan time from the dialog corpora:

* ``mine_emotion_donation_p4g``  -- the per-emotion valence weights ``w(e)`` consumed by
  ``mcts.emotion_mcts.EmotionAwareMultiObjectiveQ`` as ``EMOTION_VALENCE_MINED``.
* ``mine_emotion_history_donation_p4g`` -- history-conditioned variants of the same lift.
* ``mine_emotion_da_bonus_p4g``  -- per-(dialog-act, emotion) bonuses.
* ``learn_emotion_transition_p4g`` -- the emotion transition prior P(e' | e, da).
* ``mining_corpus``              -- corpus loading, emotion labelling and caching, shared
  by all of the above so they see exactly the same sessions and splits.

Each module is runnable as a script, e.g.::

    python src/emotion_mining/mine_emotion_donation_p4g.py --soft
"""
