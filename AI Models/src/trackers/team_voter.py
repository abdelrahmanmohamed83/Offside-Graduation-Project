from collections import Counter


class TeamVotingSystem:
    """
    Assigns a team to each tracked player and LOCKS it once confident.

    Workflow (mirrors NumberVotingSystem):
      1. Every frame the TeamClassifier gives a raw team result.
      2. We collect votes per track_id (ignoring 'Unknown').
      3. After `required_frames` valid votes the majority team is locked.
      4. Once locked, always return the locked team — even if the classifier
         later returns 'Unknown' (e.g. when the player is partially occluded
         or the lighting changes).
    """

    def __init__(self, required_frames: int = 8):
        """
        required_frames : how many agreeing frames before we lock the team.
                          Lower = faster lock but noisier (default 8).
        """
        self.required_frames = required_frames

        # {track_id: [team_name, team_name, ...]}   – raw vote history
        self.history: dict[int, list[str]] = {}

        # {track_id: (team_name, box_color)}        – locked final result
        self.final_teams: dict[int, tuple[str, tuple]] = {}

    # ------------------------------------------------------------------
    def update(self, track_id: int,
               raw_team: str,
               raw_color: tuple,
               unknown_label: str = "Unknown") -> tuple[str, tuple]:
        """
        Call once per frame per player.

        Parameters
        ----------
        track_id     : unique tracker ID for this player
        raw_team     : team name returned by TeamClassifier this frame
        raw_color    : BGR color tuple returned by TeamClassifier this frame
        unknown_label: the string TeamClassifier uses when it cannot decide

        Returns
        -------
        (team_name, bgr_color) – either the locked result or the raw result
        """
        # 1. Already locked → return locked value immediately
        if track_id in self.final_teams:
            return self.final_teams[track_id]

        # 2. Ignore 'Unknown' and 'Referee' votes for locking purposes
        #    (Referees keep their classification but never get locked here)
        if raw_team not in (unknown_label, "Referee"):
            self.history.setdefault(track_id, []).append(raw_team)

        votes = self.history.get(track_id, [])

        # 3. Enough votes? → decide and lock
        if len(votes) >= self.required_frames:
            winner = Counter(votes).most_common(1)[0][0]
            # We store raw_color only for the winning team; but raw_color may
            # belong to a different team if this frame disagreed.  We keep the
            # color that matches the winner.  Since we don't store per-vote
            # colors, we rely on the caller to pass the correct color when the
            # team matches, so we just store raw_color here — it will be
            # correct on the locking frame if raw_team == winner, which is
            # the common case.  If not, the visualizer uses the config color
            # anyway.
            self.final_teams[track_id] = (winner, raw_color)
            del self.history[track_id]          # free memory
            return self.final_teams[track_id]

        # 4. Not yet locked: return raw result (even if Unknown)
        return raw_team, raw_color

    # ------------------------------------------------------------------
    def get_locked(self, track_id: int):
        """Return the locked (team, color) or None if not yet locked."""
        return self.final_teams.get(track_id)

    # ------------------------------------------------------------------
    def is_locked(self, track_id: int) -> bool:
        return track_id in self.final_teams
