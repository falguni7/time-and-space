"""Viewer profiles: who a text is written for. The Persona agent changes framing, never facts."""

from __future__ import annotations

from pydantic import BaseModel

LANGUAGES = {"en": "English", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian", "pt": "Portuguese",
             "nl": "Dutch", "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi", "bn": "Bengali",
             "ja": "Japanese", "ko": "Korean", "zh": "Chinese"}


class Profile(BaseModel):
    id: str
    kind: str                      # casual | analyst | supporter | player_follower
    language: str = "en"
    detail: str = "short"          # short | deep
    focus_club: str | None = None  # club code, for supporters
    focus_player: str | None = None  # player id, for player followers
    alert_budget: int = 8          # moments per match this viewer wants (used by the relevance selector)
    accessibility: str | None = None  # e.g. "plain" for simple sentences, screen readers
    note: str | None = None        # why this moment was picked for them (set per moment by the feed)

    @property
    def language_name(self) -> str:
        return LANGUAGES.get(self.language, self.language)

    def max_words(self) -> int:
        return 90 if self.detail == "deep" else 55


KIND_BRIEF = {
    "casual": ("A casual fan watching for fun. Avoid jargon: say 'chance of scoring' not xG/xT, 'chance the pass "
               "arrives' not xPass, 'how tight the marking was' not pressure index. At most one number, rounded "
               "or as 'x in 100'. Warm, simple, like a friend explaining the moment."),
    "analyst": ("A performance analyst. Use the metric names (xG, xT, xPass, pressure index, decision gap) with "
                "their values, mention what the numbers imply tactically. Precise, compact, no hype."),
    "supporter": ("A supporter of {club}. Tell it from {club}'s side: celebrate their good moments, be honest "
                  "but kind about their mistakes. Partisan tone, but every fact stays correct."),
    "player_follower": ("A fan who follows {player}, who plays for {player_club}. Centre the text on {player}'s "
                        "part in the moment. If {player} is not among the moment's players, do not mention {player} "
                        "at all; tell the moment from {player_club}'s side."),
}

DEFAULT_PROFILES = [
    Profile(id="casual", kind="casual"),
    Profile(id="analyst", kind="analyst", detail="deep"),
    Profile(id="supporter", kind="supporter", focus_club="NOR"),
    Profile(id="player_follower", kind="player_follower", focus_player="NOR-18"),
]


def brief(profile: Profile, club_name: str | None = None, player_name: str | None = None,
          player_club: str | None = None) -> str:
    text = KIND_BRIEF[profile.kind].format(club=club_name or "their club", player=player_name or "their player",
                                           player_club=player_club or "their club")
    if profile.accessibility == "plain":
        text += " Use short, plain sentences (easy-read, screen-reader friendly)."
    if profile.note:
        text += f" This moment was picked for them because: {profile.note}."
    return text
