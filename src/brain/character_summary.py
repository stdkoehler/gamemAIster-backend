"""Compact per-system character sheet summaries for LLM injection."""


def _track(t: dict) -> str:
    return f"{t.get('current', 0)}/{t.get('max', 0)}"


def _shadowrun(c: dict) -> str:
    a = c.get("attributes", {})
    d = c.get("damage", {})
    attrs = (
        f"BOD {a.get('Body',0)} AGI {a.get('Agility',0)} REA {a.get('Reaction',0)} "
        f"STR {a.get('Strength',0)} WIL {a.get('Willpower',0)} LOG {a.get('Logic',0)} "
        f"INT {a.get('Intuition',0)} CHA {a.get('Charisma',0)} "
        f"ESS {a.get('Essence',6)} EDG {a.get('Edge',0)}"
    )
    tracks = f"Physical {_track(d.get('physical', {}))} Stun {_track(d.get('stun', {}))}"
    return f"{c.get('name','?')} ({c.get('metatype','')} {c.get('archetype','')}) | {attrs} | {tracks}"


def _vampire(c: dict) -> str:
    a = c.get("attributes", {})
    nature = c.get("nature", "mortal")
    identity = f"{nature.capitalize()}"
    if c.get("clan"):
        identity += f", Clan {c['clan']}"
    if c.get("generation"):
        identity += f", Gen. {c['generation']}th"
    attrs = (
        f"STR {a.get('Strength',0)} DEX {a.get('Dexterity',0)} STA {a.get('Stamina',0)} "
        f"CHA {a.get('Charisma',0)} MAN {a.get('Manipulation',0)} "
        f"INT {a.get('Intelligence',0)} WIT {a.get('Wits',0)}"
    )
    tracks = f"HP {_track(c.get('health', {}))} WP {_track(c.get('willpower', {}))}"
    hunger = f"Hunger {c.get('hunger', 0)}/5" if nature in ("kindred", "thin-blood") else ""
    parts = [f"{c.get('name','?')} ({identity})", attrs, tracks]
    if hunger:
        parts.append(hunger)
    return " | ".join(parts)


def _cthulhu(c: dict) -> str:
    ch = c.get("characteristics", {})
    attrs = (
        f"STR {ch.get('STR',0)} CON {ch.get('CON',0)} DEX {ch.get('DEX',0)} "
        f"INT {ch.get('INT',0)} POW {ch.get('POW',0)} EDU {ch.get('EDU',0)}"
    )
    tracks = (
        f"HP {_track(c.get('hitPoints', {}))} "
        f"SAN {_track(c.get('sanity', {}))} "
        f"MP {_track(c.get('magicPoints', {}))} "
        f"LCK {c.get('luck', 0)}"
    )
    return f"{c.get('name','?')} ({c.get('occupation','')}, {c.get('era','')}) | {attrs} | {tracks}"


def _seventh_sea(c: dict) -> str:
    t = c.get("traits", {})
    traits = (
        f"BRN {t.get('Brawn',0)} FIN {t.get('Finesse',0)} RES {t.get('Resolve',0)} "
        f"WIT {t.get('Wits',0)} PAN {t.get('Panache',0)}"
    )
    tracks = f"Wounds {_track(c.get('wounds', {}))} Hero {c.get('heroPoints', 0)}"
    return f"{c.get('name','?')} ({c.get('nation','')}) | {traits} | {tracks}"


def _expanse(c: dict) -> str:
    a = c.get("abilities", {})
    attrs = (
        f"ACC {a.get('Accuracy',0)} COM {a.get('Communication',0)} CON {a.get('Constitution',0)} "
        f"DEX {a.get('Dexterity',0)} FIG {a.get('Fighting',0)} INT {a.get('Intelligence',0)} "
        f"PER {a.get('Perception',0)} STR {a.get('Strength',0)} WIL {a.get('Willpower',0)}"
    )
    tracks = f"Health {_track(c.get('health', {}))} Fortune {c.get('fortune', 0)}"
    return f"{c.get('name','?')} ({c.get('origin','')}, {c.get('background','')}) | {attrs} | {tracks}"


def _slavic(c: dict) -> str:
    a = c.get("attributes", {})
    d = c.get("attributeDamage", {})
    tracks = (
        f"STR {d.get('Strength', a.get('Strength',0))}/{a.get('Strength',0)} "
        f"AGI {d.get('Agility', a.get('Agility',0))}/{a.get('Agility',0)} "
        f"WIT {d.get('Wits', a.get('Wits',0))}/{a.get('Wits',0)} "
        f"EMP {d.get('Empathy', a.get('Empathy',0))}/{a.get('Empathy',0)}"
    )
    return f"{c.get('name','?')} ({c.get('kin','')}, {c.get('calling','')}) | {tracks}"


_FORMATTERS = {
    "shadowrun": _shadowrun,
    "vampire_the_masquerade": _vampire,
    "call_of_cthulhu": _cthulhu,
    "seventh_sea": _seventh_sea,
    "expanse": _expanse,
    "slavic": _slavic,
}


def to_summary(game_type: str, sheets: list[dict]) -> str:
    """
    Return a compact party-status block for LLM injection.
    sheets: list of dicts with keys {content, is_protagonist}.
    Returns empty string when there are no sheets.
    """
    if not sheets:
        return ""

    formatter = _FORMATTERS.get(game_type, lambda c: c.get("name", "Unknown"))
    lines = []
    for sheet in sheets:
        content = sheet.get("content", {})
        marker = "★ " if sheet.get("is_protagonist", False) else "  "
        lines.append(marker + formatter(content))

    return "## Party\n" + "\n".join(lines)
