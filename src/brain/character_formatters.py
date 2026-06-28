"""Per-system formatters that turn character/NPC sheet content into
labeled, LLM-ready JSON.

One ``SystemFormatter`` subclass per game system, with two entry points:

- ``format_party``: lean — the player already knows their own skills, gear,
  and special abilities, so this is limited to identity and the dynamic
  state (attributes, resource tracks) the GM needs to narrate around.
- ``format_npc``: rich — the GM plays the NPC, so this also includes
  skills, equipment, and special abilities (disciplines, spellcasting,
  talents, ...) needed to adjudicate and roleplay them.

Fields that are structurally not applicable (e.g. a mortal vampire's clan/
generation, a non-caster's spellcasting block) are omitted rather than sent
as null/zero/empty, and fields that are constants of the NPC-generation
pipeline rather than real per-character data (e.g. a generated NPC's hero
points, always 3) are likewise left out — both would otherwise read as
real signal and aren't.

Adding a new system means adding one class here and one registry entry —
nothing else to keep in sync.
"""

import json
from abc import ABC, abstractmethod


def _track(t: dict) -> str:
    return f"{t.get('current', 0)}/{t.get('max', 0)}"


def _describe(c: dict) -> str:
    desc = c.get("description", "")
    return desc[:140] + "…" if len(desc) > 140 else desc


def _nonzero(d: dict) -> dict:
    return {k: v for k, v in d.items() if v}


class SystemFormatter(ABC):
    @abstractmethod
    def format_party(self, c: dict) -> dict: ...

    @abstractmethod
    def format_npc(self, c: dict) -> dict: ...


class ShadowrunFormatter(SystemFormatter):
    def _attributes(self, c: dict) -> dict:
        a = c.get("attributes", {})
        attrs = {
            "Body": a.get("Body", 0),
            "Agility": a.get("Agility", 0),
            "Reaction": a.get("Reaction", 0),
            "Strength": a.get("Strength", 0),
            "Willpower": a.get("Willpower", 0),
            "Logic": a.get("Logic", 0),
            "Intuition": a.get("Intuition", 0),
            "Charisma": a.get("Charisma", 0),
            "Edge": a.get("Edge", 0),
            "Essence": a.get("Essence", 6),
        }
        if a.get("Magic"):
            attrs["Magic"] = a["Magic"]
        if a.get("Resonance"):
            attrs["Resonance"] = a["Resonance"]
        return attrs

    def _damage(self, c: dict) -> dict:
        d = c.get("damage", {})
        return {
            "physical": _track(d.get("physical", {})),
            "stun": _track(d.get("stun", {})),
        }

    def format_party(self, c: dict) -> dict:
        return {
            "name": c.get("name", "?"),
            "metatype": c.get("metatype", ""),
            "archetype": c.get("archetype", ""),
            "attributes": self._attributes(c),
            "damage": self._damage(c),
        }

    def format_npc(self, c: dict) -> dict:
        d = c.get("derived", {})
        entry = {
            "name": c.get("name", "?"),
            "metatype": c.get("metatype", ""),
            "archetype": c.get("archetype", ""),
            "description": _describe(c),
            "attributes": self._attributes(c),
            "skills": _nonzero(c.get("skills", {})),
            "damage": self._damage(c),
        }
        if d.get("initiativeBase") or d.get("initiativeDice"):
            entry["initiative"] = f"{d.get('initiativeBase', 0)}+{d.get('initiativeDice', 0)}d6"
        if c.get("armor"):
            entry["armor"] = c["armor"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("cyberware"):
            entry["cyberware"] = c["cyberware"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


class VampireFormatter(SystemFormatter):
    def _identity(self, c: dict) -> dict:
        nature = c.get("nature", "mortal")
        identity = {"name": c.get("name", "?"), "nature": nature}
        if nature != "mortal":
            if c.get("clan"):
                identity["clan"] = c["clan"]
            if c.get("generation"):
                identity["generation"] = c["generation"]
            if c.get("predatorType"):
                identity["predatorType"] = c["predatorType"]
        return identity

    def _attributes(self, c: dict) -> dict:
        a = c.get("attributes", {})
        return {
            k: a.get(k, 0)
            for k in (
                "Strength",
                "Dexterity",
                "Stamina",
                "Charisma",
                "Manipulation",
                "Intelligence",
                "Wits",
            )
        }

    def _resources(self, c: dict) -> dict:
        nature = c.get("nature", "mortal")
        res = {
            "health": _track(c.get("health", {})),
            "willpower": _track(c.get("willpower", {})),
        }
        if nature in ("kindred", "thin-blood"):
            res["hunger"] = f"{c.get('hunger', 0)}/5"
        if nature == "kindred":
            res["humanity"] = f"{c.get('humanity', 0)}/10"
            res["bloodPotency"] = c.get("bloodPotency", 0)
        return res

    def format_party(self, c: dict) -> dict:
        return {**self._identity(c), "attributes": self._attributes(c), **self._resources(c)}

    def format_npc(self, c: dict) -> dict:
        nature = c.get("nature", "mortal")
        entry = {
            **self._identity(c),
            "description": _describe(c),
            "attributes": self._attributes(c),
            "skills": _nonzero(c.get("skills", {})),
            **self._resources(c),
        }
        if nature in ("kindred", "ghoul") and c.get("disciplines"):
            entry["disciplines"] = c["disciplines"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        return entry


class CthulhuFormatter(SystemFormatter):
    def _characteristics(self, c: dict) -> dict:
        ch = c.get("characteristics", {})
        return {k: ch.get(k, 0) for k in ("STR", "CON", "DEX", "INT", "POW", "EDU")}

    def format_party(self, c: dict) -> dict:
        return {
            "name": c.get("name", "?"),
            "occupation": c.get("occupation", ""),
            "characteristics": self._characteristics(c),
            "hitPoints": _track(c.get("hitPoints", {})),
            "sanity": _track(c.get("sanity", {})),
            "magicPoints": _track(c.get("magicPoints", {})),
            "luck": c.get("luck", 0),
        }

    def format_npc(self, c: dict) -> dict:
        d = c.get("derived", {})
        entry = {
            "name": c.get("name", "?"),
            "occupation": c.get("occupation", ""),
            "description": _describe(c),
            "characteristics": self._characteristics(c),
            "skills": _nonzero(c.get("skills", {})),
            "hitPoints": _track(c.get("hitPoints", {})),
            "sanity": _track(c.get("sanity", {})),
        }
        if d.get("damageBonus"):
            entry["damageBonus"] = d["damageBonus"]
        if d.get("moveRate"):
            entry["moveRate"] = d["moveRate"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


class SeventhSeaFormatter(SystemFormatter):
    def _traits(self, c: dict) -> dict:
        t = c.get("traits", {})
        return {k: t.get(k, 0) for k in ("Brawn", "Finesse", "Resolve", "Wits", "Panache")}

    def _arcana(self, c: dict) -> dict | None:
        arcana = c.get("arcana", {})
        return arcana if arcana.get("virtue") or arcana.get("hubris") else None

    def format_party(self, c: dict) -> dict:
        entry = {
            "name": c.get("name", "?"),
            "nation": c.get("nation", ""),
            "traits": self._traits(c),
            "wounds": _track(c.get("wounds", {})),
            "heroPoints": c.get("heroPoints", 0),
        }
        arcana = self._arcana(c)
        if arcana:
            entry["arcana"] = arcana
        return entry

    def format_npc(self, c: dict) -> dict:
        entry = {
            "name": c.get("name", "?"),
            "nation": c.get("nation", ""),
            "description": _describe(c),
            "traits": self._traits(c),
            "skills": _nonzero(c.get("skills", {})),
            "wounds": _track(c.get("wounds", {})),
        }
        arcana = self._arcana(c)
        if arcana:
            entry["arcana"] = arcana
        if c.get("duelingStyle"):
            entry["duelingStyle"] = c["duelingStyle"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


class ExpanseFormatter(SystemFormatter):
    def _abilities(self, c: dict) -> dict:
        a = c.get("abilities", {})
        return {
            k: a.get(k, 0)
            for k in (
                "Accuracy",
                "Communication",
                "Constitution",
                "Dexterity",
                "Fighting",
                "Intelligence",
                "Perception",
                "Strength",
                "Willpower",
            )
        }

    def format_party(self, c: dict) -> dict:
        return {
            "name": c.get("name", "?"),
            "origin": c.get("origin", ""),
            "background": c.get("background", ""),
            "abilities": self._abilities(c),
            "health": _track(c.get("health", {})),
            "fortune": c.get("fortune", 0),
        }

    def format_npc(self, c: dict) -> dict:
        entry = {
            "name": c.get("name", "?"),
            "faction": c.get("faction", ""),
            "description": _describe(c),
            "abilities": self._abilities(c),
            "health": _track(c.get("health", {})),
        }
        if c.get("focuses"):
            entry["focuses"] = c["focuses"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("armor"):
            entry["armor"] = c["armor"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


class SlavicFormatter(SystemFormatter):
    def _attributes(self, c: dict) -> dict:
        a = c.get("attributes", {})
        d = c.get("attributeDamage", {})
        return {
            k: f"{d.get(k, a.get(k, 0))}/{a.get(k, 0)}"
            for k in ("Strength", "Agility", "Wits", "Empathy")
        }

    def format_party(self, c: dict) -> dict:
        return {
            "name": c.get("name", "?"),
            "kin": c.get("kin", ""),
            "kinAbility": c.get("kinAbility", ""),
            "calling": c.get("calling", ""),
            "attributes": self._attributes(c),
        }

    def format_npc(self, c: dict) -> dict:
        entry = {
            "name": c.get("name", "?"),
            "kin": c.get("kin", ""),
            "kinAbility": c.get("kinAbility", ""),
            "calling": c.get("calling", ""),
            "description": _describe(c),
            "attributes": self._attributes(c),
            "skills": _nonzero(c.get("skills", {})),
        }
        if c.get("talents"):
            entry["talents"] = c["talents"]
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("armor"):
            entry["armor"] = c["armor"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


class DragonlanceFormatter(SystemFormatter):
    def _abilities(self, c: dict) -> dict:
        a = c.get("abilities", {})
        return {
            k: a.get(k, 10)
            for k in ("Strength", "Dexterity", "Constitution", "Intelligence", "Wisdom", "Charisma")
        }

    def format_party(self, c: dict) -> dict:
        return {
            "name": c.get("name", "?"),
            "race": c.get("race", ""),
            "characterClass": c.get("characterClass", ""),
            "abilities": self._abilities(c),
            "hitPoints": _track(c.get("hitPoints", {})),
            "armorClass": c.get("armorClass", 0),
        }

    def format_npc(self, c: dict) -> dict:
        entry = {
            "name": c.get("name", "?"),
            "race": c.get("race", ""),
            "characterClass": c.get("characterClass", ""),
            "description": _describe(c),
            "abilities": self._abilities(c),
            "skills": _nonzero(c.get("skills", {})),
            "hitPoints": _track(c.get("hitPoints", {})),
            "armorClass": c.get("armorClass", 0),
        }
        spellcasting = c.get("spellcasting", {})
        if spellcasting.get("ability"):
            entry["spellcasting"] = {
                "ability": spellcasting["ability"],
                "saveDc": spellcasting.get("saveDc"),
                "knownSpells": spellcasting.get("knownSpells", []),
            }
        if c.get("weapons"):
            entry["weapons"] = c["weapons"]
        if c.get("armorName"):
            entry["armorName"] = c["armorName"]
        if c.get("gear"):
            entry["gear"] = c["gear"]
        return entry


_FORMATTERS: dict[str, SystemFormatter] = {
    "shadowrun": ShadowrunFormatter(),
    "vampire_the_masquerade": VampireFormatter(),
    "call_of_cthulhu": CthulhuFormatter(),
    "seventh_sea": SeventhSeaFormatter(),
    "expanse": ExpanseFormatter(),
    "slavic": SlavicFormatter(),
    "dragonlance": DragonlanceFormatter(),
}


def _fallback(c: dict) -> dict:
    return {"name": c.get("name", "Unknown")}


def to_party_summary(game_type: str, sheets: list[dict]) -> str:
    """
    Return a JSON array (as a string) describing the player party's state.
    sheets: list of dicts with keys {content, is_protagonist}.
    """
    formatter = _FORMATTERS.get(game_type)
    entries = []
    for sheet in sheets:
        content = sheet.get("content", {})
        data = formatter.format_party(content) if formatter else _fallback(content)
        entries.append({**data, "protagonist": sheet.get("is_protagonist", False)})

    return json.dumps(entries, indent=2)


def to_npc_summary(game_type: str, sheets: list[dict]) -> str:
    """
    Return a JSON array (as a string) describing the NPCs currently active
    in the scene.
    sheets: list of dicts with key {content}.
    """
    formatter = _FORMATTERS.get(game_type)
    entries = [
        formatter.format_npc(sheet.get("content", {}))
        if formatter
        else _fallback(sheet.get("content", {}))
        for sheet in sheets
    ]

    return json.dumps(entries, indent=2)
