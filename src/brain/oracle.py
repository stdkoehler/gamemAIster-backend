from dataclasses import dataclass
import json
import random
from pathlib import Path


@dataclass
class Candidate:
    """Represents a selectable candidate with a weight."""

    name: str
    probability: float


class BaseOracle:
    """
    Generic Oracle engine for weighted random narrative seeds.
    Supply a config dict mapping pool names to lists of {name, probability} entries.
    """

    def __init__(self, config: dict):
        self.pools = {
            pool_name: [Candidate(**entry) for entry in entries]
            for pool_name, entries in config.items()
        }

    @staticmethod
    def weighted_choice(items: list[Candidate]) -> str:
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    def roll(self) -> dict:
        return {
            pool_name: self.weighted_choice(candidates)
            for pool_name, candidates in self.pools.items()
        }


class ShadowrunOracle(BaseOracle):
    """
    Oracle for Shadowrun missions, using external JSON definitions.
    Generates { client, mission, target } with target != client.
    """

    def __init__(self) -> None:
        json_path = Path(__file__).parent / "oracle" / "shadowrun.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # rename mission_types key to mission for consistency
        config = {
            "client": data["clients"],
            "mission": data["mission_types"],
            # keep clients for target selection
            "targets": data["clients"],
        }
        super().__init__(config)

    def mission(self, background: str) -> str:
        client = self.weighted_choice(self.pools["client"])
        mission = self.weighted_choice(self.pools["mission"])
        # select target different from client
        eligible = [c for c in self.pools["targets"] if c.name != client]
        target = self.weighted_choice(eligible)
        return json.dumps(
            {
                "client": client,
                "target": target,
                "mission": mission,
                "background": background,
            },
            ensure_ascii=False,
            indent=2,
        )


class VampireOracle(BaseOracle):
    def __init__(self) -> None:
        json_path = Path(__file__).parent / "oracle" / "vampire_the_masquerade.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = {
            "factions": data["factions"],
            "incitingIncidents": data["inciting_incidents"],
            "themes": data["themes"],
        }
        super().__init__(config)

    def mission(self, background: str) -> str:
        seed = self.roll()
        # pick two unique factions
        seed["factions"] = random.sample([c.name for c in self.pools["factions"]], k=2)
        seed["background"] = background
        # wrap inciting incident and themes in lists as needed
        return json.dumps(seed, ensure_ascii=False)


class CthulhuOracle:
    """
    Simplified Cthulhu Oracle that uses short, generic elements and combines them
    with minimal logic to create varied and consistent narrative seeds.
    Includes weighted probability for each element.
    """

    def __init__(self) -> None:
        json_path = Path(__file__).parent / "oracle" / "call_of_cthulhu.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Convert to candidates with probability weights
        self.location_types = [Candidate(**item) for item in data["location_types"]]
        self.location_modifiers = [
            Candidate(**item) for item in data["location_modifiers"]
        ]
        self.location_places = [Candidate(**item) for item in data["location_places"]]
        self.mythos_entities = [Candidate(**item) for item in data["mythos_entities"]]
        self.mythos_phenomena = [Candidate(**item) for item in data["mythos_phenomena"]]
        self.hook_subjects = [Candidate(**item) for item in data["hook_subjects"]]
        self.hook_events = [Candidate(**item) for item in data["hook_events"]]

    def weighted_choice(self, items: list[Candidate]) -> str:
        """Select an item based on its probability weight."""
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    def generate_location(self) -> str:
        """Generate a location by combining modifiers and types with probability weights."""
        # 70% chance to include a modifier
        if random.random() < 0.7:
            modifier = self.weighted_choice(self.location_modifiers)
            location_type = self.weighted_choice(self.location_types)

            # 50% chance to add a place
            if random.random() < 0.5:
                place = self.weighted_choice(self.location_places)
                return f"{modifier} {location_type} in {place}"
            else:
                return f"{modifier} {location_type}"
        else:
            location_type = self.weighted_choice(self.location_types)
            place = self.weighted_choice(self.location_places)
            return f"{location_type} in {place}"

    def generate_mythos_elements(self) -> list[str]:
        """Generate 2-3 mythos elements combining entities and phenomena, respecting probabilities."""
        num_elements = random.randint(2, 3)
        elements = []

        # Always include at least one entity and one phenomenon
        elements.append(self.weighted_choice(self.mythos_entities))
        elements.append(self.weighted_choice(self.mythos_phenomena))

        # Add one more random element if needed
        if num_elements > 2:
            # Choose from either category
            if random.random() < 0.5:
                additional = self.weighted_choice(self.mythos_entities)
            else:
                additional = self.weighted_choice(self.mythos_phenomena)

            # Avoid duplicates
            if additional not in elements:
                elements.append(additional)

        return elements

    def generate_hook(self) -> str:
        """Generate a hook by combining a subject and an event with probability weights."""
        subject = self.weighted_choice(self.hook_subjects)
        event = self.weighted_choice(self.hook_events)

        # Different hook formats
        formats = [
            f"A {subject}'s mysterious {event}",
            f"The {event} of a {subject}",
            f"A strange {event} involving a {subject}",
            f"A {subject} requests help with a {event}",
            # New, simpler CoC Hook Formats:
            f"Rumors of a {subject} and an {event}",
            f"An investigation into a {subject}'s {event}",
            f"The curious {event} affecting a {subject}",
            f"A {subject} is linked to an unusual {event}",
            f"Concern over a {subject} following an {event}",
            f"The unexplained {event} and its connection to a {subject}",
            f"A report about a {subject} and a recent {event}",
            f"The peculiar case of a {subject} and the {event}",
            f"Seeking answers about a {subject} after an {event}",
            f"A {subject} witnesses a disturbing {event}",
        ]

        return random.choice(formats)

    def mission(self, background: str) -> str:
        """Generate a complete mission seed."""
        seed = {
            "location": self.generate_location(),
            "mythosElements": self.generate_mythos_elements(),
            "hook": self.generate_hook(),
            "background": background,
        }

        return json.dumps(seed, ensure_ascii=False, indent=2)


# Example usage
def main() -> None:
    # Shadowrun
    sr = ShadowrunOracle()
    print("Shadowrun Seed:", sr.mission(""))
    # Vampire
    vt = VampireOracle()
    print("VtM Seed:", vt.mission(""))
    # Cthulhu
    ct = CthulhuOracle()
    print("Cthulhu Seed:", ct.mission(""))

    ct = CthulhuOracle()
    print("Cthulhu Seed:", ct.mission(""))


if __name__ == "__main__":
    main()
