from dataclasses import dataclass
import json
import random
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

from src.llmclient.llm_client import LLMClientBase


@dataclass
class Candidate:
    """Represents a selectable candidate with a weight."""

    name: str
    probability: float


class BaseOracle(ABC):
    """
    Generic Oracle engine for weighted random narrative seeds.
    Supply a config dict mapping pool names to lists of {name, probability} entries.
    """

    @property
    @abstractmethod
    def _alignment_prompt(self) -> str:
        pass

    def __init__(self, config: Dict[str, Any], llm_client: LLMClientBase):
        self._pools = {
            pool_name: [Candidate(**entry) for entry in entries]
            for pool_name, entries in config.items()
        }
        self._llm_client = llm_client

    @staticmethod
    def _weighted_choice(items: list[Candidate]) -> str:
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    def _roll(self) -> Dict[str, str | list[str]]:
        return {
            pool_name: self._weighted_choice(candidates)
            for pool_name, candidates in self._pools.items()
        }

    def _align(
        self, candidate: dict[str, str | list[str]], background: str
    ) -> dict[str, str | list[str]]:

        candidate["background"] = background
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._alignment_prompt,
            },
            {
                "role": "user",
                "content": json.dumps(candidate),
            },
        ]
        # Optionally, you can capture the LLM response here if needed
        # response = self._llm_client.chat_completion(messages=messages, reasoning=True)
        # return response
        response = self._llm_client.chat_completion(messages=messages, reasoning=True)
        print("### LLM Alignment")
        print(messages)
        print(response)
        return candidate


class ShadowrunOracle(BaseOracle):
    """
    Oracle for Shadowrun missions, using external JSON definitions.
    Generates { client, mission, target } with target != client.
    """

    @property
    def _alignment_prompt(self) -> str:
        return self.__alignment_prompt

    def __init__(self, llm_client: LLMClientBase) -> None:
        json_path = Path(__file__).parent / "oracle" / "shadowrun.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = {
            "client": data["clients"],
            "mission": data["mission_types"],
            # keep clients for target selection
            "targets": data["clients"],
        }

        prompt_path = (
            Path(__file__).parent
            / "prompt_templates/shadowrun"
            / "shadowrun_background_mission_aligner.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.__alignment_prompt = f.read()

        super().__init__(config=config, llm_client=llm_client)

    def mission(self, background: str) -> str:
        client = self._weighted_choice(self._pools["client"])
        mission = self._weighted_choice(self._pools["mission"])
        # select target different from client
        eligible = [c for c in self._pools["targets"] if c.name != client]
        target = self._weighted_choice(eligible)

        candidate: dict[str, str | list[str]] = {
            "client": client,
            "mission": mission,
            "target": target,
        }

        aligned = self._align(candidate, background)
        aligned["background"] = background

        return json.dumps(
            aligned,
            ensure_ascii=False,
            indent=2,
        )


class VampireOracle(BaseOracle):

    @property
    def _alignment_prompt(self) -> str:
        return self.__alignment_prompt

    def __init__(self, llm_client: LLMClientBase) -> None:
        json_path = Path(__file__).parent / "oracle" / "vampire_the_masquerade.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = {
            "factions": data["factions"],
            "incitingIncidents": data["inciting_incidents"],
            "themes": data["themes"],
        }

        prompt_path = (
            Path(__file__).parent
            / "prompt_templates/vampire"
            / "vampire_background_mission_aligner.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.__alignment_prompt = f.read()

        super().__init__(config=config, llm_client=llm_client)

    def mission(self, background: str) -> str:
        candidate = self._roll()
        # pick one or two unique factions
        k = random.randint(1, 2)
        candidate["factions"] = random.sample(
            [c.name for c in self._pools["factions"]], k=k
        )

        aligned = self._align(candidate, background)
        aligned["background"] = background

        # wrap inciting incident and themes in lists as needed
        return json.dumps(aligned, ensure_ascii=False, indent=2)


class CthulhuOracle(BaseOracle):
    """
    Cthulhu Oracle using BaseOracle for weighted random selection.
    Generates narrative seeds with location, mythos elements, and hooks.
    """

    @property
    def _alignment_prompt(self) -> str:
        return self.__alignment_prompt

    def __init__(self, llm_client: LLMClientBase) -> None:
        json_path = Path(__file__).parent / "oracle" / "call_of_cthulhu.json"
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        config = {
            "location_types": data["location_types"],
            "location_modifiers": data["location_modifiers"],
            "location_places": data["location_places"],
            "mythos_entities": data["mythos_entities"],
            "mythos_phenomena": data["mythos_phenomena"],
            "hook_subjects": data["hook_subjects"],
            "hook_events": data["hook_events"],
        }
        prompt_path = (
            Path(__file__).parent
            / "prompt_templates/cthulhu"
            / "cthulhu_background_mission_aligner.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.__alignment_prompt = f.read()
        super().__init__(config=config, llm_client=llm_client)

    def generate_location(self) -> str:
        # 70% chance to include a modifier
        if random.random() < 0.7:
            modifier = self._weighted_choice(self._pools["location_modifiers"])
            location_type = self._weighted_choice(self._pools["location_types"])
            # 50% chance to add a place
            if random.random() < 0.5:
                place = self._weighted_choice(self._pools["location_places"])
                return f"{modifier} {location_type} in {place}"
            else:
                return f"{modifier} {location_type}"
        else:
            location_type = self._weighted_choice(self._pools["location_types"])
            place = self._weighted_choice(self._pools["location_places"])
            return f"{location_type} in {place}"

    def generate_mythos_elements(self) -> list[str]:
        num_elements = random.randint(2, 3)
        elements = []
        elements.append(self._weighted_choice(self._pools["mythos_entities"]))
        elements.append(self._weighted_choice(self._pools["mythos_phenomena"]))
        if num_elements > 2:
            if random.random() < 0.5:
                additional = self._weighted_choice(self._pools["mythos_entities"])
            else:
                additional = self._weighted_choice(self._pools["mythos_phenomena"])
            if additional not in elements:
                elements.append(additional)
        return elements

    def generate_hook(self) -> str:
        subject = self._weighted_choice(self._pools["hook_subjects"])
        event = self._weighted_choice(self._pools["hook_events"])
        formats = [
            f"A {subject}'s mysterious {event}",
            f"The {event} of a {subject}",
            f"A strange {event} involving a {subject}",
            f"A {subject} requests help with a {event}",
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
        candidate: dict[str, str | list[str]] = {
            "location": self.generate_location(),
            "mythosElements": self.generate_mythos_elements(),
            "hook": self.generate_hook(),
        }
        aligned = self._align(candidate, background)
        aligned["background"] = background
        return json.dumps(aligned, ensure_ascii=False, indent=2)


# Example usage
def main() -> None:
    # set pythonpath to src
    from src.llmclient.llm_client import LLMClientLocal

    llm_client_local = LLMClientLocal(base_url="http://127.0.0.1:5000")
    # Shadowrun
    sr = ShadowrunOracle(llm_client=llm_client_local)
    print(
        "Shadowrun Seed:",
        sr.mission(
            "Bayonie is a orc street samurai, living in a rugged appartment in the squatter of Stockholm. She's currently waiting for a call from her fixer Bert."
        ),
    )
    # Vampire
    vt = VampireOracle(llm_client=llm_client_local)
    print(
        "VtM Seed:",
        vt.mission(
            "It is 1885, Egypt. Khaled al'Sadid, a mortal, joins a German archaeological expedition led by the enthusiastic Dr. Schmidt. Due to his german skills he is the foreman of the local workforce."
        ),
    )
    # Cthulhu
    ct = CthulhuOracle(llm_client=llm_client_local)
    print(
        "Cthulhu Seed:",
        ct.mission(
            "Elias Ellinghouse is a antiquarian owning a small shop in Lafayette, Lousisiana. He has not yet had contact with any unnatural phenomenon, but is a dedicated collector of peculiar items."
        ),
    )


if __name__ == "__main__":
    main()
