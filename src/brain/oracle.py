import json
import random
from typing import TypeAlias
from pathlib import Path
from abc import ABC, abstractmethod

from pydantic import BaseModel, RootModel

from src.llmclient.llm_client import LLMClientBase
from src.brain.json_tools import extract_json_schema

MissionSeed: TypeAlias = dict[str, str | list[str]]


class Candidate(BaseModel):
    name: str
    probability: float


class OracleConfig(RootModel[dict[str, list[Candidate]]]):
    """
    Pydantic RootModel for validating and parsing the entire oracle config JSON as a
    root-level dictionary.

    This allows us to accept any pool name (e.g., 'clients', 'factions', 'location_types'),
    each mapping to a list of Candidate objects, regardless of the specific field names
    in the JSON.

    Using RootModel is required in Pydantic v2+ for root-level (non-object) JSON
    structures, and provides type safety and validation for all pools in a generic way.
    """


class BaseOracle(ABC):
    """
    Generic Oracle engine for weighted random narrative seeds.
    Handles config/prompt loading, weighted selection, and mission alignment.
    Subclasses must implement _assemble_candidate.
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        config_filename: str,
        prompt_filename: str,
        config_dir: str = "oracle",
        prompt_dir: str = "prompt_templates",
    ):
        base_path = Path(__file__).parent
        # Load config
        json_path = base_path / config_dir / config_filename
        with open(json_path, "r", encoding="utf-8") as f:
            data = f.read()
        config = OracleConfig.model_validate_json(data)
        self._pools = config.root
        self._llm_client = llm_client

        prompt_path = base_path / prompt_dir / prompt_filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._alignment_prompt = f.read()

    @staticmethod
    def _weighted_choice(items: list[Candidate]) -> str:
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    def _roll(self) -> MissionSeed:
        return {
            pool_name: self._weighted_choice(candidates)
            for pool_name, candidates in self._pools.items()
        }

    def _align(self, proposal: MissionSeed, background: str) -> MissionSeed:
        proposal["background"] = background
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._alignment_prompt},
            {"role": "user", "content": json.dumps(proposal)},
        ]
        response = self._llm_client.chat_completion(messages=messages, reasoning=True)
        print("### LLM Alignment")
        print(proposal)
        print(response)
        try:
            adjusted: MissionSeed = json.loads(extract_json_schema(response))
        except json.decoder.JSONDecodeError as exc:
            raise ValueError(f"LLM response is not valid JSON: {response}") from exc

        return adjusted

    @abstractmethod
    def _assemble_proposal_seed(self) -> MissionSeed:
        pass

    def mission(self, background: str) -> str:
        candidate = self._assemble_proposal_seed()
        aligned = self._align(candidate, background)
        aligned["background"] = background
        return json.dumps(aligned, ensure_ascii=False, indent=2)


class ShadowrunOracle(BaseOracle):
    """
    Oracle for Shadowrun missions, using external JSON definitions.
    Generates { client, mission, target } with target != client.
    """

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            config_filename="shadowrun.json",
            prompt_filename="shadowrun/shadowrun_background_mission_aligner.txt",
        )
        # For target selection, keep clients pool as 'targets'
        self._pools["targets"] = self._pools["clients"]

    def _assemble_proposal_seed(self) -> MissionSeed:
        client = self._weighted_choice(self._pools["clients"])
        mission = self._weighted_choice(self._pools["mission_types"])
        eligible = [c for c in self._pools["targets"] if c.name != client]
        target = self._weighted_choice(eligible)
        return {
            "client": client,
            "mission": mission,
            "target": target,
        }


class VampireOracle(BaseOracle):
    """
    Oracle for Vampire: The Masquerade missions, using external JSON definitions.
    Generates { factions, incitingIncidents, themes } with 1-2 unique factions.
    """

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            config_filename="vampire_the_masquerade.json",
            prompt_filename="vampire/vampire_background_mission_aligner.txt",
        )

    def _assemble_proposal_seed(self) -> MissionSeed:
        candidate = self._roll()
        # pick one or two unique factions
        k = random.randint(1, 2)
        candidate["factions"] = random.sample(
            [c.name for c in self._pools["factions"]], k=k
        )
        return candidate


class CthulhuOracle(BaseOracle):
    """
    Cthulhu Oracle using BaseOracle for weighted random selection.
    Generates narrative seeds with location, mythos elements, and hooks.
    """

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            config_filename="call_of_cthulhu.json",
            prompt_filename="cthulhu/cthulhu_background_mission_aligner.txt",
        )

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

    def _assemble_proposal_seed(self) -> MissionSeed:
        return {
            "location": self.generate_location(),
            "mythosElements": self.generate_mythos_elements(),
            "hook": self.generate_hook(),
        }


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
