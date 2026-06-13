from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel, RootModel

from src.brain.structured_output import parse_with_retry
from src.llmclient.llm_client import LLMClientBase, Message, MessageContent, MessageRole
from src.llmclient.llm_config_registry import LLMTask

# ---------------------------------------------------------------------------
# Oracle pool config
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    name: str
    probability: float


class OracleConfig(RootModel[dict[str, list[Candidate]]]):
    """
    Root-level dict so we can accept any pool name (clients, factions, …)
    each mapping to a list of Candidate objects.
    """


# ---------------------------------------------------------------------------
# Per-game typed seed models  (Proposal = sent to aligner, Aligned = returned)
# ---------------------------------------------------------------------------


class ShadowrunProposal(BaseModel):
    client: str
    target: str
    mission: str


class ShadowrunAligned(BaseModel):
    client: str
    target: str
    mission: str
    location: str


# Vampire, SeventhSea, and Expanse (hero) all share the same proposal shape.
class FactionBasedProposal(BaseModel):
    factions: list[str]
    incitingIncident: str
    themes: list[str]


class VampireProposal(FactionBasedProposal):
    pass


class SeventhSeaProposal(FactionBasedProposal):
    pass


class ExpanseProposal(FactionBasedProposal):
    pass


class VampireAligned(BaseModel):
    factions: list[str]
    incitingIncident: str
    themes: list[str]
    epoch: str


class SeventhSeaAligned(BaseModel):
    factions: list[str]
    incitingIncident: str
    themes: list[str]
    epoch: str


class ExpanseAligned(BaseModel):
    factions: list[str]
    incitingIncident: str
    themes: list[str]
    era: str


class ExpanseNonHeroProposal(BaseModel):
    affectedInterests: list[str]
    complicatingSituation: str
    dailyConcerns: list[str]


class ExpanseNonHeroAligned(BaseModel):
    affectedInterests: list[str]
    complicatingSituation: str
    dailyConcerns: list[str]
    era: str


class CthulhuProposal(BaseModel):
    location: str
    mythosElements: list[str]
    hook: str


class CthulhuAligned(BaseModel):
    location: str
    # aligner prompt consolidates the list into a single descriptive string
    mythosElement: str
    hook: str
    era: str


class CustomProposal(BaseModel):
    region: str
    characterRole: str
    startingSituation: str
    seasonalContext: str


class CustomAligned(BaseModel):
    region: str
    characterRole: str
    startingSituation: str
    seasonalContext: str


# ---------------------------------------------------------------------------
# Generic base
# ---------------------------------------------------------------------------

TProposal = TypeVar("TProposal", bound=BaseModel)
TAligned = TypeVar("TAligned", bound=BaseModel)
TFactionAligned = TypeVar("TFactionAligned", bound=BaseModel)


class BaseOracle(ABC, Generic[TProposal, TAligned]):
    """
    Generic Oracle engine for weighted random narrative seeds.

    Subclasses must implement _assemble_proposal_seed and pass aligned_type
    to super().__init__() — that's the full contract.
    """

    def __init__(
        self,
        llm_client: LLMClientBase,
        aligned_type: type[TAligned],
        config_filename: str,
        prompt_filename: str,
        config_dir: str = "oracle",
        prompt_dir: str = "prompt_templates",
    ):
        base_path = Path(__file__).parent
        json_path = base_path / config_dir / config_filename
        with open(json_path, "r", encoding="utf-8") as f:
            data = f.read()
        config = OracleConfig.model_validate_json(data)
        self._pools = config.root
        self._llm_client = llm_client
        self._aligned_type: type[TAligned] = aligned_type

        prompt_path = base_path / prompt_dir / prompt_filename
        with open(prompt_path, "r", encoding="utf-8") as f:
            self._alignment_prompt = f.read()

    @abstractmethod
    def _assemble_proposal_seed(self) -> TProposal:
        """Build the pre-alignment seed from the oracle pools."""

    @staticmethod
    def _weighted_choice(items: list[Candidate]) -> str:
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    def _align(self, proposal: TProposal, background: str) -> TAligned:
        proposal_dict = proposal.model_dump()
        proposal_dict["background"] = background
        messages: list[Message] = [
            Message(
                role=MessageRole.SYSTEM,
                content=MessageContent(text=self._alignment_prompt),
            ),
            Message(
                role=MessageRole.USER,
                content=MessageContent(
                    text=json.dumps(proposal_dict, ensure_ascii=False)
                ),
            ),
        ]
        return parse_with_retry(
            messages=messages,
            result_type=self._aligned_type,
            llm_client=self._llm_client,
            reasoning=True,
            task=LLMTask.ARCHITECT,
        )

    def mission(self, background: str) -> str:
        proposal = self._assemble_proposal_seed()
        aligned = self._align(proposal, background)
        result = aligned.model_dump()
        result["background"] = background
        print("### Oracle Alignment")
        print(result)
        return json.dumps(result, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Intermediate base for faction-based games (Vampire, SeventhSea, Expanse)
# ---------------------------------------------------------------------------


class FactionBasedOracle(BaseOracle[FactionBasedProposal, TFactionAligned]):
    """
    Shared assembly logic for game systems whose oracle seed is
    factions + incitingIncident + themes.
    """

    def _assemble_proposal_seed(self) -> FactionBasedProposal:
        k = random.randint(1, 2)
        factions = random.sample([c.name for c in self._pools["factions"]], k=k)
        incident = self._weighted_choice(self._pools["inciting_incidents"])
        theme = self._weighted_choice(self._pools["themes"])
        return FactionBasedProposal(
            factions=factions,
            incitingIncident=incident,
            themes=[theme],
        )


# ---------------------------------------------------------------------------
# Concrete oracles
# ---------------------------------------------------------------------------


class ShadowrunOracle(BaseOracle[ShadowrunProposal, ShadowrunAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=ShadowrunAligned,
            config_filename="shadowrun.json",
            prompt_filename="shadowrun/shadowrun_background_mission_aligner.txt",
        )
        self._pools["targets"] = self._pools["clients"]

    def _assemble_proposal_seed(self) -> ShadowrunProposal:
        client = self._weighted_choice(self._pools["clients"])
        mission = self._weighted_choice(self._pools["mission_types"])
        eligible = [c for c in self._pools["targets"] if c.name != client]
        target = self._weighted_choice(eligible)
        return ShadowrunProposal(client=client, mission=mission, target=target)


class VampireOracle(FactionBasedOracle[VampireAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=VampireAligned,
            config_filename="vampire_the_masquerade.json",
            prompt_filename="vampire/vampire_background_mission_aligner.txt",
        )


class SeventhSeaOracle(FactionBasedOracle[SeventhSeaAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=SeventhSeaAligned,
            config_filename="seventh_sea.json",
            prompt_filename="seventh_sea/seventh_sea_background_mission_aligner.txt",
        )


class ExpanseOracle(FactionBasedOracle[ExpanseAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=ExpanseAligned,
            config_filename="expanse.json",
            prompt_filename="expanse/expanse_background_mission_aligner.txt",
        )


class ExpanseNonHeroOracle(BaseOracle[ExpanseNonHeroProposal, ExpanseNonHeroAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=ExpanseNonHeroAligned,
            config_filename="expanse_non_hero.json",
            prompt_filename="expanse/expanse_background_mission_aligner_non_hero.txt",
        )

    def _assemble_proposal_seed(self) -> ExpanseNonHeroProposal:
        interests = self._weighted_choice(self._pools["affectedInterests"])
        situation = self._weighted_choice(self._pools["complicatingSituations"])
        concern = self._weighted_choice(self._pools["dailyConcerns"])
        return ExpanseNonHeroProposal(
            affectedInterests=[interests],
            complicatingSituation=situation,
            dailyConcerns=[concern],
        )


class CthulhuOracle(BaseOracle[CthulhuProposal, CthulhuAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=CthulhuAligned,
            config_filename="call_of_cthulhu.json",
            prompt_filename="cthulhu/cthulhu_background_mission_aligner.txt",
        )

    def generate_location(self) -> str:
        if random.random() < 0.7:
            modifier = self._weighted_choice(self._pools["location_modifiers"])
            location_type = self._weighted_choice(self._pools["location_types"])
            if random.random() < 0.5:
                place = self._weighted_choice(self._pools["location_places"])
                return f"{modifier} {location_type} in {place}"
            return f"{modifier} {location_type}"
        location_type = self._weighted_choice(self._pools["location_types"])
        place = self._weighted_choice(self._pools["location_places"])
        return f"{location_type} in {place}"

    def generate_mythos_elements(self) -> list[str]:
        num_elements = random.randint(2, 3)
        elements = [
            self._weighted_choice(self._pools["mythos_entities"]),
            self._weighted_choice(self._pools["mythos_phenomena"]),
        ]
        if num_elements > 2:
            pool = "mythos_entities" if random.random() < 0.5 else "mythos_phenomena"
            additional = self._weighted_choice(self._pools[pool])
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

    def _assemble_proposal_seed(self) -> CthulhuProposal:
        return CthulhuProposal(
            location=self.generate_location(),
            mythosElements=self.generate_mythos_elements(),
            hook=self.generate_hook(),
        )


class CustomOracle(BaseOracle[CustomProposal, CustomAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=CustomAligned,
            config_filename="custom.json",
            prompt_filename="custom/custom_background_mission_aligner.txt",
        )

    def _assemble_proposal_seed(self) -> CustomProposal:
        return CustomProposal(
            region=self._weighted_choice(self._pools["regions"]),
            characterRole=self._weighted_choice(self._pools["characterRoles"]),
            startingSituation=self._weighted_choice(self._pools["startingSituations"]),
            seasonalContext=self._weighted_choice(self._pools["seasonalContexts"]),
        )


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def main() -> None:
    # set pythonpath to src
    from src.llmclient.llm_client import LLMClientLocal

    llm_client_local = LLMClientLocal(base_url="http://127.0.0.1:5000")
    # # Shadowrun
    # sr = ShadowrunOracle(llm_client=llm_client_local)
    # print(
    #     "Shadowrun Seed:",
    #     sr.mission(
    #         "Bayonie is a orc street samurai, living in a rugged appartment in the squatter of Stockholm. She's currently waiting for a call from her fixer Bert."
    #     ),
    # )
    # # Vampire
    # vt = VampireOracle(llm_client=llm_client_local)
    # print(
    #     "VtM Seed:",
    #     vt.mission(
    #         "It is 1885, Egypt. Khaled al'Sadid, a mortal, joins a German archaeological expedition led by the enthusiastic Dr. Schmidt. Due to his german skills he is the foreman of the local workforce."
    #     ),
    # )
    # # Seventh Sea
    # vt = SeventhSeaOracle(llm_client=llm_client_local)
    # print(
    #     "Seventh Sea Seed:",
    #     vt.mission(
    #         "It is 1668 somewhere in the Caribbean. I am Adjoua Mbah a slave who went overboard and is stranded on a small island near an archipelago. I need to find a way to escape this island."
    #     ),
    # )
    # # Expanse
    # sr = ExpanseOracle(llm_client=llm_client_local)
    # print(
    #     "Expanse Seed:",
    #     sr.mission(
    #         "Luna City, Laconia Era. The crew operates a small freight hauler called the Meridian Runner, struggling to make ends meet under the strict regulations of the Laconian Empire. After the Ring Gates reopened, they've been running legitimate cargo between Sol system stations, but their mixed crew of former Belters and Inner Planet refugees has made them targets of suspicion from Laconian authorities who view any non-Imperial crew as potential insurgents."
    #     ),
    # )
    # sr = ExpanseNonHeroOracle(llm_client=llm_client_local)
    # print(
    #     "Expanse Seed:",
    #     sr.mission(
    #         "I'm David Lahoola, a belter on an ice trawler in the Belt. It's pre-canterbury era and we're scraping by, but the crew is tight-knit. We just started our return leg to Ceres after a long haul."
    #     ),
    # )
    # # Cthulhu
    # ct = CthulhuOracle(llm_client=llm_client_local)
    # print(
    #     "Cthulhu Seed:",
    #     ct.mission(
    #         "Elias Ellinghouse is a antiquarian owning a small shop in Lafayette, Lousisiana. He has not yet had contact with any unnatural phenomenon, but is a dedicated collector of peculiar items."
    #     ),
    # )
    sr = CustomOracle(llm_client=llm_client_local)
    print(
        "Custom Seed:",
        sr.mission(
            "I'm Doromir, a farmer in Starigard. I have a dispute about farmland with my neighbor and need to resolve it."
        ),
    )


if __name__ == "__main__":
    main()
