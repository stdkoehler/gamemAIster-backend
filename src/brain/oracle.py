from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Generic, TypeVar

from pydantic import BaseModel, RootModel

from src.llmclient.llm_client import LLMClientBase
from src.llmclient.llm_config_registry import LLMTask
from src.utils.sqllogger import LogType

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


@dataclass
class OracleResult:
    roll: str     # JSON of the random proposal drawn from pools
    aligned: str  # JSON of the LLM-aligned result
    topic: str    # Final JSON sent to the mission LLM (aligned + background)


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


class FactionEraAligned(BaseModel):
    factions: list[str]
    incitingIncident: str
    themes: list[str]
    era: str


class VampireAligned(FactionEraAligned):
    pass


class SeventhSeaAligned(FactionEraAligned):
    pass


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
    culturalFoci: str


class CustomAligned(BaseModel):
    region: str
    characterRole: str
    startingSituation: str
    seasonalContext: str
    culturalFoci: str


class SlavicProposal(BaseModel):
    region: str
    characterRole: str
    startingSituation: str
    seasonalContext: str
    culturalFoci: str


class SlavicAligned(BaseModel):
    region: str
    characterRole: str
    startingSituation: str
    seasonalContext: str
    culturalFoci: str


class DragonlanceProposal(FactionBasedProposal):
    pass


class DragonlanceAligned(FactionEraAligned):
    pass


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

    @classmethod
    def _weighted_sample(cls, items: list[Candidate], k: int) -> list[str]:
        """Weighted sampling without replacement (random.sample ignores .probability)."""
        pool = list(items)
        chosen: list[str] = []
        for _ in range(min(k, len(pool))):
            name = cls._weighted_choice(pool)
            chosen.append(name)
            pool = [c for c in pool if c.name != name]
        return chosen

    def _align(self, proposal: TProposal, background: str) -> TAligned:
        proposal_dict = proposal.model_dump()
        proposal_dict["background"] = background
        user_prompt = json.dumps(proposal_dict, ensure_ascii=False)
        agent = self._llm_client.build_agent(
            LLMTask.ARCHITECT,
            output_type=self._aligned_type,
            system_prompt=self._alignment_prompt,
            reasoning=True,
        )
        return self._llm_client.run_agent(agent, self._alignment_prompt, user_prompt, LogType.ORACLE_ALIGN)

    def mission(self, background: str) -> OracleResult:
        proposal = self._assemble_proposal_seed()
        aligned = self._align(proposal, background)
        topic = {**aligned.model_dump(), "background": background}
        return OracleResult(
            roll=json.dumps(proposal.model_dump(), ensure_ascii=False, indent=2),
            aligned=json.dumps(aligned.model_dump(), ensure_ascii=False, indent=2),
            topic=json.dumps(topic, ensure_ascii=False, indent=2),
        )


# ---------------------------------------------------------------------------
# Intermediate base for faction-based games (Vampire, SeventhSea, Expanse)
# ---------------------------------------------------------------------------


class FactionBasedOracle(BaseOracle[FactionBasedProposal, TFactionAligned]):
    """
    Shared assembly logic for game systems whose oracle seed is
    factions + incitingIncident + themes.

    Subclasses set _proposal_type to their game-specific Proposal class so
    _assemble_proposal_seed returns the right concrete type without duplicating
    the pool-sampling logic in every oracle.
    """

    _proposal_type: ClassVar[type[FactionBasedProposal]] = FactionBasedProposal

    def _assemble_proposal_seed(self) -> FactionBasedProposal:
        k = random.randint(1, 2)
        factions = self._weighted_sample(self._pools["factions"], k)
        incident = self._weighted_choice(self._pools["inciting_incidents"])
        theme = self._weighted_choice(self._pools["themes"])
        return self._proposal_type(
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
    _proposal_type = VampireProposal

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=VampireAligned,
            config_filename="vampire_the_masquerade.json",
            prompt_filename="vampire/vampire_background_mission_aligner.txt",
        )


class SeventhSeaOracle(FactionBasedOracle[SeventhSeaAligned]):
    _proposal_type = SeventhSeaProposal

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=SeventhSeaAligned,
            config_filename="seventh_sea.json",
            prompt_filename="seventh_sea/seventh_sea_background_mission_aligner.txt",
        )


class ExpanseOracle(FactionBasedOracle[ExpanseAligned]):
    _proposal_type = ExpanseProposal

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

    @staticmethod
    def _indefinite_article(word: str) -> str:
        return "an" if word[0].lower() in "aeiou" else "a"

    def generate_hook(self) -> str:
        subject = self._weighted_choice(self._pools["hook_subjects"])
        event = self._weighted_choice(self._pools["hook_events"])
        subj_art = self._indefinite_article(subject)
        event_art = self._indefinite_article(event)
        formats = [
            f"{subj_art.capitalize()} {subject}'s mysterious {event}",
            f"The {event} of {subj_art} {subject}",
            f"A strange {event} involving {subj_art} {subject}",
            f"{subj_art.capitalize()} {subject} requests help with {event_art} {event}",
            f"Rumors of {subj_art} {subject} and {event_art} {event}",
            f"An investigation into {subj_art} {subject}'s {event}",
            f"The curious {event} affecting {subj_art} {subject}",
            f"{subj_art.capitalize()} {subject} is linked to an unusual {event}",
            f"Concern over {subj_art} {subject} following {event_art} {event}",
            f"The unexplained {event} and its connection to {subj_art} {subject}",
            f"A report about {subj_art} {subject} and a recent {event}",
            f"The peculiar case of {subj_art} {subject} and the {event}",
            f"Seeking answers about {subj_art} {subject} after {event_art} {event}",
            f"{subj_art.capitalize()} {subject} witnesses a disturbing {event}",
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
            culturalFoci=self._weighted_choice(self._pools["culturalFoci"]),
        )


class SlavicOracle(BaseOracle[SlavicProposal, SlavicAligned]):
    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=SlavicAligned,
            config_filename="slavic.json",
            prompt_filename="slavic/slavic_background_mission_aligner.txt",
        )

    def _assemble_proposal_seed(self) -> SlavicProposal:
        return SlavicProposal(
            region=self._weighted_choice(self._pools["regions"]),
            characterRole=self._weighted_choice(self._pools["characterRoles"]),
            startingSituation=self._weighted_choice(self._pools["startingSituations"]),
            seasonalContext=self._weighted_choice(self._pools["seasonalContexts"]),
            culturalFoci=self._weighted_choice(self._pools["culturalFoci"]),
        )


class DragonlanceOracle(FactionBasedOracle[DragonlanceAligned]):
    _proposal_type = DragonlanceProposal

    def __init__(self, llm_client: LLMClientBase) -> None:
        super().__init__(
            llm_client=llm_client,
            aligned_type=DragonlanceAligned,
            config_filename="dragonlance.json",
            prompt_filename="dragonlance/dragonlance_background_mission_aligner.txt",
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
    result = sr.mission(
        "I'm Doromir, a farmer in Starigard. I have a dispute about farmland with my neighbor and need to resolve it."
    )
    print("Roll:", result.roll)
    print("Aligned:", result.aligned)
    print("Topic:", result.topic)


if __name__ == "__main__":
    main()
