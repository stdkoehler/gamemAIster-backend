from dataclasses import dataclass
import json
import random


@dataclass
class Candidate:
    """Candidate"""

    name: str
    probability: float


clients = [
    Candidate(name="Underworld fixer", probability=1.0),
    Candidate(name="Renraku Corporation", probability=1.0),
    Candidate(name="Saeder-Krupp Industries", probability=1.0),
    Candidate(name="Aztechnology", probability=1.0),
    Candidate(name="Mitsuhama Computer Technologies", probability=1.0),
    Candidate(name="Ares Macrotechnology", probability=1.0),
    Candidate(name="Shiawase Corporation", probability=1.0),
    Candidate(name="Wuxing Incorporated", probability=1.0),
    Candidate(name="EVO Corporation", probability=1.0),
    Candidate(name="NeoNET", probability=1.0),
    Candidate(name="Horizon Group", probability=1.0),
    Candidate(name="Lone Star Security Services", probability=1.0),
    Candidate(name="Universal Brotherhood", probability=1.0),
    Candidate(name="Megacorporation executive", probability=0.8),
    Candidate(name="Crime lord", probability=0.8),
    Candidate(name="Reclusive billionaire", probability=0.8),
    Candidate(name="Hacker collective leader", probability=0.8),
    Candidate(name="Rogue AI programmer", probability=0.8),
    Candidate(name="Cross Applied Technologies", probability=0.8),
    Candidate(name="Fuchi Industrial Electronics", probability=0.8),
    Candidate(name="High-ranking government official", probability=0.5),
    Candidate(name="Tir Tairngire government official", probability=0.5),
    Candidate(name="Aztlan cartel boss", probability=0.5),
    Candidate(name="Halloweeners gang leader", probability=0.5),
    Candidate(name="Tech startup CEO", probability=0.5),
    Candidate(name="Renowned scientist", probability=0.5),
    Candidate(name="Media mogul", probability=0.5),
    Candidate(name="Ex-military commander", probability=0.5),
    Candidate(name="Revolutionary ideologue", probability=0.5),
    Candidate(name="Yakushima elven leader", probability=0.5),
    Candidate(name="Wealthy private collector", probability=0.3),
    Candidate(name="Famous celebrity", probability=0.3),
    Candidate(name="Humanitarian aid organization", probability=0.3),
    Candidate(name="Underground resistance leader", probability=0.3),
    Candidate(name="Corporate security chief", probability=0.8),
    Candidate(name="Megacorp black-ops handler", probability=0.8),
    Candidate(name="Cybernetics smuggler", probability=0.5),
    Candidate(name="Street-level gang boss", probability=0.5),
    Candidate(name="Corporate espionage specialist", probability=0.8),
    Candidate(name="Wetwork coordinator", probability=0.8),
    Candidate(name="Arcano-tech researcher", probability=0.5),
    Candidate(name="Media blackmailer", probability=0.1),
    Candidate(name="Elven nationalist operative", probability=0.5),
    Candidate(name="Drake in disguise", probability=0.3),
    Candidate(name="Free spirit needing help", probability=0.3),
    Candidate(name="Yakuza lieutenant", probability=0.5),
    Candidate(name="Triad negotiator", probability=0.5),
]

mission_types = [
    Candidate(name="Data Extraction", probability=1.0),
    Candidate(name="Asset Recovery", probability=0.95),
    Candidate(name="Infiltration", probability=0.95),
    Candidate(name="Extraction", probability=0.9),
    Candidate(name="Surveillance", probability=0.9),
    Candidate(name="Counterintelligence", probability=0.85),
    Candidate(name="Assassination", probability=0.85),
    Candidate(name="Heist", probability=0.85),
    Candidate(name="Smuggling", probability=0.8),
    Candidate(name="Rescue", probability=0.8),
    Candidate(name="Protection", probability=0.8),
    Candidate(name="Corporate Espionage", probability=0.8),
    Candidate(name="Bounty Hunting", probability=0.75),
    Candidate(name="Artifact Retrieval", probability=0.75),
    Candidate(name="Cyberwarfare", probability=0.75),
    Candidate(name="Hostage Negotiation", probability=0.7),
    Candidate(name="Blackmail", probability=0.7),
    Candidate(name="Sabotage", probability=0.7),
    Candidate(name="Courier Run", probability=0.7),
    Candidate(name="Interception", probability=0.7),
    Candidate(name="Hacking", probability=0.7),
    Candidate(name="Research and Development Theft", probability=0.65),
    Candidate(name="Mercenary Contract", probability=0.65),
    Candidate(name="Urban Exploration", probability=0.6),
    Candidate(name="Gang Warfare", probability=0.6),
    Candidate(name="Piracy", probability=0.6),
    Candidate(name="Illegal Auction", probability=0.6),
    Candidate(name="Arms Dealing", probability=0.6),
    Candidate(name="Bodyguarding", probability=0.6),
    Candidate(name="Undercover Operation", probability=0.6),
    Candidate(name="Hijacking", probability=0.6),
    Candidate(name="Media Manipulation", probability=0.6),
    Candidate(name="Intellectual Property Theft", probability=0.6),
    Candidate(name="Corporate Raid", probability=0.6),
    Candidate(name="Heist Planning", probability=0.6),
    Candidate(name="Political Assassination", probability=0.6),
    Candidate(name="Clandestine Meeting", probability=0.6),
    Candidate(name="Defector Extraction", probability=0.6),
    Candidate(name="Underworld Negotiation", probability=0.6),
    Candidate(name="Bribery", probability=0.6),
    Candidate(name="Extortion", probability=0.6),
    Candidate(name="Witness Protection", probability=0.6),
    Candidate(name="Spirit Negotiation", probability=0.6),
    Candidate(name="Toxic Cleanup", probability=0.6),
    Candidate(name="Reality Show Sabotage", probability=0.6),
    Candidate(name="Virtual Reality Hacking", probability=0.6),
    Candidate(name="Genetic Engineering Sabotage", probability=0.6),
    Candidate(name="Mind Control Disruption", probability=0.6),
    Candidate(name="Bioweapon Containment", probability=0.6),
    Candidate(name="Society Infiltration", probability=0.6),
    Candidate(name="High-Stakes Gambling", probability=0.6),
    Candidate(name="Threat Neutralization", probability=0.6),
    Candidate(name="Criminal Syndicate Takedown", probability=0.6),
    Candidate(name="Mafia Turf War", probability=0.6),
    Candidate(name="Corporate Cover-Up", probability=0.6),
    Candidate(name="Identity Theft", probability=0.6),
    Candidate(name="Digital Assassination", probability=0.6),
    Candidate(name="Psychological Warfare", probability=0.6),
    Candidate(name="Time-Sensitive Courier Delivery", probability=0.6),
    Candidate(name="Body Part Retrieval", probability=0.6),
    Candidate(name="Archaeological Excavation", probability=0.6),
    Candidate(name="Astral Rift Closure", probability=0.6),
    Candidate(name="Rogue AI Containment", probability=0.6),
    Candidate(name="Shadow Spirit Banishment", probability=0.6),
    Candidate(name="Matrix Dive", probability=0.6),
    Candidate(name="Quantum Encryption Cracking", probability=0.6),
    Candidate(name="Secret Society Initiation", probability=0.6),
    Candidate(name="Museum Heist", probability=0.6),
    Candidate(name="Underground Fighting Ring", probability=0.6),
    Candidate(name="VIP Extraction", probability=0.6),
    Candidate(name="Body Disposal", probability=0.6),
    Candidate(name="Paranormal Investigation", probability=0.6),
    Candidate(name="Rogue Nanite Swarm Destruction", probability=0.6),
    Candidate(name="Urban Legend Debunking", probability=0.6),
    Candidate(name="Corporate Retreat Infiltration", probability=0.6),
    Candidate(name="Mysterious Disappearance Investigation", probability=0.6),
    Candidate(name="Disinformation Campaign", probability=0.6),
    Candidate(name="Illegal Drug Manufacturing Facility Destruction", probability=0.6),
    Candidate(name="Criminal Mastermind Capture", probability=0.6),
    Candidate(name="Organ Harvesting Ring Dismantling", probability=0.5),
]


class Oracle:
    @staticmethod
    def weighted_choice(items: list[Candidate]) -> str:
        """Selects an item based on its probability."""
        names = [item.name for item in items]
        weights = [item.probability for item in items]
        return random.choices(names, weights=weights, k=1)[0]

    @staticmethod
    def mission() -> str:
        # Step 1: Draw a client
        client = Oracle.weighted_choice(clients)

        # Step 2: Draw a mission type
        mission = Oracle.weighted_choice(mission_types)

        # Step 3: Draw a target (different from client)
        eligible_targets = [c for c in clients if c.name != client]
        target = Oracle.weighted_choice(eligible_targets)

        return json.dumps(
            {
                "client": client,
                "target": target,
                "mission": mission,
            }
        )
