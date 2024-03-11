""" LLM Client Types """

from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """
    A class representing the configuration for prompts in a chat conversation.

    Attributes:
        max_tokens (int): The maximum number of tokens allowed in a generated response.
        temperature (float): The temperature value for controlling the creativity of responses.
        top_p (float): The top-p value for controlling the diversity of responses.
        min_p (int): The minimum number of tokens to keep in a generated response.
        top_k (int): The top-k value for creating randomness in responses. (1 = no randomness)
        frequency_penalty (float): The penalty value for penalizing frequent tokens in responses.
        stop_words (list[str]): A list of stop words to which stop continuing response generation.
    """

    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    min_p: float = 0
    top_k: int = 20
    repetition_penalty: float = 1  # nous capbyara #1.15 mixtral
    presence_penalty: float = 0
    frequency_penalty: float = 0
    guidance_scale: float = 1
    mirostat_mode: int = 0
    mirostat_tau: float = 5
    mirostat_eta: float = 0.1
    smoothing_factor: float = 0
    repetition_penalty_range: int = 1024
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    sampler_priority: list[str] = field(
        default_factory=lambda: [
            "temperature",
            "dynamic_temperature",
            "quadratic_sampling",
            "top_k",
            "top_p",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
            "tfs",
            "top_a",
            "min_p",
            "mirostat",
        ]
    )
    stop: list[str] = field(default_factory=list)
    logits_processor: list[str] = field(default_factory=list)
