from src.llmclient.llm_parameters import LLMConfig


# PROFILE: THE THINKING ENGINE
# Optimized for: Logic, JSON extraction, and high-fidelity summaries.
LLM_CONFIG_THINKING = LLMConfig(
    temperature=0.1,
    min_p=0.05,
    top_p=1.0,
    top_k=0,
    repetition_penalty=1.05,  # Prevents logic loops in long summaries
    repetition_penalty_range=512,
    smoothing_factor=0.0,
    sampler_priority=[
        "min_p",
        "temperature",
        "repetition_penalty",
    ],
)

# PROFILE: THE ARCHITECT (Story Harness Generation)
# Optimized for: Creative world-building within strict structural constraints.
LLM_CONFIG_ARCHITECT = LLMConfig(
    temperature=0.95,  # High enough for creative sparks
    min_p=0.05,  # Strict cutoff to prevent JSON syntax errors
    top_p=1.0,
    top_k=0,
    smoothing_factor=0.25,  # Adds "flavor" to the prose without losing the plot
    repetition_penalty=1.1,  # Slightly higher to force diverse NPC archetypes
    repetition_penalty_range=1024,
    presence_penalty=0.25,  # Encourages the model to introduce new concepts
    sampler_priority=[
        "min_p",
        "smoothing_factor",
        "temperature",
        "repetition_penalty",
    ],
)

# PROFILE: THE STORYTELLER ENGINE
# Optimized for: Narrative flow, NPC dialogue, and immersion.
LLM_CONFIG_STORY = LLMConfig(
    temperature=1.0,
    min_p=0.1,
    top_p=1.0,
    top_k=0,
    smoothing_factor=0.23,  # Prevents "cliché" word choices.
    repetition_penalty=1.05,
    repetition_penalty_range=600,
    presence_penalty=0.1,
    frequency_penalty=0.0,  # Generally redundant if using Rep-Pen and Min-P
    sampler_priority=[
        "min_p",  # 1. Precision filter.
        "smoothing_factor",  # 3. Prose variety.
        "temperature",  # 4. Final creative weight.
        "repetition_penalty",  # 2. Loop prevention.
    ],
)
