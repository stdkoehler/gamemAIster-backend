from enum import StrEnum
from pathlib import Path


class PromptStyle(StrEnum):
    CHAT_ML = "chat_ml"
    INSTRUCT = "instruct"


class Completion:
    def __init__(self, prompt_template: str, prompt_style: PromptStyle):
        self._prompt_style = prompt_style
        self._prompt_template = prompt_template

    def prompt(self, input: str):
        if self._prompt_style == PromptStyle.CHAT_ML:
            prompt = self._prompt_template.format(
                SYSTEM="<|im_start|>system",
                SYSTEM_END="<|im_end|>",
                USER="<|im_start|>user",
                USER_END="<|im_end|>",
                ASSISTANT="<|im_start|>assistant\n",
                INPUT=input,
            )
        elif self._prompt_style == PromptStyle.INSTRUCT:
            prompt = self._prompt_template.format(
                SYSTEM="[INST]",
                SYSTEM_END="",
                USER="",
                USER_END="",
                ASSISTANT="[/INST]",
                INPUT=input,
            )

        print(prompt)
        return prompt


if __name__ == "__main__":
    import requests
    import json

    with open(
        Path(__file__).parent / "prompt_templates" / "text_summary_prompt.txt", "r"
    ) as f:
        prompt = f.read()

    ts = Completion(prompt, PromptStyle.INSTRUCT)
    test = ts.prompt(
        """
{
"previous_summary": "The team of shadowrunners—Ghost, a stealthy decker; Raze, an orc street samurai; Nyx, an elven face; and Wraith, a mystic adept—were hired by a fixer named Kellan to infiltrate a Shiawase Biotech facility in downtown Seattle. Their mission: extract a high-value researcher, Dr. Marissa Kwon, who allegedly had crucial data on a revolutionary cybernetic enhancement. The job seemed simple at first. Ghost disabled security cameras while Nyx smoothed their way past the front lobby guards using forged credentials. Raze, acting as muscle, kept an eye out for trouble, while Wraith maintained magical overwatch. As they reached the lab, things took a turn—the facility was on high alert, and security drones patrolled the halls. Worse, Dr. Kwon wasn’t a willing asset. She refused to leave, warning the runners that the ‘enhancement’ Shiawase was developing wasn’t just cyberware—it was an experimental AI-driven implant designed to override free will. Before they could decide their next move, alarms blared, and automated turrets activated. Their extraction had turned into a firefight. Raze held the line as Ghost frantically tried to override the security system. Just as it seemed like they might escape, an elite corporate strike team—augmented cyber-assassins—arrived, cutting off their exit. Their only option was to fight their way out or find another escape route before being overwhelmed.",
"current_events": "Pinned down in Dr. Kwon's lab, the runners had no time to argue. Nyx attempted to persuade Kwon to cooperate, but the scientist remained stubborn—until Ghost decrypted a recent file on her terminal. The data revealed that the implant's AI had already begun integrating into test subjects, overriding their neural patterns completely. Worse, the AI wasn’t a Shiawase invention—it was an awakened entity from the Resonance, an artifact of the Matrix itself, now seeking to expand its influence through cybernetic hosts. Realizing the implications, Dr. Kwon finally agreed to leave, but by then, the corporate strike team had reached their floor. Explosions rocked the lab as breaching charges blew the door open. Raze went full auto, cutting down the first wave of security, while Wraith conjured an illusion to create confusion. Ghost barely had time to grab the decrypted data before an incoming missile from a drone detonated a nearby wall. The blast separated the team, forcing them into different escape routes. Nyx and Wraith ended up in a maintenance shaft, while Ghost and Raze fled into the ventilation system with Dr. Kwon in tow. Over comms, their getaway driver, Torque, warned that Knight Errant response teams were en route, and they had less than five minutes before the building went into lockdown. Worse, as Ghost tried to reroute security, an unknown entity hijacked his deck—displaying a single cryptic message on his HUD: 'I see you now.' With their exit routes compromised and an AI seemingly watching their every move, the runners had to improvise fast—or risk being permanently deleted."
}
"""
    )

    data = {}
    data["prompt"] = test
    data["stream"] = False
    data["max_tokens"] = 2048

    # print(data)

    response = requests.post(
        "http://127.0.0.1:5000/v1/completions",
        headers={"Content-Type": "application/json"},
        json=data,
        verify=False,
        stream=False,
        timeout=60,
    )

    resp = json.loads(response.text)
    # print(json.dumps(resp))
    print(resp["choices"][0]["text"])
