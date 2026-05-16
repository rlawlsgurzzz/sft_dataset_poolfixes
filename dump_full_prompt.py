import ast
import json
import sys
from pathlib import Path

sys.path.append("scripts")

from sft_generation_request import build_mixed_generation_payload

src = Path("scripts/sft_teacher_client.py").read_text(encoding="utf-8")
module = ast.parse(src)

system_prompt = None
for node in module.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "SYSTEM_PROMPT":
                value = node.value
                if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                    system_prompt = ast.literal_eval(value.func.value).strip()
                else:
                    system_prompt = ast.literal_eval(value)

if system_prompt is None:
    raise RuntimeError("SYSTEM_PROMPT not found")

payload = build_mixed_generation_payload(
    mixed_requests=[
        {
            "request": "c1-1-1-1-1-1.1",
            "cycle_start_offset": 0,
        }
    ],
    target_split="train",
)

contents = json.dumps(payload, ensure_ascii=False, indent=2)

combined_prompt = (
    "<SYSTEM_INSTRUCTION>`n"
    + system_prompt
    + "`n</SYSTEM_INSTRUCTION>`n`n"
    + "<USER_CONTENTS>`n"
    + contents
    + "`n</USER_CONTENTS>`n"
)

Path("sft_full_prompt_c1-1-1-1-1-1_train.txt").write_text(
    combined_prompt,
    encoding="utf-8",
)

Path("sft_system_instruction.txt").write_text(
    system_prompt,
    encoding="utf-8",
)

Path("sft_user_contents_train.json").write_text(
    contents,
    encoding="utf-8",
)

print("wrote sft_full_prompt_c1-1-1-1-1-1_train.txt")
print("wrote sft_system_instruction.txt")
print("wrote sft_user_contents_train.json")
