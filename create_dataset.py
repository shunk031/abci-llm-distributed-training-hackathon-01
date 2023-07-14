import json

import datasets as ds
from tqdm.auto import tqdm


#
# 日本語で instruction を書いたら loss の下がりが悪かったので英語のままに
#
# PROMPT_DICT = {
#     "prompt_input": (
#         "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。"
#         "要求を適切に満たす応答を書きなさい。\n\n"
#         "### 指示:\n{instruction}\n\n### 入力:{input}\n\n### 応答:"
#     ),
#     "prompt_no_input": (
#         "以下は、タスクを説明する指示です。" "要求を適切に満たす応答を書きなさい。\n\n" "### 指示:\n{instruction}\n\n### 応答:"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def main():
    dataset = ds.load_dataset("izumi-lab/llm-japanese-dataset")

    with open("train.jsonl", "w") as wf:
        for row in tqdm(dataset["train"]):
            if "input" in row:
                prompt = PROMPT_DICT["prompt_input"].format_map(row)
            else:
                prompt = PROMPT_DICT["prompt_no_input"].format_map(row)

            response = row["output"]

            if response is not None:
                data = {"prompt": prompt, "response": response}
                json.dump(data, wf, ensure_ascii=False)
                wf.write("\n")


if __name__ == "__main__":
    main()
