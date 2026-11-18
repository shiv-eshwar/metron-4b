"""A fixed, zero-shot sample from two public test sets. No teacher calls."""

import random

from huggingface_hub import hf_hub_download

SOURCES = {
    "gsm8k": ("openai/gsm8k", "740312add88f781978c0658806c59bc2815b9866", "main/test-00000-of-00001.parquet"),
    "arc": ("allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge/test-00000-of-00001.parquet"),
}


def public_tasks():
    import pyarrow.parquet as pq

    tasks = []
    for name, (repo, revision, filename) in SOURCES.items():
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        rows = pq.read_table(path).to_pylist()
        for i in sorted(random.Random(20261118).sample(range(len(rows)), 16)):
            row = rows[i]
            if name == "gsm8k":
                answer = row["answer"].rsplit("####", 1)[1].strip().replace(",", "")
                prompt = row["question"] + '\nShow a brief calculation and end with "Answer: <number>" without units.'
                check = {"kind": "number", "answer": answer}
            else:
                choices = "\n".join(f"{key}. {value}" for key, value in
                                    zip(row["choices"]["label"], row["choices"]["text"], strict=True))
                prompt = row["question"] + "\n" + choices + "\nReply with the correct choice label only."
                check = {"kind": "choice", "answer": row["answerKey"]}
            tasks.append({"id": f"{name}-test-{i}", "domain": name, "prompt": prompt, "check": check})
    return tasks
