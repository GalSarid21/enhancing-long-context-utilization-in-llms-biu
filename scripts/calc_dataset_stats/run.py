import matplotlib.pyplot as plt
import numpy as np
import asyncio
import json
import sys

from statistics import mean, median, stdev, variance
from pathlib import Path
from typing import List, Dict, Any, Union


project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.helpers.nq_data import read_data_file
from src.entities.enums import PromptingMode
from src.wrappers import HfTokenizer


def _get_all_unique_docs() -> Dict:
    print(f"_get_all_unique_docs - started")
    questions_data = asyncio.run(
        read_data_file(prompting_mode=PromptingMode.OPENBOOK)
    )
    print(f"_get_all_unique_docs - fetched data")

    all_docs = {}

    for question_data in questions_data:
        question_docs = question_data.documents + question_data.gold_docs
        for question_doc in question_docs:
            if question_doc.id not in all_docs:
                print(f"_get_all_unique_docs - added new unique doc: id={question_doc.id}")
                all_docs.update({question_doc.id: f"{question_doc.title}\n{question_doc.text}"})
    
    print(f"_get_all_unique_docs - finisehd: found {len(all_docs.keys())} unique questions")
    return all_docs


def _collect_unique_docs_tokens_cnt_by_model(models: List[str], unique_docs: Dict) -> None:
    print(f"_collect_unique_docs_tokens_cnt_by_model - started: {models=}")
    res = {}
    
    for model in models:
        print(f"_collect_unique_docs_tokens_cnt_by_model - model in use: {model}")
        tokenizer = HfTokenizer(model)
        print(f"_collect_unique_docs_tokens_cnt_by_model - tokenizer initiated")

        num_unique_docs = len(unique_docs.values())
        model_res = {}

        for i, (id, text) in enumerate(unique_docs.items(), 1):
            print(f"_collect_unique_docs_tokens_cnt_by_model - start processing: {i}/{num_unique_docs}")
            tokens_cnt = tokenizer.count_tokens(prompt=text)
            model_res[id] = tokens_cnt
        
        res[model] = model_res
    
    print(f"_collect_unique_docs_tokens_cnt_by_model - finished")
    return res


def _get_stats(tokens_cnt_by_model: Dict) -> Dict:
    print(f"_get_stats - started")
    res = {}

    for model, data in tokens_cnt_by_model.items():
        print(f"_get_stats - start processing model {model}")
        all_tokens_cnt = [v for v in data.values()]
        stats = _analyze_token_stats(token_counts=all_tokens_cnt)
        short_model_name = model.split("/")[-1]
        res[short_model_name] = stats
    
    return res


def _analyze_token_stats(token_counts: List[int], percentiles=(50, 75, 90, 95, 99)):
    n = len(token_counts)

    # Basic stats
    stats = {
        "count": n,
        "total_tokens": sum(token_counts),
        "min": min(token_counts),
        "max": max(token_counts),
        "mean": mean(token_counts),
        "median": median(token_counts),
    }

    # Only compute stdev/variance if enough data
    if n > 1:
        stats["std"] = stdev(token_counts)
        stats["variance"] = variance(token_counts)
    else:
        stats["std"] = None
        stats["variance"] = None

    # Percentiles (prefer NumPy if available)
    stats["percentiles"] = {
        f"p{p}": float(np.percentile(token_counts, p)) for p in percentiles
    }

    # Histogram (counts only — no dependencies)
    sorted_vals = sorted(token_counts)
    bins = 10
    min_val, max_val = sorted_vals[0], sorted_vals[-1]
    bin_width = (max_val - min_val) / bins if max_val != min_val else 1

    histogram = {f"bin_{i}": 0 for i in range(bins)}
    for x in token_counts:
        if max_val == min_val:
            histogram["bin_0"] += 1
        else:
            idx = min(int((x - min_val) / bin_width), bins - 1)
            histogram[f"bin_{idx}"] += 1

    stats["histogram"] = histogram
    return stats    


def _save_stats_by_model(
    stats_by_model: Dict[str, Dict[str, Any]],
    base_dir: Union[str, Path] = "./data/stats",
) -> None:
    """
    Given a dict: {model_name: stats_dict} (from _get_stats/_analyze_token_stats),
    create ./data/stats/{model_name} folder for each model, and inside it:
      1. stats.json - all stats except histogram
      2. histogram.png - bar plot of the histogram
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for model_name, stats in stats_by_model.items():
        print(f"_save_stats_by_model - started: {model_name=}")

        model_dir = base_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        print(f"_save_stats_by_model - creating json")
        json_path = model_dir / "stats.json"
        with open(json_path, "w") as f:
            json.dump(
                {k: v for k, v in stats.items() if k != "histogram"},
                f,
                indent=2
            )

        print(f"_save_stats_by_model - creating histogram plot")
        histogram = stats.get("histogram")
        if histogram:
            _plot_histogram(
                stats=stats,
                histogram=histogram,
                model_name=model_name,
                output_path=model_dir / "histogram.png",
            )


def _plot_histogram(
    stats: Dict[str, Any],
    histogram: Dict[str, int],
    model_name: str,
    output_path: Path,
) -> None:
    # Reconstruct binning info from stats + histogram structure
    min_val = stats["min"]
    max_val = stats["max"]

    # Number of bins is the number of entries in the histogram dict
    # (assuming keys like 'bin_0', 'bin_1', ..., 'bin_9')
    bin_keys = sorted(histogram.keys(), key=lambda k: int(k.split("_")[1]))
    bin_counts = [histogram[k] for k in bin_keys]
    num_bins = len(bin_keys)

    if max_val == min_val:
        bin_width = 1
    else:
        bin_width = (max_val - min_val) / num_bins

    # Bin edges & centers (for nicer X-axis meaning)
    edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(num_bins)]

    # Plot
    plt.figure()
    plt.bar(centers, bin_counts, width=bin_width * 0.9)
    plt.title(f"Token count histogram - {model_name}")
    plt.xlabel("Token count (approx binned)")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run() -> None:
    unique_docs: Dict = _get_all_unique_docs()
    models = ["google/gemma-3-4b-it", "meta-llama/Llama-3.2-3B-Instruct", "Qwen/Qwen3-4B-Instruct-2507"]
    tokens_cnt_by_model: Dict = _collect_unique_docs_tokens_cnt_by_model(models=models,unique_docs=unique_docs)
    stats: Dict = _get_stats(tokens_cnt_by_model=tokens_cnt_by_model)
    _save_stats_by_model(stats_by_model=stats)


if __name__ == "__main__":
    run()