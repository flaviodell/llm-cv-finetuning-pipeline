import json
import wandb
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from src.utils.common import get_env


def load_results() -> dict:
    with open("outputs/cv/cv_results.json")  as f:
        cv_results  = json.load(f)
    with open("outputs/llm/llm_results.json") as f:
        llm_results = json.load(f)
    return {"cv": cv_results, "llm": llm_results}


def generate_benchmark_report(results: dict, config: dict):
    output_dir = Path("outputs/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    cv  = results["cv"]
    llm = results["llm"]

    # Summary figure with two panels
    fig = plt.figure(figsize=(14, 6))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Left panel — CV metrics
    ax1 = fig.add_subplot(gs[0])
    metrics_cv = ["Accuracy", "F1 macro"]
    values_cv  = [cv["test_accuracy"], cv["test_f1_macro"]]
    bars = ax1.bar(metrics_cv, values_cv, color=["#4C8BF5", "#34A853"], width=0.4)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("CV model — ResNet50\nOxford Pets test set", fontsize=12)
    ax1.set_ylabel("Score")
    for bar, val in zip(bars, values_cv):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold"
        )

    # Right panel — LLM metrics
    ax2 = fig.add_subplot(gs[1])
    metrics_llm = ["Faithfulness", "Answer\nrelevancy"]
    values_llm  = [llm["faithfulness"], llm["answer_relevancy"]]
    bars2 = ax2.bar(metrics_llm, values_llm, color=["#EA4335", "#FBBC05"], width=0.4)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("LLM — Mistral 7B + LoRA\nRAGAS evaluation", fontsize=12)
    ax2.set_ylabel("Score")
    for bar, val in zip(bars2, values_llm):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold"
        )

    plt.suptitle("Project evaluation summary", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "benchmark_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Benchmark summary saved to outputs/benchmark/benchmark_summary.png")

    # Log everything to W&B
    wandb.init(
        project=config["wandb"]["project"],
        name="benchmark-final",
        config=config
    )
    wandb.log({
        "cv/test_accuracy":        cv["test_accuracy"],
        "cv/test_f1_macro":        cv["test_f1_macro"],
        "llm/faithfulness":        llm["faithfulness"],
        "llm/answer_relevancy":    llm["answer_relevancy"],
    })
    wandb.log({
        "benchmark_summary": wandb.Image("outputs/benchmark/benchmark_summary.png"),
        "confusion_matrix":  wandb.Image("outputs/cv/confusion_matrix.png"),
        "per_class_accuracy": wandb.Image("outputs/cv/per_class_accuracy.png"),
    })
    wandb.finish()
    print("All metrics and charts logged to W&B.")

    # Save combined JSON
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Combined results saved to outputs/benchmark/benchmark_results.json")


if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    results = load_results()
    generate_benchmark_report(results, config)
