import json
import wandb
import yaml
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from src.utils.common import get_env


def log_cv_training_curves(project: str):
    # Re-logs CV training history if the original run is missing or incomplete
    # Load W&B run history from local outputs if available
    cv_results_path = Path("outputs/cv/cv_results.json")
    if not cv_results_path.exists():
        print("outputs/cv/cv_results.json not found — skipping CV curve logging.")
        return

    with open(cv_results_path) as f:
        cv_results = json.load(f)

    run = wandb.init(
        project=project,
        name="cv-final-summary",
        job_type="summary"
    )

    # Log final test metrics as summary values
    wandb.summary["cv/test_accuracy"] = cv_results["test_accuracy"]
    wandb.summary["cv/test_f1_macro"] = cv_results["test_f1_macro"]
    wandb.summary["cv/best_val_accuracy"] = cv_results["best_val_accuracy"]
    wandb.summary["cv/best_val_loss"]     = cv_results["best_val_loss"]

    # Log confusion matrix image
    cm_path = Path("outputs/cv/confusion_matrix.png")
    if cm_path.exists():
        wandb.log({"cv/confusion_matrix": wandb.Image(str(cm_path))})

    # Log per-class accuracy image
    pc_path = Path("outputs/cv/per_class_accuracy.png")
    if pc_path.exists():
        wandb.log({"cv/per_class_accuracy": wandb.Image(str(pc_path))})

    wandb.finish()
    print("CV summary logged to W&B.")


def log_llm_summary(project: str):
    llm_results_path = Path("outputs/llm/llm_results.json")
    if not llm_results_path.exists():
        print("outputs/llm/llm_results.json not found — skipping LLM summary logging.")
        return

    with open(llm_results_path) as f:
        llm_results = json.load(f)

    quant_path = Path("outputs/llm/quantization_results.json")
    quant_results = {}
    if quant_path.exists():
        with open(quant_path) as f:
            quant_results = json.load(f)

    run = wandb.init(
        project=project,
        name="llm-final-summary",
        job_type="summary"
    )

    wandb.summary["llm/faithfulness"]     = llm_results["faithfulness"]
    wandb.summary["llm/answer_relevancy"] = llm_results["answer_relevancy"]

    if quant_results:
        fp16_mb  = quant_results["fp16"]["params_mb"]
        quant_mb = quant_results["4bit"]["params_mb"]
        reduction = (1 - quant_mb / fp16_mb) * 100
        wandb.summary["llm/fp16_params_mb"]         = fp16_mb
        wandb.summary["llm/4bit_params_mb"]          = quant_mb
        wandb.summary["llm/quantization_reduction_pct"] = round(reduction, 1)

    wandb.finish()
    print("LLM summary logged to W&B.")


def create_final_dashboard_report(project: str):
    # Creates a single comprehensive summary run with all metrics and charts
    cv_results_path  = Path("outputs/cv/cv_results.json")
    llm_results_path = Path("outputs/llm/llm_results.json")
    quant_path       = Path("outputs/llm/quantization_results.json")

    if not cv_results_path.exists() or not llm_results_path.exists():
        print("Missing results files — run eval_cv.py and eval_llm.py first.")
        return

    with open(cv_results_path)  as f: cv_results  = json.load(f)
    with open(llm_results_path) as f: llm_results = json.load(f)

    quant_results = {}
    if quant_path.exists():
        with open(quant_path) as f:
            quant_results = json.load(f)

    run = wandb.init(
        project=project,
        name="project-dashboard",
        job_type="dashboard"
    )

    # --- Comprehensive summary figure ---
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1 — CV test metrics
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ["Accuracy", "F1 macro"]
    values  = [cv_results["test_accuracy"], cv_results["test_f1_macro"]]
    bars = ax1.bar(metrics, values, color=["#4C8BF5", "#34A853"], width=0.4)
    ax1.set_ylim(0, 1.0)
    ax1.set_title("CV — test metrics\nResNet50 + Oxford Pets", fontsize=11)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold"
        )

    # Panel 2 — LLM RAGAS metrics
    ax2 = fig.add_subplot(gs[0, 1])
    metrics_llm = ["Faithfulness", "Answer\nrelevancy"]
    values_llm  = [llm_results["faithfulness"], llm_results["answer_relevancy"]]
    bars2 = ax2.bar(metrics_llm, values_llm, color=["#EA4335", "#FBBC05"], width=0.4)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("LLM — RAGAS metrics\nMistral 7B + LoRA", fontsize=11)
    for bar, val in zip(bars2, values_llm):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}", ha="center", fontsize=11, fontweight="bold"
        )

    # Panel 3 — Quantization reduction
    ax3 = fig.add_subplot(gs[0, 2])
    if quant_results:
        fp16_mb  = quant_results["fp16"]["params_mb"] / 1024
        quant_mb = quant_results["4bit"]["params_mb"] / 1024
        ax3.bar(["FP16", "4-bit NF4"], [fp16_mb, quant_mb],
                color=["#FF6B6B", "#4ECDC4"], width=0.4)
        ax3.set_ylabel("Size (GB)")
        ax3.set_title(f"Quantization\n{((1 - quant_mb/fp16_mb)*100):.1f}% size reduction", fontsize=11)
        for i, val in enumerate([fp16_mb, quant_mb]):
            ax3.text(i, val + 0.1, f"{val:.1f} GB",
                     ha="center", fontsize=11, fontweight="bold")

    # Panel 4 — Val vs best val comparison for CV
    ax4 = fig.add_subplot(gs[1, 0])
    categories = ["Best val acc", "Test acc"]
    values4    = [cv_results["best_val_accuracy"], cv_results["test_accuracy"]]
    ax4.bar(categories, values4, color=["#A8DADC", "#457B9D"], width=0.4)
    ax4.set_ylim(0, 1.0)
    ax4.set_title("CV — val vs test accuracy", fontsize=11)
    for i, val in enumerate(values4):
        ax4.text(i, val + 0.02, f"{val:.3f}",
                 ha="center", fontsize=11, fontweight="bold")

    # Panel 5 — Project summary text
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis("off")
    summary_text = (
        f"Project summary\n\n"
        f"CV model   ResNet50 pretrained on ImageNet\n"
        f"           Fine-tuned on Oxford Pets (37 breeds)\n"
        f"           Test accuracy : {cv_results['test_accuracy']:.3f}\n"
        f"           Test F1 macro : {cv_results['test_f1_macro']:.3f}\n\n"
        f"LLM model  Mistral 7B Instruct v0.3\n"
        f"           Fine-tuned with LoRA (r=16) on synthetic dataset\n"
        f"           185 examples, 37 breeds, veterinary domain\n"
        f"           Faithfulness     : {llm_results['faithfulness']:.3f}\n"
        f"           Answer relevancy : {llm_results['answer_relevancy']:.3f}\n\n"
        f"Quantization  FP16 → 4-bit NF4\n"
    )
    if quant_results:
        fp16_mb  = quant_results["fp16"]["params_mb"] / 1024
        quant_mb = quant_results["4bit"]["params_mb"] / 1024
        reduction = (1 - quant_mb / fp16_mb) * 100
        summary_text += f"              {fp16_mb:.1f} GB → {quant_mb:.1f} GB ({reduction:.1f}% reduction)"

    ax5.text(
        0.05, 0.95, summary_text,
        transform=ax5.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5)
    )

    output_dir = Path("outputs/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Project finetuning — full dashboard", fontsize=14, fontweight="bold")
    plt.savefig(output_dir / "full_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Dashboard figure saved to outputs/benchmark/full_dashboard.png")

    # Log everything to W&B
    wandb.log({
        "dashboard/full_report":       wandb.Image("outputs/benchmark/full_dashboard.png"),
        "dashboard/cv_test_accuracy":  cv_results["test_accuracy"],
        "dashboard/cv_test_f1":        cv_results["test_f1_macro"],
        "dashboard/llm_faithfulness":  llm_results["faithfulness"],
        "dashboard/llm_relevancy":     llm_results["answer_relevancy"],
    })

    if Path("outputs/cv/confusion_matrix.png").exists():
        wandb.log({"dashboard/confusion_matrix": wandb.Image("outputs/cv/confusion_matrix.png")})
    if Path("outputs/cv/per_class_accuracy.png").exists():
        wandb.log({"dashboard/per_class_accuracy": wandb.Image("outputs/cv/per_class_accuracy.png")})
    if Path("outputs/benchmark/benchmark_summary.png").exists():
        wandb.log({"dashboard/benchmark_summary": wandb.Image("outputs/benchmark/benchmark_summary.png")})

    wandb.finish()
    print("Full dashboard logged to W&B.")


def run_wandb_summary(config: dict):
    project = config["wandb"]["project"]
    print(f"Logging summary to W&B project: {project}\n")
    log_cv_training_curves(project)
    log_llm_summary(project)
    create_final_dashboard_report(project)
    print("\nW&B summary complete.")


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    load_dotenv()
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    run_wandb_summary(config)
