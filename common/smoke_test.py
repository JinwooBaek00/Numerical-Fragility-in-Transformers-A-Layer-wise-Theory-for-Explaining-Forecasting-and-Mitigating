from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from common.artifacts import save_json_artifact, save_text_artifact
    from common.run import create_run_context
else:
    from .artifacts import save_json_artifact, save_text_artifact
    from .run import create_run_context


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    experiment_dir = workspace_root / "e1_controlled"

    context = create_run_context(
        experiment_dir,
        short_tag="dummy_smoke",
        config={
            "experiment": "E1",
            "case": "dummy",
            "notes": "Smoke test for common infrastructure.",
        },
        metadata={
            "model_name": "controlled-dummy",
            "dataset_name": "synthetic",
            "precision": "fp32",
            "seed": 0,
            "sequence_length": 0,
        },
        workspace_root=workspace_root,
    )
    context.append_stdout("Starting common infrastructure smoke test.")
    context.write_metrics(
        {
            "num_rows_per_step": 3,
            "num_rows_per_layer": 2,
            "pass": True,
        }
    )
    context.write_rows(
        "per_step_metrics.csv",
        [
            {"step": 0, "seed": 0, "precision": "fp32", "sequence_length": 0, "loss": 0.0, "final_mismatch": 0.0, "predicted_risk_sum": 0.0, "event_flag": 0},
            {"step": 1, "seed": 0, "precision": "fp32", "sequence_length": 0, "loss": 0.0, "final_mismatch": 0.1, "predicted_risk_sum": 0.1, "event_flag": 0},
            {"step": 2, "seed": 0, "precision": "fp32", "sequence_length": 0, "loss": 0.0, "final_mismatch": 0.2, "predicted_risk_sum": 0.2, "event_flag": 1},
        ],
    )
    context.write_rows(
        "per_layer_metrics.csv",
        [
            {"step": 2, "layer": 0, "seed": 0, "precision": "fp32", "sequence_length": 0, "risk_score": 0.2, "ln_magnitude": 0.1, "attn_magnitude": 0.08, "remainder_magnitude": 0.02, "ln_dominance": 0.5, "rho_ln": 0.3},
            {"step": 2, "layer": 1, "seed": 0, "precision": "fp32", "sequence_length": 0, "risk_score": 0.3, "ln_magnitude": 0.15, "attn_magnitude": 0.1, "remainder_magnitude": 0.05, "ln_dominance": 0.5, "rho_ln": 0.2},
        ],
    )
    context.write_summary(
        {
            "goal": "Validate that the common infrastructure writes a contract-compliant run.",
            "setup": [
                "Experiment: E1 dummy smoke test",
                "Workspace root inferred from nft_experiments/",
                "No model execution is performed",
            ],
            "key_metrics": [
                "per_step_metrics.csv written",
                "per_layer_metrics.csv written",
                "metrics.json written",
            ],
            "pass_fail_verdict": "Pass",
            "anomalies": "None",
            "follow_up": "Use the same API from e1_controlled/src/ when implementing the first real run.",
        }
    )
    save_text_artifact(context.paths.outputs_dir, "e1_smoke_note.md", "# Smoke Output\n\nThis file validates the outputs/ copy target.\n")
    save_json_artifact(
        context.paths.outputs_dir,
        "e1_smoke_manifest.json",
        {"experiment_id": "e1_controlled", "source_run_id": context.run_id},
    )
    context.mark_completed()
    context.append_stdout("Smoke test completed.")


if __name__ == "__main__":
    main()
