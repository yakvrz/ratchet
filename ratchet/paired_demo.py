from __future__ import annotations

from ratchet.__main__ import paired_demo_defaults, run_optimizer


def main() -> None:
    adapter_spec, evals_path, out_dir = paired_demo_defaults()
    run_optimizer(
        adapter_spec=adapter_spec,
        evals_path=evals_path,
        out_dir=out_dir,
        env_file=".env",
        dev_budget=20,
        holdout_top_k=5,
        harnesser_model="gpt-5.4",
        harnesser_reasoning="medium",
        harnesser_enabled=True,
    )


if __name__ == "__main__":
    main()
