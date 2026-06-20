# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end greedy-parity tests for the ReplaySSM Mamba2 AR-decode path.

Enabling ``--use-replayssm`` must produce token-identical greedy output to the
stored-state baseline on a real hybrid Mamba2 model, for both compute routes
("output_only" / "state_and_output"), in eager and full-CUDA-graph modes. The
decode length exceeds the ring-buffer length so the buffer wraps and flushes
multiple times during the run (the highest-value correctness case).
"""

import pytest

from ...utils import check_outputs_equal

pytestmark = pytest.mark.hybrid_model

# Small real Mamba2 hybrid (every layer has a Mamba2 mixer). Greedy parity does
# not depend on weight quality, only on replay/stored-state path equivalence.
MODEL = "tiiuae/Falcon-H1-0.5B-Base"
BUFFER_LEN = 4
MAX_TOKENS = 32  # >> BUFFER_LEN so the ring buffer wraps/flushes during decode
MAX_NUM_SEQS = 4


@pytest.mark.parametrize("route", ["output_only", "state_and_output"])
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_replayssm_matches_baseline(
    vllm_runner,
    example_prompts,
    route: str,
    enforce_eager: bool,
) -> None:
    with vllm_runner(
        MODEL,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
    ) as baseline:
        baseline_out = baseline.generate_greedy(example_prompts, MAX_TOKENS)

    with vllm_runner(
        MODEL,
        max_num_seqs=MAX_NUM_SEQS,
        enforce_eager=enforce_eager,
        enable_prefix_caching=False,
        use_replayssm=True,
        replayssm_route=route,
        replayssm_buffer_len=BUFFER_LEN,
    ) as replay:
        replay_out = replay.generate_greedy(example_prompts, MAX_TOKENS)

    check_outputs_equal(
        outputs_0_lst=baseline_out,
        outputs_1_lst=replay_out,
        name_0="baseline",
        name_1=f"replayssm-{route}",
    )
