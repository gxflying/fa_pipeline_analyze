"""针对周期回卷、流式依赖和统计并集的回归检查，在 WSL 中运行。"""
import copy
import json
import unittest
import numpy as np
import flashattention_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(pipeline.DEFAULT_CONFIG.read_text())

    def test_default_work_and_optimal_schedule(self):
        ops, caps, ii, lower, total, pools = pipeline.solve(self.config)
        self.assertEqual((ii, lower), (520, 520))
        self.assertLessEqual(total, 512 * 1024)
        report = pipeline.make_report(self.config, ops, caps, ii, lower, total, pools)
        expected = {"TENSOR": 512, "MUL": 193, "ADD": 129, "SUB": 65,
                    "EXP": 520, "CVT": 64, "REDMAX": 63, "REDSUM": 63}
        for unit, cycles in expected.items():
            self.assertEqual(report["units"][unit]["per_period_cycles"], cycles)
        for engine, resources in pipeline.ENGINE_RESOURCES.items():
            cycles = set()
            for op in ops:
                for resource in resources:
                    for begin, end, _ in op.runs(resource):
                        cycles.update(t % ii for t in range(begin, end))
            self.assertEqual(report["engines"][engine]["per_period_cycles"], len(cycles))
        self.assertLess(report["engines"]["Vector"]["per_period_cycles"],
                        sum(report["units"][u]["per_period_cycles"] for u in pipeline.VECTOR_UNITS))

    def test_streaming_read_and_write_at_default_rates(self):
        ops, _, _, _ = pipeline.make_model(self.config)
        by_name = {op.name: op for op in ops}
        scale = by_name["SCALE"]
        self.assertEqual((scale.read_end, scale.ready_offset), (64, 65))
        self.assertEqual(scale.issue_offsets[0], 1)
        for group, issue in enumerate(scale.issue_offsets):
            arrived = scale.profiles["VECTOR_LD"][:issue].sum()
            self.assertGreaterEqual(arrived, (group + 1) * 256)
        cast = by_name["CVT_P"]
        self.assertEqual(cast.ready_offset, 65)
        self.assertEqual(cast.profiles["VECTOR_ST"].sum(), 8192)
        self.assertEqual(cast.profiles["VECTOR_ST"].max(), 128)

    def test_lower_bandwidth_recomputes_stream_and_schedule(self):
        self.config["l1"]["read_gbps"] = 128
        ops, caps, ii, lower, _, _ = pipeline.solve(self.config)
        scale = next(op for op in ops if op.name == "SCALE")
        self.assertEqual((scale.read_end, scale.ready_offset), (128, 129))
        self.assertEqual(scale.issue_offsets[:3], [2, 4, 6])
        self.assertGreaterEqual(lower, 832)
        self.assertGreaterEqual(ii, lower)
        pipeline.verify_schedule(ops, caps, ii)

    def test_exp_parallelism_and_latency_are_independent(self):
        self.config["vector"]["units"]["EXP"]["parallelism"] = 4
        ops, _, ii, lower, _, _ = pipeline.solve(self.config)
        self.assertEqual(lower, 1040)
        self.assertGreaterEqual(ii, 1040)
        self.config["vector"]["units"]["EXP"]["latency_cycles"] = 9
        changed, _, _, _ = pipeline.make_model(self.config)
        first = {op.name: op for op in ops}
        for op in changed:
            if op.unit == "EXP":
                self.assertEqual(op.ready_offset, first[op.name].ready_offset + 8)
                self.assertEqual(op.issue_offsets, first[op.name].issue_offsets)

    def test_partial_groups_and_single_column(self):
        self.config["shape"].update(query_rows=5, kv_cols=1, head_dim_qk=17, head_dim_v=19)
        ops, caps, ii, _, _, _ = pipeline.solve(self.config)
        by_name = {op.name: op for op in ops}
        self.assertEqual(by_name["ROWMAX"].work, 0)
        self.assertEqual(by_name["ROWSUM"].ready_offset, 0)
        self.assertEqual(by_name["SCALE"].profiles["VECTOR_LD"].sum(), 5 * 4)
        self.assertEqual(by_name["CVT_P"].profiles["VECTOR_ST"].sum(), 5 * 2)
        self.assertEqual(len(by_name["QK"].issue_offsets), 2)
        pipeline.verify_schedule(ops, caps, ii)

    def test_insufficient_l1_is_reported(self):
        self.config["l1"]["size_kib"] = 32
        with self.assertRaisesRegex(ValueError, "至少需要"):
            pipeline.solve(self.config)

    def test_output_bandwidth_and_extra_interface_latency(self):
        self.config["l1"]["write_gbps"] = 64
        ops, _, _, _ = pipeline.make_model(self.config)
        cast = next(op for op in ops if op.name == "CVT_P")
        self.assertEqual(cast.issue_offsets[:3], [0, 2, 4])
        self.assertEqual(cast.ready_offset, 129)
        self.assertEqual(cast.profiles["VECTOR_ST"].max(), 64)
        self.config["vector"]["output_latency_cycles"] = 4
        ops, _, _, _ = pipeline.make_model(self.config)
        cast = next(op for op in ops if op.name == "CVT_P")
        self.assertEqual(cast.ready_offset, 132)

    def test_validator_rejects_missing_or_invalid_parameters(self):
        self.config["vector"]["units"]["EXP"]["parallelism"] = 0
        with self.assertRaises(ValueError):
            pipeline.validate_config(self.config)
        self.config["vector"]["units"]["EXP"]["parallelism"] = 8
        self.config["l1"]["bandwith"] = 123
        with self.assertRaises(ValueError):
            pipeline.validate_config(self.config)

    def test_verifier_catches_periodic_resource_collision(self):
        ops, caps, ii, _, _, _ = pipeline.solve(self.config)
        broken = copy.deepcopy(ops)
        q_load = next(op for op in broken if op.name == "Q_LD")
        broken.append(copy.deepcopy(q_load))
        with self.assertRaises(AssertionError):
            pipeline.verify_schedule(broken, caps, ii)

    def test_phase_search_matches_exhaustive_reference(self):
        random = np.random.default_rng(42)
        for ii in (1, 7, 31):
            for _ in range(20):
                patterns = {"R": random.integers(0, 4, ii).astype(float)}
                occupied = {"R": random.integers(0, 4, ii).astype(float)}
                actual = pipeline.feasible_phases(patterns, occupied, {"R": 4}, ii)
                expected = [np.all(np.roll(patterns["R"], shift) + occupied["R"] <= 4)
                            for shift in range(ii)]
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
