"""针对周期回卷、流式依赖和统计并集的回归检查，在 WSL 中运行。"""
import copy
import json
import unittest
import numpy as np
import flashattention_pipeline as pipeline


class PipelineTests(unittest.TestCase):
    def test_bandwidth_contributors_preserve_identity_and_cross_period_tiles(self):
        report = {"ii_cycles":4,"window_cycles":4,
                  "operations":[
                      {"name":"A","resource_intervals":{"L1_R":[[3,5,16]]}},
                      {"name":"B","resource_intervals":{"L1_R":[[1,3,16]]}}],
                  "l1_bandwidth_intervals":{"L1_R":[{"start":0,"end":4,"bytes_per_cycle":16}]}}
        sources = pipeline.bandwidth_contributors(report,"L1_R")
        self.assertEqual(set(sources),{"A[t-1]","A[t+0]","B[t+0]"})
        np.testing.assert_array_equal(sources["A[t-1]"],[16,0,0,0])
        np.testing.assert_array_equal(sources["A[t+0]"],[0,0,0,16])
        np.testing.assert_array_equal(sources["B[t+0]"],[0,16,16,0])
        np.testing.assert_array_equal(sum(sources.values()),[16,16,16,16])

    def test_bandwidth_view_preserves_samples_and_marks_dense_regions(self):
        values = np.r_[np.full(10,96),np.tile([256,128],8),np.zeros(6)]
        self.assertEqual(pipeline.bandwidth_regions(values),[(10,26)])
        self.assertEqual(pipeline.bandwidth_regions(np.zeros(20)),[])
        self.assertEqual(pipeline.bandwidth_regions(np.tile([0,16],6)),[(0,12)])
        report = {"window_cycles":len(values), "l1_bandwidth_intervals":{"L1_R":[]}}
        edges = np.r_[0,np.flatnonzero(np.diff(values))+1,len(values)]
        report["l1_bandwidth_intervals"]["L1_R"] = [
            dict(start=int(a),end=int(b),bytes_per_cycle=float(values[a]))
            for a,b in zip(edges,edges[1:])]
        np.testing.assert_array_equal(pipeline.bandwidth_samples(report,"L1_R"),values)

    def test_mask_a_traffic_and_slot_pool_capacity(self):
        cfg = json.loads(pipeline.DEFAULT_CONFIG.with_name("flashattention_pipeline.fp16_inline.json").read_text())
        ops,caps,ii,lower,total,pools = pipeline.solve(cfg)
        named = {o.name:o for o in ops}
        self.assertEqual(named["MASK_DMA"].profiles["L1_W"].sum(),16384)
        self.assertEqual(named["MASK"].profiles["L1_R"].sum(),16384)
        self.assertEqual(named["SCALE_A"].bytes_read,32768)
        self.assertEqual(named["SCALE_A"].bytes_written,32768)
        self.assertEqual(named["ADD_A"].bytes_read,32768+16384)
        self.assertEqual(named["ADD_A"].bytes_written,32768)
        self.assertGreaterEqual(named["ADD_A"].start,named["SCALE_A"].ready)
        self.assertGreaterEqual(named["SCALE_A"].start+ii,named["ADD_A"].ready)
        self.assertEqual(total,49152+sum(p["slots"]*p["bytes_per_slot"] for p in pools.values()))
        report = pipeline.make_report(cfg,ops,caps,ii,lower,total,pools)
        self.assertLessEqual(report["l1_capacity"]["live_peak_bytes"],total)
        self.assertEqual(set(report["l1_capacity"]["intervals"]),
                         {"Q+A","K","V","mask","S","P","Y","总存活"})
        for pool in pools.values():
            a,b = pool["allocate_cycle"],pool["release_cycle"]
            self.assertGreaterEqual(a+pool["slots"]*ii,b)
            if pool["slots"]>1:
                self.assertLess(a+(pool["slots"]-1)*ii,b)
        # 独立展开跨窗口事件，检查每拍占用及总存活。
        for t in range(ii):
            expected = 49152
            for pool in pools.values():
                a,b = pool["allocate_cycle"],pool["release_cycle"]
                expected += pool["bytes_per_slot"]*sum(a+j*ii<=t<b+j*ii for j in range(-20,21))
            interval = next(v for v in report["l1_capacity"]["intervals"]["总存活"] if v["start"]<=t<v["end"])
            self.assertEqual(interval["bytes"],expected)

    def test_tensor_streaming_bytes_latency_and_final_output(self):
        cfg = json.loads(pipeline.DEFAULT_CONFIG.with_name("flashattention_pipeline.fp16_inline.json").read_text())
        ops, _, _, _ = pipeline.make_model(cfg)
        named = {o.name:o for o in ops}
        for name, read_bytes, write_bytes in (("QK",32768,8192),("PV",24576,16384)):
            op = named[name]
            self.assertEqual(op.profiles["L1_R"].sum(),read_bytes)
            self.assertEqual(op.profiles["L1_W"].sum(),write_bytes)
            self.assertEqual(op.profiles["TENSOR"].sum(),256)
            self.assertEqual(op.issue_offsets[0],1)
            self.assertLess(op.write_start,op.result_offset)
            self.assertGreater(op.read_end,op.issue_offsets[0])
            blocks = 64 if name == "QK" else 128
            self.assertEqual(op.write_start,op.issue_offsets[-blocks]+15)
            self.assertEqual(op.ready_offset,op.result_offset+1)
        cfg["tensor"]["input_latency_cycles"] = 3
        cfg["tensor"]["output_latency_cycles"] = 4
        changed, _, _, _ = pipeline.make_model(cfg)
        for op in changed:
            if op.unit == "TENSOR":
                old = named[op.name]
                self.assertEqual(op.issue_offsets[0],old.issue_offsets[0]+2)
                self.assertEqual(op.ready_offset,old.ready_offset+5)
                self.assertEqual(op.profiles["L1_R"].sum(),old.bytes_read)
                self.assertEqual(op.profiles["L1_W"].sum(),old.bytes_written)

    def test_tensor_bandwidth_limits_stream_and_preserves_bytes(self):
        self.config["tensor"]["read_gbps"] = 32
        self.config["tensor"]["write_gbps"] = 64
        ops, caps, _, _ = pipeline.make_model(self.config)
        for op in ops:
            if op.unit != "TENSOR":
                continue
            self.assertGreater(op.issue_offsets[1]-op.issue_offsets[0],1)
            self.assertLessEqual(op.profiles["TENSOR_LD"].max(),caps["TENSOR_LD"])
            self.assertLessEqual(op.profiles["TENSOR_ST"].max(),caps["TENSOR_ST"])
            self.assertAlmostEqual(op.profiles["L1_R"].sum(),op.bytes_read)
            self.assertAlmostEqual(op.profiles["L1_W"].sum(),op.bytes_written)

    def test_overview_uses_resource_union_and_real_join_readiness(self):
        config = json.loads(pipeline.DEFAULT_CONFIG.read_text())
        ops, _, ii, _, _, _ = pipeline.solve(config)
        data, _, _ = pipeline.execution_overview(ops, ii, 4*ii)
        for engine, resources in pipeline.ENGINE_RESOURCES.items():
            selected = [r for r in resources if r == "TENSOR" or r in pipeline.VECTOR_UNITS]
            if engine == "DMA":
                selected = resources
            per_resource = {r: set() for r in selected}
            for op in ops:
                for r in selected:
                    for a, b, _ in op.runs(r):
                        per_resource[r].update(range(a,b))
            expected = set().union(*per_resource.values())
            actual = data["single_tile"][engine]["primary"]
            self.assertEqual(actual["busy_cycles"], len(expected))
            self.assertEqual(actual["work_resource_cycles"], sum(map(len,per_resource.values())))
            phases = {t % ii for t in expected}
            self.assertEqual(data["steady_window"][engine]["primary"]["busy_cycles"],4*len(phases))
        vector = data["single_tile"]["Vector"]["primary"]
        self.assertGreater(vector["work_resource_cycles"],vector["busy_cycles"])
        for join in data["joins"]:
            self.assertEqual(join["dependency_gate"],max(join["input_ready"].values()))
            self.assertGreaterEqual(join["scheduled_wait_cycles"],0)

    def setUp(self):
        self.config = json.loads(pipeline.DEFAULT_CONFIG.read_text())

    def test_default_work_and_feasible_schedule(self):
        ops, caps, ii, lower, total, pools = pipeline.solve(self.config)
        # FP32 Tensor 输出的连续写流改变相位可行性，贪心调度不保证达到下界。
        self.assertEqual(lower, 520)
        self.assertGreaterEqual(ii, lower)
        self.assertLessEqual(total, 512 * 1024)
        report = pipeline.make_report(self.config, ops, caps, ii, lower, total, pools)
        expected = {"TENSOR": 512, "MUL": 193, "ADD": 320, "SUB": 65,
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
        mask = by_name["MASK"]
        self.assertEqual((mask.unit, mask.work, mask.deps), ("ADD", 4096, ("SCALE","MASK_DMA")))
        self.assertEqual(by_name["ROWMAX"].deps, ("MASK",))
        self.assertEqual(int(mask.profiles["ADD"].sum()), 64)
        self.assertNotIn("mask", pipeline.ATTENTION_EQUATIONS["SCALE"])
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
        q_load = next(op for op in broken if op.name == "QK")
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

    def test_inline_conversion_preserves_local_result_and_delays_l1_visibility(self):
        path = pipeline.DEFAULT_CONFIG.with_name("flashattention_pipeline.fp16_inline.json")
        config = pipeline.validate_config(json.loads(path.read_text()))
        ops, caps, ii, lower, _, _ = pipeline.solve(config)
        by_name = {op.name: op for op in ops}
        self.assertEqual(lower,520)
        self.assertGreaterEqual(ii,lower)
        self.assertNotIn("CVT_P", by_name)
        self.assertEqual(by_name["SCALE"].issue_offsets[0], 3)
        self.assertEqual(by_name["ADD_A"].issue_offsets[0], 4)
        self.assertEqual(by_name["SCALE"].profiles["VECTOR_LD"].sum(), 8192)
        exp = by_name["EXP_P"]
        self.assertEqual(exp.profiles["VECTOR_ST"].sum(), 8192)
        self.assertEqual(exp.write_start, 3)
        self.assertEqual(exp.ready - exp.start - exp.result_offset, 3)
        self.assertEqual(by_name["ROWSUM"].dependency_ready(exp), exp.start + exp.result_offset)
        self.assertEqual(by_name["PV"].dependency_ready(exp), exp.ready)
        pipeline.verify_schedule(ops, caps, ii)

        config["numeric"]["conversion"]["vector_load_extra_cycles"] = 1
        config["numeric"]["conversion"]["vector_store_extra_cycles"] = 1
        changed, _, _, _ = pipeline.make_model(config)
        changed_by_name = {op.name: op for op in changed}
        self.assertEqual(changed_by_name["SCALE"].issue_offsets[0], 2)
        self.assertEqual(changed_by_name["EXP_P"].write_start, 2)
        self.assertEqual(changed_by_name["EXP_P"].issue_offsets, exp.issue_offsets)

    def test_inline_dma_conversion_accounts_for_both_formats(self):
        path = pipeline.DEFAULT_CONFIG.with_name("flashattention_pipeline.fp16_inline.json")
        config = json.loads(path.read_text())
        config["numeric"]["ddr_k_dtype"] = "fp32"
        config["numeric"]["ddr_v_dtype"] = "bf16"
        ops, caps, ii, _, _, _ = pipeline.solve(pipeline.validate_config(config))
        by_name = {op.name: op for op in ops}
        k = by_name["K_DMA"]
        self.assertEqual(k.profiles["DMA_RD"].sum(), 32768)
        self.assertEqual(k.profiles["L1_W"].sum(), 16384)
        self.assertEqual(k.runs("L1_W")[0][0], k.start + 2)
        self.assertEqual(k.ready, k.runs("DMA_RD")[-1][1] + 2)
        v = by_name["V_DMA"]
        self.assertEqual(v.bytes_read, v.bytes_written)
        self.assertEqual(v.conversion["source"], "bf16")
        self.assertEqual(v.ready, v.runs("DMA_RD")[-1][1] + 2)
        pipeline.verify_schedule(ops, caps, ii)

    def test_explicit_conversion_reserves_the_cvt_unit(self):
        path = pipeline.DEFAULT_CONFIG.with_name("flashattention_pipeline.fp16_inline.json")
        config = json.loads(path.read_text())
        config["numeric"]["conversion"]["mode"] = "explicit"
        ops, _, _, _, _, _ = pipeline.solve(pipeline.validate_config(config))
        named = {op.name: op for op in ops}
        self.assertEqual({op.name for op in ops if op.unit == "CVT"}, {"CVT_S", "CVT_P", "CVT_Y"})
        self.assertEqual(named["SCALE"].deps, ("CVT_S",))
        self.assertEqual(named["PV"].deps, ("CVT_P", "V_DMA"))
        self.assertEqual(named["CVT_P"].issue_offsets[-1] + 1, 64)
        self.assertIn("CVT_Y", named["ADD_A"].deps)


if __name__ == "__main__":
    unittest.main()
