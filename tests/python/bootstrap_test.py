"""Bootstrap workflows exercised without hardware, downloads or system changes."""
import contextlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import cluster_doctor
import setup
import site_env


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "deploy").mkdir()
        self.template = self.root / "deploy/cluster_test_w1.example.json"
        self.template.write_text(json.dumps({"model": "org/model", "world_size": 1, "engine": {"mtp": True}}))
        self.config = self.root / "deploy/cluster_test_w1.json"
        self.site = self.root / ".env"
        for patcher in (patch.object(setup, "ROOT", self.root), patch.object(site_env, "ROOT", self.root),
                        patch.dict(os.environ, {"PATH": os.environ["PATH"]}, clear=True),
                        patch.object(sys.stdin, "isatty", return_value=False)):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.output = io.StringIO()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(self.output))
        self.args = ["--non-interactive", "--template", str(self.template), "--env-file", str(self.site)]

    def fake_prerequisites(self, failed=0):
        self.stack.enter_context(patch.object(setup, "check_ssh"))
        self.stack.enter_context(patch.object(setup.cluster_doctor, "require_head"))
        self.stack.enter_context(patch.object(setup, "configure_roce", return_value={}))
        self.stack.enter_context(patch.object(setup, "build_checks", return_value=failed))
        self.stack.enter_context(patch.object(cluster_doctor, "check_cluster", return_value=0))

    def test_fresh_configuration_and_repeat_preserve_tuning_and_credentials(self):
        self.site.write_text("# private site\nHF_TOKEN='secret-do-not-print'\nUNRELATED=$(do-not-run)\n")
        self.site.chmod(0o600)
        args = [*self.args, "--configure-only", "--cache-dir", "~/models/hub"]
        self.assertEqual(setup.main(args), 0)
        values = site_env.read_env(self.site)
        self.assertEqual(values["DGPP_NODES"], "127.0.0.1")
        self.assertEqual(values["DGPP_CLUSTER_CONFIG"], str(self.config))
        self.assertEqual(values["HF_HUB_CACHE"], "~/models/hub")
        tuned = self.config.read_text().replace('"mtp": true', '"mtp": false')
        self.config.write_text(tuned)
        original = self.site.read_bytes()
        self.assertEqual(setup.main(["--non-interactive", "--configure-only", "--env-file", str(self.site)]), 0)
        self.assertEqual(self.site.read_bytes(), original)
        self.assertEqual(self.config.read_text(), tuned)
        self.assertEqual(stat.S_IMODE(self.site.stat().st_mode), 0o600)
        self.assertIn("HF_TOKEN='secret-do-not-print'", self.site.read_text())
        self.assertNotIn("secret-do-not-print", self.output.getvalue())

    def test_invalid_nodes_or_override_does_not_write_configuration(self):
        # "head head" is no longer invalid — a host listed twice stacks two
        # ranks on one machine (test below); only shell syntax and a node
        # count short of world_size are refused.
        for nodes in ("$(touch marker)", "head"):
            self.template.write_text(json.dumps({"model": "org/model", "world_size": 2}))
            with self.assertRaises(ValueError):
                setup.main([*self.args, "--configure-only", "--nodes", nodes])
            self.assertFalse(self.site.exists())
            self.assertFalse(self.config.exists())

    def test_repeated_host_stacks_ranks_and_configures(self):
        self.template.write_text(json.dumps({"model": "org/model", "world_size": 2}))
        self.assertEqual(setup.main([*self.args, "--configure-only", "--nodes", "head head"]), 0)
        values = site_env.settings({"DGPP_ENV_FILE": str(self.site)})
        self.assertEqual(site_env.site_nodes(values), ["head", "head"])

    def test_exported_conflict_is_reported_before_writes(self):
        with patch.dict(os.environ, {"DGPP_NODES": "head"}):
            with self.assertRaisesRegex(ValueError, "unset"):
                setup.main([*self.args, "--configure-only", "--nodes", "127.0.0.1"])
        self.assertFalse(self.site.exists())
        self.assertFalse(self.config.exists())

    def test_exported_input_is_saved_even_when_file_does_not_exist(self):
        with patch.dict(os.environ, {"DGPP_NODES": "127.0.0.1"}):
            setup.main([*self.args, "--configure-only", "--nodes", "127.0.0.1"])
        self.assertEqual(site_env.read_env(self.site)["DGPP_NODES"], "127.0.0.1")

    def test_exported_relative_config_can_select_the_same_deployment(self):
        with patch.dict(os.environ, {"DGPP_CLUSTER_CONFIG": "deploy/cluster_test_w1.json"}):
            setup.main([*self.args, "--configure-only"])
        self.assertEqual(site_env.read_env(self.site)["DGPP_CLUSTER_CONFIG"], str(self.config))

    def test_conflicting_existing_deployment_is_never_overwritten(self):
        self.config.write_text(json.dumps({"model": "org/other", "world_size": 1}))
        before = self.config.read_bytes()
        with self.assertRaisesRegex(ValueError, "different deployment"):
            setup.main([*self.args, "--configure-only"])
        self.assertEqual(self.config.read_bytes(), before)
        self.assertFalse(self.site.exists())

    def test_read_only_check_creates_no_files_and_runs_no_preparation(self):
        self.fake_prerequisites()
        with patch.object(setup, "run") as run, patch.object(setup, "prepare_model") as model:
            self.assertEqual(setup.main([*self.args, "--check", "--model-action", "verify"]), 0)
        run.assert_not_called()
        model.assert_not_called()
        self.assertFalse(self.config.exists())
        self.assertFalse(self.site.exists())
        self.assertIn("readiness have not been checked", self.output.getvalue())

    def test_fresh_interactive_run_selects_template_and_saves_prompt_defaults(self):
        self.fake_prerequisites()
        with patch.object(sys.stdin, "isatty", return_value=True), \
                patch.object(setup, "ask", side_effect=lambda label, default="": default), \
                patch.object(setup, "run") as run, patch.object(setup, "install_packages") as install:
            self.assertEqual(setup.main(["--env-file", str(self.site), "--model-action", "verify"]), 0)
        install.assert_not_called()
        values = site_env.read_env(self.site)
        self.assertEqual(values["DGPP_NODES"], "127.0.0.1")
        self.assertEqual(values["DGPP_HTTP_PORT"], "18080")
        self.assertEqual(values["HF_HUB_CACHE"], "~/.cache/huggingface/hub")
        self.assertEqual(values["DGPP_RESIDENT_CACHE_DIR"], "~/.cache/dgpp/resident")
        self.assertFalse(any("up" in call.args[0] for call in run.call_args_list))

    def test_failed_prerequisites_do_not_build_download_or_start(self):
        self.fake_prerequisites(failed=1)
        with patch.object(setup, "run") as run, patch.object(setup, "prepare_model") as model:
            with self.assertRaisesRegex(ValueError, "prerequisites failed"):
                setup.main([*self.args, "--start", "--model-action", "verify"])
        run.assert_not_called()
        model.assert_not_called()
        self.assertNotIn("Setup complete", self.output.getvalue())

    def test_full_workflow_builds_then_syncs_then_checks_then_starts(self):
        self.fake_prerequisites()
        calls = []
        with patch.object(setup, "run", side_effect=lambda command, **kw: calls.append([str(c) for c in command])):
            self.assertEqual(setup.main([*self.args, "--start", "--model-action", "sync"]), 0)
        self.assertEqual(calls[0][:3], ["cmake", "--preset", "release"])
        self.assertIn("-DDGPP_ENABLE_IBV=ON", calls[0])
        self.assertEqual(calls[1][:2], ["cmake", "--build"])
        self.assertTrue(calls[2][1].endswith("download_model.py"))
        self.assertIn("--sync-only", calls[2])
        self.assertEqual(calls[3][2], "doctor")
        self.assertEqual(calls[4][2], "up")
        self.assertNotIn("--replace", calls[4])
        self.assertFalse((self.root / ".venv").exists())

    def test_failed_final_preflight_does_not_start_or_claim_success(self):
        self.fake_prerequisites()
        def command(cmd, **kwargs):
            if "doctor" in cmd:
                raise RuntimeError("preflight failed")
        with patch.object(setup, "run", side_effect=command) as run:
            with self.assertRaisesRegex(RuntimeError, "preflight failed"):
                setup.main([*self.args, "--start", "--skip-build", "--model-action", "verify"])
        self.assertFalse(any("up" in call.args[0] for call in run.call_args_list))
        self.assertNotIn("Setup complete", self.output.getvalue())

    def test_existing_checkpoint_auto_syncs_without_hub_or_venv(self):
        cfg = {"model": "org/model", "node_env": [{}]}
        with patch.object(cluster_doctor, "cached_snapshot", return_value=self.root), \
                patch.object(cluster_doctor, "checkpoint_size", return_value=42):
            self.assertEqual(setup.model_action("auto", cfg), "sync")
        with patch.object(cluster_doctor, "cached_snapshot", side_effect=ValueError("missing")):
            self.assertEqual(setup.model_action("auto", cfg), "download")
        with patch.object(setup, "run") as run:
            setup.prepare_model("verify", self.config)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--verify-only", run.call_args.args[0])

    def test_download_prepares_venv_and_requirements_before_downloader(self):
        with patch.object(setup, "run") as run:
            setup.prepare_model("download", self.config)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][1:3], ["-m", "venv"])
        self.assertEqual(commands[1][1:4], ["-m", "pip", "install"])
        self.assertTrue(str(commands[2][1]).endswith("download_model.py"))

    def test_ssh_failure_has_recovery_and_does_not_disable_host_verification(self):
        cfg = {"nodes": ["head", "peer"], "ssh_user": "ops"}
        with patch.object(setup.subprocess, "run", return_value=Mock(returncode=255, stderr="Host key verification failed")) as run:
            with self.assertRaisesRegex(ValueError, "SSH access"):
                setup.check_ssh(cfg)
        self.assertIn("StrictHostKeyChecking=yes", run.call_args.args[0])
        # The probe never reads the terminal: queued answers stay for the prompts.
        self.assertIs(run.call_args.kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertIn("ssh-copy-id ops@peer", self.output.getvalue())
        self.assertIn("fingerprint", self.output.getvalue())

    def test_unattended_package_install_never_requests_a_sudo_password_or_peer_build_tools(self):
        cfg = {"nodes": ["head", "peer"], "ssh_user": "ops"}
        with patch.object(setup, "run") as run, patch.object(setup, "check_ssh") as ssh, \
                patch.object(setup.os, "geteuid", return_value=1000), \
                patch.object(setup.shutil, "which", return_value="/usr/bin/apt-get"):
            setup.install_packages(cfg, interactive=False)
        ssh.assert_called_once_with(cfg)
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["sudo", "-n"])
        self.assertIn("build-essential", commands[1])
        self.assertEqual(commands[2][0], "ssh")
        self.assertNotIn("-t", commands[2])
        self.assertIn("sudo -n", commands[2][-1])
        self.assertIn("DEBIAN_FRONTEND=noninteractive", commands[2][-1])
        self.assertNotIn("build-essential", commands[2][-1])

    def test_roce_choices_keep_rank_order_and_unrelated_node_settings(self):
        values = {**site_env.DEFAULTS, "DGPP_NODES": "head peer", "DGPP_NODE_OVERRIDES":
                  json.dumps({"peer": {"HF_HUB_CACHE": "/disk/hub"}})}
        cfg = {"nodes": ["head", "peer"], "node_env": site_env.node_environments(values)}
        inventories = {host: [{"device": nic, "usable": True, "gids": [{"index": 3}]}]
                       for host, nic in (("head", "nic0"), ("peer", "nic2"))}
        report = {"inventories": inventories, "selections": {}, "errors": {}}
        with patch.object(setup.discover_roce, "inspect_cluster", return_value=report), \
                patch.object(setup.discover_roce, "print_report"), \
                patch.object(setup, "ask", side_effect=["nic0", "3", "nic2", "3"]):
            updates = setup.configure_roce(cfg, values, True)
        overrides = json.loads(updates["DGPP_NODE_OVERRIDES"])
        self.assertEqual(overrides["peer"]["HF_HUB_CACHE"], "/disk/hub")
        self.assertEqual(overrides["head"]["DGPP_ROCE_DEVICES"], "nic0")
        self.assertEqual(overrides["peer"]["DGPP_ROCE_DEVICES"], "nic2")
        setup.save_env(self.site, updates)
        self.assertEqual(site_env.read_env(self.site)["DGPP_NODE_OVERRIDES"], updates["DGPP_NODE_OVERRIDES"])

    def test_preparation_probe_does_not_require_checkpoint_or_binary(self):
        spec = {"rank": 0, "nodes": ["127.0.0.1"], "model": "org/model", "env": {"HF_HUB_CACHE": str(self.root)},
                "paths": {}, "http_bind": "127.0.0.1", "ports": {"http": 0}, "binary": None, "preparing": True}
        with patch.object(cluster_doctor, "cached_snapshot") as snapshot, \
                patch.object(cluster_doctor, "command", return_value=Mock(returncode=0, stdout="", stderr="")):
            report = cluster_doctor.probe(spec)
        snapshot.assert_not_called()
        checks = {item["check"] for item in report["checks"]}
        self.assertNotIn("checkpoint", checks)
        self.assertIn("model cache", checks)
        self.assertIn("runtime libraries", checks)
        spec["preparing"] = False
        with patch.object(cluster_doctor, "cached_snapshot", side_effect=ValueError("missing")), \
                patch.object(cluster_doctor, "command", return_value=Mock(returncode=0, stdout="", stderr="")):
            report = cluster_doctor.probe(spec)
        self.assertTrue(any(item["check"] == "checkpoint" and item["status"] == "fail" for item in report["checks"]))

    @unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
    def test_read_only_model_cache_blocks_preparation_but_not_serving(self):
        # `dgpp-cluster up` runs this probe: a read-only checkpoint store must
        # not refuse a start. Preparation (download / sync) does need to write.
        store = self.root / "models"
        store.mkdir()
        store.chmod(0o500)
        self.addCleanup(store.chmod, 0o700)
        spec = {"rank": 0, "nodes": ["127.0.0.1"], "model": "org/model", "env": {"HF_HUB_CACHE": str(store)},
                "paths": {}, "http_bind": "127.0.0.1", "ports": {"http": 0}, "binary": None, "preparing": False}
        def cache_check():
            with patch.object(cluster_doctor, "cached_snapshot", side_effect=ValueError("missing")), \
                    patch.object(cluster_doctor, "command", return_value=Mock(returncode=0, stdout="", stderr="")):
                report = cluster_doctor.probe(spec)
            return next(item for item in report["checks"] if item["check"] == "model cache")
        self.assertEqual(cache_check()["status"], "ok")
        spec["preparing"] = True
        check = cache_check()
        self.assertEqual(check["status"], "fail")
        self.assertIn("not writable", check["detail"])

    def test_dependency_checks_detect_old_cmake_and_cuda_and_missing_headers(self):
        def probe(command, **kwargs):
            text = "cmake version 3.24.1" if command[0] == "cmake" else "release 12.9, V12.9"
            return Mock(returncode=1 if "infiniband" in kwargs.get("input", "") else 0, stdout=text, stderr="")
        with patch.object(setup.subprocess, "run", side_effect=probe), patch.object(setup, "cuda_compiler", return_value=["nvcc"]):
            self.assertEqual(setup.build_checks(False), 3)
        self.assertIn("install libibverbs-dev", self.output.getvalue())
        self.assertIn("CUDACXX", self.output.getvalue())

    def test_old_interpreter_gets_the_version_message_not_a_traceback(self):
        # The gate runs before the project imports, and the file itself must
        # parse on the interpreters it turns away.
        text = (ROOT / "scripts/setup.py").read_text()
        import ast
        ast.parse(text, feature_version=(3, 7))
        self.assertLess(text.index("sys.version_info < (3, 10)"), text.index("import cluster_doctor"))
        probe = ("import runpy, sys\nsys.version_info = (3, 9, 0, 'final', 0)\n"
                 "runpy.run_path(%r, run_name='setup_probe')\n" % str(ROOT / "scripts/setup.py"))
        result = subprocess.run([sys.executable, "-c", probe], text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("Python 3.10+", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_wrapper_has_no_inline_python(self):
        text = (ROOT / "scripts/setup.sh").read_text()
        self.assertNotRegex(text, r"\bpython[\d.]*\s+(?:-[cu]\b|-(?=\s|$)|<<)")

    def test_wrapper_runs_from_another_directory(self):
        result = subprocess.run([str(ROOT / "scripts/setup.sh"), "--list-templates"], cwd=self.root,
                                text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("cluster_qwen-3.8-flash-next_nvfp4_w1.example.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
