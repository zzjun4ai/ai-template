"""Check training orchestration without installing the GPU training stack."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock


ROOT = Path(__file__).resolve().parents[1]


class TrainingTests(unittest.TestCase):
    def test_checkpoint_selection(self):
        tree = ast.parse((ROOT / "scripts/train.py").read_text(encoding="utf-8-sig"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_training")
        code = compile(ast.Module(body=[function], type_ignores=[]), "scripts/train.py", "exec")

        for resume, best in [(None, "best.ckpt"), ("last.ckpt", "best.ckpt"), ("last.ckpt", "")]:
            with self.subTest(resume=resume, best=best):
                cfg = SimpleNamespace(
                    train=SimpleNamespace(seed=3407, data="data", compile=False, trainer="trainer"),
                    model="model",
                    callbacks={}, callback_names=[], loggers={}, logger_names=[],
                    paths=SimpleNamespace(output_dir="unused"), ckpt_path=resume,
                )
                model, data = object(), object()
                trainer = Mock(callback_metrics={"test/loss": 0.5})
                trainer.checkpoint_callback.best_model_path = best
                namespace = {
                    "DictConfig": object, "Any": object, "Path": MagicMock(),
                    "L": Mock(), "OmegaConf": Mock(),
                    "instantiate": Mock(side_effect=[data, model, trainer]),
                    "instantiate_selected": Mock(return_value=[]),
                }
                exec(code, namespace)
                result = namespace["run_training"](cfg)
                trainer.fit.assert_called_once_with(model=model, datamodule=data, ckpt_path=resume)
                trainer.test.assert_called_once_with(model=model, datamodule=data, ckpt_path=best or None)
                self.assertEqual(result["best_checkpoint"], best)
                self.assertEqual(result["callback_metrics"], {"test/loss": 0.5})

    def test_base_module_explains_missing_implementation(self):
        tree = ast.parse((ROOT / "models/model.py").read_text(encoding="utf-8-sig"))
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BaseModule")
        namespace = {"pl": SimpleNamespace(LightningModule=object)}
        exec(compile(ast.Module(body=[base], type_ignores=[]), "models/model.py", "exec"), namespace)
        module = object.__new__(namespace["BaseModule"])
        module.named_parameters = lambda: iter(())
        for method, args in [
            ("forward", ()), ("training_step", (None, 0)),
            ("validation_step", (None, 0)), ("test_step", (None, 0)),
            ("predict_step", (None, 0)), ("configure_optimizers", ()),
        ]:
            with self.subTest(method=method), self.assertRaises(NotImplementedError):
                getattr(module, method)(*args)


if __name__ == "__main__":
    unittest.main()
