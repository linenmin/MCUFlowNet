"""Compare the publication checkout with the recorded HPC commit, without ML dependencies.

Only explicitly listed import-module renames and docstrings are normalized.
Matching syntax trees do not prove runtime or checkpoint compatibility.
"""
import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

HISTORICAL = '75bea8028275040a4039ca48a2d34b3bdceeedfe'
RENAMES = {
    'efnas.engine.retrain_v3_trainer': 'efnas.engine.retrain_trainer',
    'efnas.engine.retrain_v2_eval_scaling': 'efnas.engine.retrain_eval_scaling',
    'efnas.engine.retrain_v2_sintel_runtime': 'efnas.engine.retrain_sintel_runtime',
    'efnas.engine.retrain_v2_evaluator': 'efnas.engine.retrain_evaluator',
    'efnas.engine.retrain_v2_resume': 'efnas.engine.retrain_resume',
    'efnas.network.fixed_arch_models_v3': 'efnas.network.fixed_arch_models',
    'efnas.network.fixed_arch_models_v2': 'efnas.network.fixed_arch_models_eval',
    'efnas.network.MultiScaleResNet_supernet_v3': 'efnas.network.multiscale_supernet',
    'efnas.network.MultiScaleResNet_supernet_v2': 'efnas.network.multiscale_supernet_eval',
    'efnas.network.MultiScaleResNet_supernet': 'efnas.network.multiscale_supernet_base',
    'efnas.nas.search_space_v3': 'efnas.nas.search_space',
}
PAIRS = [
    ('efnas/engine/retrain_trainer.py', 'efnas/engine/retrain_v3_trainer.py'),
    ('efnas/network/fixed_arch_models.py', 'efnas/network/fixed_arch_models_v3.py'),
    ('efnas/engine/retrain_eval_scaling.py', 'efnas/engine/retrain_v2_eval_scaling.py'),
    ('efnas/engine/retrain_sintel_runtime.py', 'efnas/engine/retrain_v2_sintel_runtime.py'),
    ('efnas/network/multiscale_supernet_base.py', 'efnas/network/MultiScaleResNet_supernet.py'),
    ('efnas/nas/search_space.py', 'efnas/nas/search_space_v3.py'),
] + [(p, p) for p in [
    'efnas/engine/distill_or_not_trainer.py', 'efnas/engine/distill_or_not_sintel_runtime.py',
    'efnas/engine/standalone_trainer.py', 'efnas/engine/train_step.py', 'efnas/engine/eval_step.py',
    'efnas/data/fc2_dataset.py', 'efnas/data/ft3d_dataset.py', 'efnas/data/dataloader_builder.py',
    'efnas/network/base_layers.py', 'efnas/network/decorators.py',
]]


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode('utf-8-sig').strip()


class Normalize(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        node.module = RENAMES.get(node.module, node.module)
        return node

    def visit_Import(self, node):
        for alias in node.names:
            alias.name = RENAMES.get(alias.name, alias.name)
        return node

    def generic_visit(self, node):
        node = super().generic_visit(node)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    node.body = node.body[1:]
        return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = dict(publication_head=git(root, 'rev-parse', 'HEAD'),
                  dev_head=git(args.dev, 'rev-parse', 'HEAD'), historical_commit=HISTORICAL,
                  comparison_scope='Selected files only; line endings normalized; no runtime execution',
                  allowed_import_renames=RENAMES, comparisons=[], parse_errors=[], missing_modules=[])
    pairs = [('EdgeFlowNAS/'+a, 'EdgeFlowNAS/'+b) for a,b in PAIRS]
    pairs += [('EdgeFlowNet/'+p, 'EdgeFlowNet/'+p) for p in
              ['code/misc/utils.py', 'code/misc/processor.py', 'code/misc/ImageUtils.py']]
    for published, historical in pairs:
        current = (root/published).read_text(encoding='utf-8-sig').strip()
        old = git(args.dev, 'show', HISTORICAL+':'+historical)
        same_ast = ast.dump(Normalize().visit(ast.parse(current))) == ast.dump(Normalize().visit(ast.parse(old)))
        report['comparisons'].append(dict(published=published, historical=historical,
            text_identical=current == old, normalized_ast_identical=same_ast,
            publication_sha256=hashlib.sha256(current.encode()).hexdigest(),
            historical_sha256=hashlib.sha256(old.encode()).hexdigest()))
    files = sorted((root/'EdgeFlowNAS').rglob('*.py'))
    report['python_files_parsed'] = len(files)
    for file in files:
        rel = file.relative_to(root).as_posix()
        try:
            tree = ast.parse(file.read_text(encoding='utf-8-sig'))
        except (SyntaxError, UnicodeError) as exc:
            report['parse_errors'].append(dict(file=rel, error=str(exc)))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('efnas.'):
                target = root/'EdgeFlowNAS'/node.module.replace('.', '/')
                if not target.with_suffix('.py').is_file() and not target.is_dir():
                    report['missing_modules'].append(dict(file=rel, line=node.lineno, module=node.module))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(compared=len(pairs), text_identical=sum(r['text_identical'] for r in report['comparisons']),
        normalized_ast_identical=sum(r['normalized_ast_identical'] for r in report['comparisons']),
        parsed=len(files), parse_errors=report['parse_errors'], missing_modules=report['missing_modules']), indent=2))


if __name__ == '__main__':
    main()
