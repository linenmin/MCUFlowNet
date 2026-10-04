"""Compile existing random-arm INT8 exports; preserve every Vela verdict.

The configured cache budget is a compiler input, not the board's usable SRAM.
Compilation never changes the model, substitutes a graph, or proves board FPS.
"""
import argparse
import configparser
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    if sha(args.config) != args.config_sha256.lower():
        raise ValueError('Grove configuration SHA256 differs')
    config = configparser.ConfigParser()
    config.read(args.config, encoding='utf-8')
    system = dict(config['System_Config.Grove_Sys_Config'])
    memory = dict(config['Memory_Mode.Grove_Mem_Mode'])
    if (int(memory['arena_cache_size']) != 1468006
            or float(system['offchipflash_clock_scale']) != 0.015625
            or float(system['core_clock']) != 400e6):
        raise ValueError('Expected the previously checked 1.4 MiB/400 MHz Grove configuration')
    cases = [case for case in json.loads((args.audit/'cases.json').read_text(encoding='utf-8'))
             if case['geometry'] == 'random']
    expected = {'random-edge-208','random-S-208','random-L-208','random-S-224','random-L-224'}
    if len(cases) != 5 or {case['id'] for case in cases} != expected:
        raise ValueError('Expected the five random-arm exports')
    root = args.audit/'vela'
    root.mkdir(parents=True, exist_ok=False)
    version = subprocess.check_output([sys.executable,'-m','ethosu.vela','--version'],text=True).strip()
    report = dict(status='running', started=datetime.now(timezone.utc).isoformat(),
        python=sys.executable, vela_version=version, command=sys.argv,
        script_sha256=sha(__file__), configuration=str(args.config.resolve()),
        configuration_sha256=sha(args.config), system=system, memory=memory,
        accelerator='ethos-u55-64', optimization='Size',
        scope='Compile unchanged four-channel-accumulation INT8 exports; pre-Vela scores retained',
        limitation='Configured cache budget is not physical SRAM or firmware arena; no board validation or measured FPS',
        results=[])
    save(root/'summary.json',report)
    for case in cases:
        started = time.monotonic()
        destination = root/case['id']/'Size'
        destination.mkdir(parents=True, exist_ok=False)
        source = args.audit/'exports'/case['id']/'model_int8.tflite'
        exported_path = source.parent/'export.json'
        item = dict(case=case, status='running', input=str(source.resolve()),
                    export_report=str(exported_path.resolve()), output=str(destination.resolve()))
        try:
            exported = json.loads(exported_path.read_text(encoding='utf-8'))
            if (exported['status'] != 'passed' or exported['input_hw'] != case['hw']
                    or exported['model'] != case['model']):
                raise ValueError('Export identity, shape or acceptance differs')
            source_sha = sha(source)
            if source_sha != exported['exports']['int8']['sha256']:
                raise ValueError('Exported INT8 file hash differs')
            item.update(input_sha256=source_sha, export_report_sha256=sha(exported_path),
                        checkpoint_sha256=exported['checkpoint_sha256'])
            command = [sys.executable,'-m','ethosu.vela',str(source),
                '--accelerator-config','ethos-u55-64','--config',str(args.config),
                '--system-config','Grove_Sys_Config','--memory-mode','Grove_Mem_Mode',
                '--optimise','Size','--output-dir',str(destination),
                '--verbose-performance','--show-cpu-operations']
            item['command'] = command
            log = destination/'compile.log'
            with log.open('w',encoding='utf-8') as file:
                code = subprocess.call(command,stdout=file,stderr=subprocess.STDOUT)
            text = log.read_text(encoding='utf-8',errors='replace')
            item.update(returncode=code, log=str(log.resolve()), log_sha256=sha(log))
            match = re.search(r'CPU operators\s*=\s*(\d+)',text)
            item['cpu_operators'] = int(match[1]) if match else None
            summaries = list(destination.glob('*_summary_*.csv'))
            if len(summaries) == 1:
                with summaries[0].open(newline='',encoding='utf-8') as file:
                    values = list(csv.DictReader(file))
                if len(values) != 1:
                    raise ValueError('Unexpected Vela summary row count')
                item['vela_summary'] = values[0]
                item['summary_csv'] = str(summaries[0].resolve())
                item['summary_csv_sha256'] = sha(summaries[0])
                peak = float(values[0]['sram_memory_used'])
                item.update(sram_peak_kib=peak, sram_peak_bytes=peak*1024,
                    within_configured_cache_budget=peak*1024 <= int(memory['arena_cache_size']),
                    estimated_fps=float(values[0]['inferences_per_second']))
            artifacts = list(destination.glob('*_vela.tflite'))
            if len(artifacts) == 1:
                item.update(compiled_model=str(artifacts[0].resolve()),
                            compiled_sha256=sha(artifacts[0]), compiled_bytes=artifacts[0].stat().st_size)
            if sha(source) != source_sha or sha(args.config) != args.config_sha256.lower():
                raise AssertionError('Source model or configuration changed during compilation')
            if code != 0 or len(summaries) != 1 or len(artifacts) != 1 or match is None:
                raise RuntimeError('Vela failed or did not return complete compilation evidence')
            item['status'] = 'compiled_not_board_validated'
            item['input_unchanged'] = True
        except Exception as error:
            item.update(status='failed',error=repr(error))
        finally:
            item['seconds'] = time.monotonic()-started
            save(destination/'result.json',item)
            report['results'].append(item)
            save(root/'summary.json',report)
            print(json.dumps({key:item.get(key) for key in ('case','status','sram_peak_kib','cpu_operators','estimated_fps','error')}),flush=True)
    failed = sum(item['status'] == 'failed' for item in report['results'])
    report.update(status='completed_with_failures' if failed else 'completed_not_board_validated',
                  failed=failed, completed=datetime.now(timezone.utc).isoformat())
    save(root/'summary.json',report)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
