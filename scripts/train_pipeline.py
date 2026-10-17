#!/usr/bin/env python3
"""Clone-local, non-overwriting pipeline for the two ViT-L/14 experiments."""
import argparse
import datetime as dt
import hashlib
import importlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
QB = ROOT / 'IRRA-QueryBalanced-ViT-L14'
FROZEN = ROOT / 'IRRA-QB-ViT-L14-FrozenMultiScaleLocalAttr'
WEIGHT_NAME = 'ViT-L-14.pt'
WEIGHT_SHA256 = 'b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836'
WEIGHT_URL = ('https://openaipublic.azureedge.net/clip/models/' +
              WEIGHT_SHA256 + '/' + WEIGHT_NAME)
MIN_GPU_MEMORY_MIB = 11000
GPU_IDLE_LIMIT_MIB = 256


def stamp(prefix):
    return '%s_%s_%d_%s' % (prefix, dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S'),
                            os.getpid(), secrets.token_hex(3))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def data_root_from_args(args, fallback=None):
    supplied = getattr(args, 'data_root', None)
    if supplied:
        return supplied.expanduser().resolve()
    env_path = os.environ.get('UPAR_DATA_ROOT')
    if env_path:
        return Path(env_path).expanduser().resolve()
    if fallback is not None:
        return Path(fallback).expanduser().resolve()
    return (ROOT / 'data').resolve()


def requested_gpu_from_args(args):
    return getattr(args, 'gpu', None) or os.environ.get('UPAR_GPU_UUID')


def run_checked(command, *, cwd=None, env=None):
    print('+ ' + ' '.join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=str(cwd) if cwd else None,
                   env=env, check=True)


def check_gpu(requested=None):
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('nvidia-smi를 실행할 수 없습니다. NVIDIA 드라이버와 CUDA 환경을 확인하세요.') from error

    devices = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(',')]
        if len(fields) < 4:
            continue
        try:
            devices.append({'uuid': fields[0], 'name': fields[1],
                             'total_mib': int(fields[2]), 'used_mib': int(fields[3])})
        except ValueError:
            continue
    if not devices:
        raise RuntimeError('nvidia-smi에서 사용할 GPU를 찾지 못했습니다.')

    try:
        apps = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
             '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        occupied = {line.split(',', 1)[0].strip() for line in apps.stdout.splitlines()
                    if line.strip() and 'no running processes' not in line.lower()}
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('GPU 실행 프로세스 상태를 확인하지 못했습니다.') from error

    if requested:
        candidates = [device for device in devices if device['uuid'] == requested]
        if not candidates:
            raise RuntimeError('지정한 GPU UUID가 보이지 않습니다: ' + requested)
    else:
        candidates = devices

    for device in candidates:
        if device['total_mib'] < MIN_GPU_MEMORY_MIB:
            continue
        if device['used_mib'] > GPU_IDLE_LIMIT_MIB or device['uuid'] in occupied:
            continue
        return device

    requirement = '지정된 GPU' if requested else '사용 가능한 GPU'
    raise RuntimeError('%s가 없습니다. ViT-L/14 실험에는 비어 있는 GPU(%d MiB 이상)가 필요합니다.' %
                       (requirement, MIN_GPU_MEMORY_MIB))


def child_env(gpu_uuid, python_root=None):
    env = os.environ.copy()
    env.update({
        'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
        'CUDA_VISIBLE_DEVICES': gpu_uuid,
        'NVIDIA_TF32_OVERRIDE': '0',
        'OMP_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
        'WORLD_SIZE': '1',
    })
    if python_root is not None:
        existing = [part for part in env.get('PYTHONPATH', '').split(os.pathsep) if part]
        env['PYTHONPATH'] = os.pathsep.join([str(python_root)] + existing)
    return env


def check_cuda_python(gpu_uuid):
    code = ('import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; '
            'assert torch.cuda.device_count() == 1, "expected one visible GPU"; '
            'print("CUDA", torch.version.cuda, torch.cuda.get_device_name(0))')
    run_checked([sys.executable, '-c', code], env=child_env(gpu_uuid))


def validate_data(data_root):
    if not data_root.is_dir():
        raise FileNotFoundError('데이터 루트가 없습니다: %s (기본값은 저장소 루트의 data/)' % data_root)
    sys.path.insert(0, str(QB))
    try:
        from upar.data import UPARSplit
        train = UPARSplit.read(data_root, 'train')
        val = UPARSplit.read(data_root, 'val')
    finally:
        if sys.path and sys.path[0] == str(QB):
            sys.path.pop(0)
        sys.modules.pop('upar.data', None)
        sys.modules.pop('upar', None)
    print('Dataset valid: train=%d images, val=%d images, root=%s' %
          (len(train.images), len(val.images), data_root), flush=True)


def validate_data_layout(data_root):
    """Check the expected split files before looking up a smoke manifest."""
    if not data_root.is_dir():
        raise FileNotFoundError('데이터 루트가 없습니다: %s (기본값은 저장소 루트의 data/)' % data_root)
    required = ('queries.csv', 'gt.csv', 'ids.csv')
    missing = [data_root / 'annotations' / 'task2' / split / name
               for split in ('train', 'val') for name in required
               if not (data_root / 'annotations' / 'task2' / split / name).is_file()]
    if missing:
        raise FileNotFoundError('UPAR 데이터 구조에 필수 파일이 없습니다:\n' +
                                '\n'.join(str(path) for path in missing))


def dataset_fingerprint(data_root):
    digest = hashlib.sha256()
    for split in ('train', 'val'):
        for name in ('queries.csv', 'gt.csv', 'ids.csv'):
            path = data_root / 'annotations' / 'task2' / split / name
            digest.update((split + '/' + name + '\0').encode())
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate_dependencies():
    packages = ('torch', 'torchvision', 'numpy', 'PIL', 'ftfy', 'regex')
    versions = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
        except Exception as error:
            raise RuntimeError('의존성 import 실패: %s (%s)' % (package, error)) from error
        versions[package] = getattr(module, '__version__', 'installed')
    print('Python dependencies valid:', json.dumps(versions, sort_keys=True), flush=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def validate_project_configs():
    qb_config = load_json(QB / 'configs' / 'train_fp32_micro4.json')
    frozen_config = load_json(FROZEN / 'configs' / 'frozen_localattr_e5.json')
    if qb_config['train']['num_epoch'] != 60 or qb_config['train']['batch_size'] != 32:
        raise RuntimeError('QueryBalanced 설정이 기대한 60 Epoch / Batch 32와 다릅니다.')
    if frozen_config['train']['num_epoch'] != 5 or frozen_config['train']['batch_size'] != 32:
        raise RuntimeError('Frozen 설정이 기대한 5 Epoch / Batch 32와 다릅니다.')
    return qb_config, frozen_config


def verify_pretrained():
    path = QB / 'weights' / WEIGHT_NAME
    if not path.is_file():
        raise FileNotFoundError('CLIP weight가 없습니다. 먼저 `bash scripts/train_pipeline.sh download`를 실행하세요.')
    digest = sha256_file(path)
    if digest != WEIGHT_SHA256:
        raise RuntimeError('CLIP ViT-L/14 SHA-256 불일치: %s' % path)
    return path, digest


def static_smoke(data_root, requested_gpu=None):
    validate_dependencies()
    validate_project_configs()
    weight_path, weight_digest = verify_pretrained()
    validate_data(data_root)
    gpu = check_gpu(requested_gpu)
    check_cuda_python(gpu['uuid'])
    for script in (QB / 'train_upar.py', FROZEN / 'scripts' / 'preflight.py',
                   FROZEN / 'scripts' / 'train_frozen.py'):
        if not script.is_file():
            raise FileNotFoundError('Training command is missing: %s' % script)
    print('Smoke checks passed: weight=%s sha256=%s GPU=%s (%s)' %
          (weight_path.relative_to(ROOT), weight_digest, gpu['uuid'], gpu['name']), flush=True)
    return gpu, weight_digest


def write_smoke_manifest(data_root, gpu, weight_digest):
    smoke_dir = new_experiment_dir(QB / 'experiments', 'pipeline_smoke')
    manifest = {
        'passed': True,
        'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'data_root': str(data_root),
        'dataset_fingerprint': dataset_fingerprint(data_root),
        'gpu_uuid': gpu['uuid'],
        'gpu_name': gpu['name'],
        'weight_sha256': weight_digest,
    }
    path = smoke_dir / 'smoke_manifest.json'
    with path.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')
    return path


def find_smoke_manifest(data_root, data_hash, weight_hash, requested_gpu=None):
    matches = []
    cutoff = time.time() - 24 * 60 * 60
    for path in (QB / 'experiments').glob('pipeline_smoke_*/smoke_manifest.json'):
        try:
            manifest = load_json(path)
            mtime = path.stat().st_mtime
            if (manifest.get('passed') is True and mtime >= cutoff and
                    manifest.get('data_root') == str(data_root) and
                    manifest.get('dataset_fingerprint') == data_hash and
                    manifest.get('weight_sha256') == weight_hash and
                    (requested_gpu is None or manifest.get('gpu_uuid') == requested_gpu)):
                matches.append((mtime, path, manifest))
        except (OSError, ValueError, KeyError):
            continue
    for _, path, manifest in sorted(matches, reverse=True):
        try:
            gpu = check_gpu(manifest['gpu_uuid'])
            check_cuda_python(gpu['uuid'])
            print('Reusing passing smoke manifest:', path, flush=True)
            return gpu, manifest['weight_sha256'], str(path)
        except RuntimeError:
            if requested_gpu:
                raise
    return None


def new_experiment_dir(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / stamp(prefix)
    path.mkdir(exist_ok=False)
    return path


def new_run_path(parent, prefix):
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / stamp(prefix)
    if path.exists():
        raise FileExistsError('Refusing to reuse an existing run directory: %s' % path)
    return path


def create_querybalanced_run(data_root, requested_gpu=None):
    validate_dependencies()
    validate_project_configs()
    weight_path, weight_hash = verify_pretrained()
    validate_data_layout(data_root)
    data_hash = dataset_fingerprint(data_root)
    preflight = find_smoke_manifest(data_root, data_hash, weight_hash, requested_gpu)
    if preflight is None:
        gpu, weight_hash = static_smoke(data_root, requested_gpu)
        smoke_manifest = 'validated inline before this run'
    else:
        gpu, weight_hash, smoke_manifest = preflight
    weight_path = QB / 'weights' / WEIGHT_NAME

    run_dir = new_run_path(QB / 'experiments', 'full')
    env = child_env(gpu['uuid'], QB)
    run_checked([
        sys.executable, '-B', '-u', QB / 'train_upar.py',
        '--output', run_dir,
        '--config', QB / 'configs' / 'train_fp32_micro4.json',
        '--data-root', data_root,
        '--pretrained', weight_path,
        '--gpu-uuid', gpu['uuid'],
        '--device', 'cuda',
    ], cwd=QB, env=env)

    status = load_json(run_dir / 'status.json')
    if status.get('phase') != 'complete' or status.get('completed_epochs') != 60:
        raise RuntimeError('QueryBalanced run did not complete all 60 epochs; Frozen was not started.')
    checkpoint = run_dir / 'epoch_002.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError('QueryBalanced run has no Epoch 2 checkpoint: %s' % checkpoint)
    metadata = {
        'schema_version': 1,
        'phase': 'complete',
        'run_id': run_dir.name,
        'run_dir': str(run_dir.resolve()),
        'completed_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'completed_epochs': status['completed_epochs'],
        'data_root': str(data_root),
        'dataset_fingerprint': data_hash,
        'gpu_uuid': gpu['uuid'],
        'gpu_name': gpu['name'],
        'config': str((run_dir / 'config.json').resolve()),
        'config_sha256': sha256_file(run_dir / 'config.json'),
        'foundation_sha256': weight_hash,
        'checkpoint_epoch_2': str(checkpoint.resolve()),
        'checkpoint_epoch_2_sha256': sha256_file(checkpoint),
        'smoke_passed': True,
        'smoke_manifest': smoke_manifest,
    }
    with (run_dir / 'pipeline.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print('QueryBalanced completed: run=%s, epoch_002=%s' % (run_dir, checkpoint), flush=True)
    return metadata


def find_latest_querybalanced_run():
    runs = []
    for metadata_path in (QB / 'experiments').glob('*/pipeline.json'):
        try:
            metadata = load_json(metadata_path)
            run_dir = Path(metadata['run_dir']).resolve()
            if run_dir != metadata_path.parent.resolve() or metadata.get('phase') != 'complete':
                continue
            status = load_json(run_dir / 'status.json')
            checkpoint = run_dir / 'epoch_002.pth'
            config_path = run_dir / 'config.json'
            if (status.get('phase') != 'complete' or status.get('completed_epochs') != 60 or
                    metadata.get('completed_epochs') != 60 or not checkpoint.is_file() or
                    not config_path.is_file() or
                    not (run_dir / 'train.log').is_file() or
                    not (run_dir / 'metrics.jsonl').is_file()):
                continue
            if (Path(metadata.get('config', '')).resolve() != config_path.resolve() or
                    sha256_file(config_path) != metadata.get('config_sha256')):
                continue
            saved_config = load_json(config_path)
            if (tuple(saved_config.get('model', {}).get('img_size', ())) != (378, 126) or
                    saved_config.get('model', {}).get('stride_size') != 14 or
                    saved_config.get('train', {}).get('num_epoch') != 60):
                continue
            if sha256_file(checkpoint) != metadata.get('checkpoint_epoch_2_sha256'):
                continue
            runs.append((metadata_path.stat().st_mtime, metadata))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if not runs:
        raise RuntimeError('완료된 QueryBalanced run을 못 찾았습니다. 먼저 `querybalanced`를 실행하세요.')
    return max(runs, key=lambda item: item[0])[1]


def build_frozen_config(source, data_root, gpu_uuid):
    source_run = Path(source['run_dir']).resolve()
    source_checkpoint = Path(source['checkpoint_epoch_2']).resolve()
    if source_checkpoint != (source_run / 'epoch_002.pth').resolve():
        raise RuntimeError('QueryBalanced pipeline metadata points outside its Run directory.')
    digest = sha256_file(source_checkpoint)
    if digest != source['checkpoint_epoch_2_sha256']:
        raise RuntimeError('QueryBalanced Epoch 2 checkpoint SHA-256 changed after the run.')
    config = load_json(FROZEN / 'configs' / 'frozen_localattr_e5.json')
    config['initial_run_dir'] = str(source_run)
    config['initial_checkpoint'] = str(source_checkpoint)
    config['initial_checkpoint_sha256'] = digest
    config['data_root'] = str(data_root)
    config['expected_gpu_uuid'] = gpu_uuid
    return config, source_run, source_checkpoint, digest


def create_frozen_run(args):
    source = find_latest_querybalanced_run()
    data_root = data_root_from_args(args, fallback=source['data_root'])
    validate_dependencies()
    validate_project_configs()
    validate_data(data_root)
    if dataset_fingerprint(data_root) != source.get('dataset_fingerprint'):
        raise RuntimeError('Frozen은 QueryBalanced와 동일한 데이터 annotations로 실행해야 합니다.')
    gpu = check_gpu(requested_gpu_from_args(args))
    check_cuda_python(gpu['uuid'])

    config, source_run, source_checkpoint, digest = build_frozen_config(
        source, data_root, gpu['uuid'])

    work_dir = new_experiment_dir(FROZEN / 'experiments', 'frozen_pipeline')
    run_dir = work_dir / 'run'
    config_path = work_dir / 'config.json'
    preflight_path = work_dir / 'preflight.json'
    with config_path.open('x') as stream:
        json.dump(config, stream, indent=2)
        stream.write('\n')

    env = child_env(gpu['uuid'], FROZEN)
    run_checked([
        sys.executable, '-B', '-u', FROZEN / 'scripts' / 'preflight.py',
        '--config', config_path,
        '--device', 'cuda',
        '--batch-size', '32',
        '--report', preflight_path,
    ], cwd=FROZEN, env=env)
    preflight = load_json(preflight_path)
    if preflight.get('passed') is not True or preflight.get('preflight_batch_size') != 32:
        raise RuntimeError('Frozen CUDA Batch-32 preflight did not pass; training was not started.')
    if preflight.get('initial_checkpoint', {}).get('sha256') != digest:
        raise RuntimeError('Frozen preflight loaded a different Epoch 2 checkpoint.')

    run_checked([
        sys.executable, '-B', '-u', FROZEN / 'scripts' / 'train_frozen.py',
        '--config', config_path,
        '--run-dir', run_dir,
        '--preflight-report', preflight_path,
    ], cwd=FROZEN, env=env)
    status = load_json(run_dir / 'status.json')
    if status.get('phase') != 'complete' or status.get('completed_epochs') != 5:
        raise RuntimeError('Frozen run did not complete all five epochs.')

    metadata = {
        'schema_version': 1,
        'phase': 'complete',
        'run_id': work_dir.name,
        'run_dir': str(run_dir.resolve()),
        'source_querybalanced_run_id': source['run_id'],
        'source_querybalanced_run_dir': str(source_run),
        'source_epoch_2_checkpoint': str(source_checkpoint),
        'source_epoch_2_sha256': digest,
        'config': str(config_path.resolve()),
        'preflight': str(preflight_path.resolve()),
        'gpu_uuid': gpu['uuid'],
        'data_root': str(data_root),
        'completed_epochs': 5,
    }
    with (work_dir / 'pipeline.json').open('x') as stream:
        json.dump(metadata, stream, indent=2)
        stream.write('\n')
    print('FrozenMultiScaleLocalAttr completed: run=%s' % run_dir, flush=True)
    return metadata


def download_weight():
    target = QB / 'weights' / WEIGHT_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = sha256_file(target)
        if digest != WEIGHT_SHA256:
            raise RuntimeError('기존 파일은 공식 체크섬과 다릅니다. 덮어쓰지 않았습니다: %s' % target)
        print('Official CLIP weight already verified:', target, flush=True)
        return target

    fd, temporary_name = tempfile.mkstemp(prefix='.' + WEIGHT_NAME + '.tmp-', dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        request = Request(WEIGHT_URL, headers={'User-Agent': 'UPAR-Challenge-training-pipeline'})
        with os.fdopen(fd, 'wb') as output, urlopen(request, timeout=60) as response:
            digest = hashlib.sha256()
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
                digest.update(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != WEIGHT_SHA256:
            raise RuntimeError('다운로드한 CLIP weight의 SHA-256이 공식 체크섬과 다릅니다.')
        try:
            os.link(str(temporary), str(target))
        except FileExistsError:
            if sha256_file(target) != WEIGHT_SHA256:
                raise RuntimeError('다른 weight 파일이 동시에 생겼습니다. 덮어쓰지 않았습니다: %s' % target)
        print('Downloaded and verified official CLIP weight:', target, flush=True)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


def make_parser():
    parser = argparse.ArgumentParser(
        description='Train QueryBalanced ViT-L/14 and FrozenMultiScaleLocalAttr in sequence.')
    subparsers = parser.add_subparsers(dest='command', required=True)
    for command in ('download', 'smoke', 'querybalanced', 'frozen', 'all'):
        sub = subparsers.add_parser(command)
        sub.add_argument('--data-root', type=Path, default=None,
                         help='Dataset root; default: UPAR_DATA_ROOT or repository data/')
        sub.add_argument('--gpu', default=None,
                         help='GPU UUID; default: select one idle GPU with enough memory')
    return parser


def main():
    args = make_parser().parse_args()
    if args.command == 'download':
        download_weight()
        return 0
    if args.command == 'all':
        download_weight()
        data_root = data_root_from_args(args)
        create_querybalanced_run(data_root, requested_gpu_from_args(args))
        create_frozen_run(args)
        return 0
    if args.command == 'smoke':
        data_root = data_root_from_args(args)
        gpu, weight_hash = static_smoke(data_root, requested_gpu_from_args(args))
        manifest_path = write_smoke_manifest(data_root, gpu, weight_hash)
        print('Smoke manifest:', manifest_path, flush=True)
        return 0
    if args.command == 'querybalanced':
        data_root = data_root_from_args(args)
        create_querybalanced_run(data_root, requested_gpu_from_args(args))
        return 0
    if args.command == 'frozen':
        create_frozen_run(args)
        return 0
    raise AssertionError('Unhandled command: ' + args.command)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Interrupted by user. Existing completed checkpoints and logs were preserved.', file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print('Pipeline stopped: %s' % error, file=sys.stderr)
        sys.exit(1)
