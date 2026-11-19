#!/usr/bin/env python3
"""Recover target-free P21 coordinates from original GRID and audit kernel parity.

No train.h5 is created and no INIT/PORO or frozen-test labels are opened. All
paths are explicit. Original research sources are reused, with prepared arrays
saved separately from the query truth used only for the final parity audit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits
from wellseismic_sixtrack_pilot.manifest import atomic_json, sha256_file
from wellseismic_sixtrack_pilot.worker.reconstruction_p21 import predict_prepared


def checked_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f'Input hash mismatch: {path}')
    return actual


def run(args) -> dict:
    root = args.competition_root.resolve()
    science = root / '_pipelines/02_task_datasets/reconstruction'
    output = args.output_directory.resolve()
    if output.exists():
        raise ValueError('Use a new output directory')
    sys.path[:0] = [str(science), str(root)]
    p21 = importlib.import_module('p21_fixed_foundation_ensemble')
    p18, p17, base = p21.p18, p21.p17, p21.base
    base.ensure_no_holdout_paths([args.stage3_root, args.feature_cache, args.eclipse_zip, output])
    reference_dir = science / '_outputs/p21_fixed_foundation_ensemble'
    # Bind accepted research bytes before consuming scientific contracts.
    summary_path = reference_dir / 'summary.json'
    checked_hash(summary_path, '25a92577ca954ac1b7ca214848d24a25d0cac8a9b2ba7af5d584e4360da882b8')
    checked_hash(science / 'p21_fixed_foundation_ensemble.py', '27d3bafbb149c32b26dcbc01785bc5341bd27c45163506213dccd13c2a62166e')
    summary = json.loads(summary_path.read_text())
    reference_path = reference_dir / 'predictions.npz'
    checked_hash(reference_path, '7eb149021973362e8a9a788f2919a7775bffc702061f2e2ce44031f9ee835afb')
    feature_hash = checked_hash(args.feature_cache, summary['foundation_feature_audit']['feature_cache_sha256'])
    cache_meta = json.loads(args.feature_cache.with_suffix('.json').read_text())
    if cache_meta['encoder_weight_mode'] != 'pretrained' or cache_meta['npz_sha256'] != feature_hash:
        raise ValueError('Foundation cache identity mismatch')
    geometry = importlib.import_module('build_dataset')
    # Read exactly the GRID member. Never call load_reference_volume, which
    # reads dense PORO/INIT labels as well as geometry.
    print('Reading original GRID geometry only', file=sys.stderr, flush=True)
    with ZipFile(args.eclipse_zip) as archive:
        shape, centres, active = geometry._parse_eclipse_grid(archive)
        with archive.open(geometry.ECLIPSE_GRID_MEMBER) as stream:
            grid_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if shape != (63, 100, 108):
        raise ValueError('Original Eclipse shape mismatch')
    active = active.reshape(shape)
    reference = SimpleNamespace(active=active, **{
        key: centres[:, column].reshape(shape)
        for column, key in enumerate(('x', 'y', 'z'))
    })
    x, y, z, bounds, _ = geometry._normalise_coordinates(reference)
    accepted_bounds = json.loads((science / 'build_summary.json').read_text())['coordinate_bounds']
    if bounds != accepted_bounds:
        raise ValueError('Original coordinate bounds mismatch')
    mask = active.copy()
    mask[:, :, 72:] = False  # Original four 18-cell training i-blocks.
    indices = np.argwhere(mask).astype(np.int64)
    if len(indices) != summary['fold_loading_audit']['active_coordinate_rows']:
        raise ValueError('Original active training cell count mismatch')
    coordinates = np.column_stack([values[mask] for values in (x, y, z)]).astype(np.float64)
    tree = cKDTree(coordinates)
    print('Geometry ready; validating original five-fold indices', file=sys.stderr, flush=True)
    oof = base.load_oof_development(args.stage3_root)
    folds = []
    kriging_coordinates = {}
    matches = []
    for accepted in summary['fold_loading_audit']['folds']:
        fold_id = accepted['fold_id']
        directory = args.stage3_root / 'cache/conditional' / f'fold_{fold_id:02d}'
        for name, filename in (('point_train_npz', 'point_train.npz'),
                               ('point_validation_npz', 'point_validation.npz'),
                               ('preprocess_stats', 'preprocess_stats.json')):
            checked_hash(directory / filename, accepted['source_hashes'][name])
        stats = json.loads((directory / 'preprocess_stats.json').read_text())['stats']
        with np.load(directory / 'point_train.npz', allow_pickle=False) as data:
            train_raw = p17._raw_from_preprocessed(data['input_values'], stats)
            train_target = np.asarray(data['target_values'], dtype=np.float64)
            train_kriging = np.asarray(data['input_values'], dtype=np.float64)[:, -3:]
        with np.load(directory / 'point_validation.npz', allow_pickle=False) as data:
            query_raw = p17._raw_from_preprocessed(data['input_values'], stats)
            query_kriging = np.asarray(data['input_values'], dtype=np.float64)[:, -3:]
        kriging_coordinates[fold_id] = (train_kriging, query_kriging)
        train_indices, train_error = p17._match_coordinates(tree, indices, train_raw[:, 3:6])
        query_indices, query_error = p17._match_coordinates(tree, indices, query_raw[:, 3:6])
        for values, role in ((train_indices, 'train_indices_sha256'), (query_indices, 'validation_indices_sha256')):
            if p17._array_sha256(values) != accepted[role]:
                raise ValueError(f'Original fold {fold_id} {role} mismatch')
        np.testing.assert_array_equal(query_indices, oof.indices_kji[oof.fold_ids == fold_id])
        folds.append(p17.FoldSamples(fold_id, train_target, train_raw, query_raw,
                                    train_indices, query_indices, accepted['source_hashes']))
        matches.append({'fold_id': fold_id, 'train_indices_sha256': accepted['train_indices_sha256'],
                        'query_indices_sha256': accepted['validation_indices_sha256'],
                        'maximum_coordinate_match_distance': max(train_error, query_error)})
    with np.load(args.feature_cache, allow_pickle=False) as cache:
        requested = np.asarray(cache['indices_kji'], dtype=np.int64)
        foundation = np.asarray(cache['features'], dtype=np.float64)
    np.testing.assert_array_equal(requested, p17._unique_indices(folds))
    with threadpool_limits(limits=1):
        prepared, audits = p18._prepare_fold_metrics(folds=folds, requested_indices=requested,
                                                    foundation_features=foundation)
        prediction = np.full(len(oof.target), np.nan)
        output.mkdir(parents=True)
        payloads = []
        feature_inputs = []
        for row in prepared:
            fold = row['fold']
            selected = oof.fold_ids == fold.fold_id
            inputs = {key.replace('validation_', 'query_'): row[key]
                      for key in ('train_coordinate', 'validation_coordinate', 'train_seismic',
                                  'validation_seismic', 'train_latent', 'validation_latent')}
            inputs.update(train_target=fold.train_target, pykrige_prediction=oof.baseline[selected])
            path = output / f'fold_{fold.fold_id}_prepared.npz'
            np.savez_compressed(path, **inputs)
            values, members = predict_prepared(inputs)
            prediction[selected] = values
            payloads.append({'path': path.name, 'sha256': sha256_file(path), 'query_targets_present': False})
            if args.include_feature_inputs:
                feature_path = output / f'fold_{fold.fold_id}_feature_input.npz'
                train_kriging, query_kriging = kriging_coordinates[fold.fold_id]
                np.savez_compressed(feature_path,
                    train_raw_features=fold.train_raw_features, query_raw_features=fold.validation_raw_features,
                    train_foundation=p17._feature_rows(requested, foundation, fold.train_indices_kji),
                    query_foundation=p17._feature_rows(requested, foundation, fold.validation_indices_kji),
                    train_target=fold.train_target, train_kriging_coordinate=train_kriging,
                    query_kriging_coordinate=query_kriging)
                feature_inputs.append({'fold_id': fold.fold_id, 'path': feature_path.name,
                                       'sha256': sha256_file(feature_path), 'query_targets_present': False})
    # Query truth is accessed here only for evaluation, after inference.
    with np.load(reference_path, allow_pickle=False) as reference:
        np.testing.assert_array_equal(reference['indices_kji'], oof.indices_kji)
        expected = reference['candidate_prediction']
    difference = float(np.max(np.abs(prediction - expected)))
    np.testing.assert_allclose(prediction, expected, rtol=0, atol=1e-12)
    result = {'status': 'passed', 'scope': 'original development OOF kernel parity; not new-area or platform runtime',
              'rows': len(prediction), 'folds': matches, 'maximum_absolute_prediction_difference': difference,
              'rmse': float(np.sqrt(np.mean((prediction - oof.target) ** 2))),
              'coordinate_source': 'original Eclipse GRID via original parser and normalization',
              'grid_member_sha256': grid_hash, 'feature_cache_sha256': feature_hash,
              'raw_poro_or_init_labels_read': False, 'train_h5_recreated': False,
              'frozen_test_read': False, 'prepared_payloads': payloads, 'transform_audits': audits,
              'feature_inputs': feature_inputs,
              'sources': {name: sha256_file(science / name) for name in
                          ('build_dataset.py', 'p17_foundation_geostatistics.py',
                           'p18_anisotropic_foundation_geostatistics.py', 'p21_fixed_foundation_ensemble.py')}}
    atomic_json(output / 'parity.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('competition-root', 'stage3-root', 'feature-cache', 'eclipse-zip', 'output-directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--include-feature-inputs', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False))


if __name__ == '__main__':
    main()
