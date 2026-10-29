"""Compare published originals with historical native head images diagnostically.

Run: OMP_NUM_THREADS=2 python src/wia/runtime/commands/audit_task_edit2_reflection_images.py
Only full-frame resolution normalization is applied. Descriptor matches neither
verify task feature identities nor qualify a new reset or task revision.
"""

from wia.resources import ROOT as WIA_RUNTIME_ROOT
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = WIA_RUNTIME_ROOT
OUTPUT = ROOT / 'outputs/task_edit2_20261018/original_esi_images'
CAMPAIGN = Path('/home/zhouxinyang/esi_category_conversion/core_revision_20261018')


def main():
    cv2.setNumThreads(2)
    core = {row['source']: row for row in json.loads(
        (CAMPAIGN / 'core_candidates_for_review.json').read_text())['tasks']}
    rows, thumbnails = [], []
    provenance = json.loads((OUTPUT / 'provenance.json').read_text())
    for source in provenance['records']:
        identity = source['source_identity']
        job = core[identity].get('capture', {}).get('job_id')
        path = CAMPAIGN / 'captures/jobs' / str(job) / 'attempts/01/head.png'
        row = {'source_identity': identity, 'original_image_file': source['original_image_file'],
            'historical_head_file': str(path), 'qualifies_new_revision': False}
        if not path.is_file():
            row['status'] = 'historical_head_unavailable'
            rows.append(row)
            continue
        original = np.array(Image.open(source['original_image_file']).convert('RGB'))
        native = np.array(Image.open(path).convert('RGB'))
        height, width = native.shape[:2]
        normalized = cv2.resize(original, (width, height), interpolation=cv2.INTER_AREA)
        detector = cv2.SIFT_create(nfeatures=3000)
        points_a, desc_a = detector.detectAndCompute(cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY), None)
        points_b, desc_b = detector.detectAndCompute(cv2.cvtColor(native, cv2.COLOR_RGB2GRAY), None)
        pairs = [] if desc_a is None or desc_b is None else cv2.BFMatcher().knnMatch(desc_a, desc_b, k=2)
        matches = [first for pair in pairs if len(pair) == 2 for first, second in [pair]
                   if first.distance < .75 * second.distance]
        errors = [float(np.linalg.norm((np.array(points_a[match.queryIdx].pt)
            - np.array(points_b[match.trainIdx].pt)) / [width, height])) for match in matches]
        cells = {(int(points_a[match.queryIdx].pt[0] / width * 6),
                  int(points_a[match.queryIdx].pt[1] / height * 6))
                 for match, error in zip(matches, errors) if error <= .015}
        row.update(status='diagnostic_only', source_shape=list(original.shape), native_shape=list(native.shape),
            descriptor_matches=len(matches), full_frame_normalized_error_median=float(np.median(errors)) if errors else None,
            matches_with_normalized_error_at_most_0_015=sum(error <= .015 for error in errors),
            occupied_6_by_6_cells=len(cells), feature_identity_review='pending')
        tile = Image.new('RGB', (600, 340), 'white')
        ImageDraw.Draw(tile).text((5, 2), '/'.join(identity.split('/')[1:3]) + '/' + Path(identity).name, fill='black')
        for pixels, x in ((original, 0), (native, 300)):
            picture = Image.fromarray(pixels)
            picture.thumbnail((295, 295))
            tile.paste(picture, (x, 30))
        thumbnails.append(tile)
        rows.append(row)
    canvas = Image.new('RGB', (1200, 340 * max(1, (len(thumbnails) + 1) // 2)), 'white')
    for index, tile in enumerate(thumbnails):
        canvas.paste(tile, ((index % 2) * 600, (index // 2) * 340))
    canvas.save(OUTPUT / 'historical_head_comparison.png')
    (OUTPUT / 'historical_head_audit.json').write_text(json.dumps({
        'method': 'Full-frame resolution-normalized SIFT; no fitted warp; descriptive thresholds only.',
        'new_native_experiments_run': False, 'new_revision_qualification': False, 'rows': rows}, indent=2) + '\n')
    for row in rows:
        print(row['source_identity'], row['status'], row.get('descriptor_matches'),
              row.get('matches_with_normalized_error_at_most_0_015'))


if __name__ == '__main__':
    main()
