"""Lifecycle events for native workers. Run via `python -m wia bench --config FILE`."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def emit(event, **fields):
    record = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
              'gpu': os.environ.get('WIA_BENCH_GPU'), **fields}
    line = json.dumps(record, ensure_ascii=True)
    print('[lifecycle] ' + line, flush=True)
    destination = os.environ.get('WIA_BENCH_EVENTS')
    if destination:
        with Path(destination).open('a') as stream:
            stream.write(line + '\n')


def read_events(path, offset):
    """Consume complete records only, retaining partial trailing writes for next poll."""
    try:
        with Path(path).open() as stream:
            stream.seek(offset)
            records = []
            while line := stream.readline():
                if not line.endswith('\n'):
                    break
                records.append(json.loads(line))
                offset = stream.tell()
            return records, offset
    except FileNotFoundError:
        return [], offset


def format_event(record):
    details = ' '.join(f'{key}={value}' for key, value in record.items()
                       if key not in ('time', 'event') and value is not None)
    return f'[{record["time"]}] {record["event"]} {details}'
