#!/usr/bin/env python3
"""Build a disk-backed reference index and review offline face hard cases.

Dependencies: pyarrow (index command), Pillow (review command).
No torch, GPU, model inference, or embedding-vector reads required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import random
import re
import sqlite3
import struct
import sys
import uuid
from collections import OrderedDict
from pathlib import Path

VERSION = 1
MAGIC = 0xCED7230A
PREFIX = re.compile(r"^[A-Za-z0-9_.-]+$")


def prefix_name(value):
    value = str(value)
    if not PREFIX.fullmatch(value) or value in ('.', '..'):
        raise ValueError(f'Invalid RecordIO prefix: {value!r}')
    return value


def csv_rows(path, columns):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        for line, row in enumerate(reader, 2):
            yield line, row


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise ValueError(f'Expected nonnegative integer, got {value!r}')
    return result


def score_value(value):
    result = float(value)
    if not math.isfinite(result) or not -1.00001 <= result <= 1.00001:
        raise ValueError(f'Invalid cosine score: {value!r}')
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def batches(rows, size=10000):
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def open_db(path):
    conn = sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True)
    conn.row_factory = sqlite3.Row
    version = conn.execute("SELECT value FROM meta WHERE key='version'").fetchone()
    if not version or int(json.loads(version[0])) != VERSION:
        conn.close()
        raise ValueError('Unsupported review-index version')
    return conn


def build_index(args):
    import pyarrow.dataset as pads

    embed_dir = Path(args.embeddings_dir).resolve()
    rec_dir = Path(args.rec_dir).resolve()
    hard_dir = Path(args.hard_dir).resolve()
    output = Path(args.db).resolve()
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new --db path')
    files = sorted(embed_dir.rglob('*.parquet'))
    if not files:
        raise ValueError(f'No Parquet files under {embed_dir}')
    # Explicit files exclude root CSVs/markers; hive recreates file_prefix column.
    dataset = pads.dataset([str(p) for p in files], format='parquet',
                           partitioning='hive', partition_base_dir=str(embed_dir))
    required = {'identity', 'rec_idx', 'file_prefix'}
    if required - set(dataset.schema.names):
        raise ValueError(f'Parquet missing metadata columns: {required-set(dataset.schema.names)}')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.building-{uuid.uuid4().hex}')
    conn = sqlite3.connect(temporary)
    source_ids = {}
    total = 0
    try:
        conn.executescript('''
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=NORMAL;
            PRAGMA temp_store=FILE;
            PRAGMA cache_size=-65536;
            CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE sources(id INTEGER PRIMARY KEY,prefix TEXT UNIQUE NOT NULL);
            CREATE TABLE images(source_id INTEGER NOT NULL,rec_idx INTEGER NOT NULL,
                identity INTEGER NOT NULL,PRIMARY KEY(source_id,rec_idx)) WITHOUT ROWID;
            CREATE TABLE hard_neg(a INTEGER NOT NULL,b INTEGER NOT NULL,score REAL NOT NULL,
                PRIMARY KEY(a,b)) WITHOUT ROWID;
            CREATE TABLE hard_pos(source_id INTEGER NOT NULL,rec_idx INTEGER NOT NULL,
                label INTEGER NOT NULL,score REAL NOT NULL,
                PRIMARY KEY(source_id,rec_idx)) WITHOUT ROWID;
        ''')
        scanner = dataset.scanner(columns=['identity', 'rec_idx', 'file_prefix'],
                                  batch_size=args.batch_size, use_threads=False)
        for batch in scanner.to_batches():
            columns = batch.to_pydict()
            rows = []
            for identity, rec_idx, prefix in zip(columns['identity'], columns['rec_idx'],
                                                  columns['file_prefix']):
                if None in (identity, rec_idx, prefix):
                    raise ValueError('Null identity/rec_idx/file_prefix in Parquet')
                prefix = prefix_name(prefix)
                if prefix not in source_ids:
                    for suffix in ('.rec', '.idx'):
                        if not (rec_dir / (prefix + suffix)).is_file():
                            raise FileNotFoundError(rec_dir / (prefix + suffix))
                    source_ids[prefix] = len(source_ids) + 1
                    conn.execute('INSERT INTO sources VALUES(?,?)', (source_ids[prefix], prefix))
                rows.append((source_ids[prefix], nonnegative(rec_idx), nonnegative(identity)))
            # Exact duplicates are allowed; conflicting labels fail NOT NULL.
            conn.executemany('''INSERT INTO images VALUES(?,?,?)
                ON CONFLICT(source_id,rec_idx) DO UPDATE SET identity=
                CASE WHEN images.identity=excluded.identity THEN images.identity ELSE NULL END''', rows)
            conn.commit()
            total += len(rows)
            print(f'Indexed metadata rows: {total:,}', flush=True)
        print('Building identity lookup index (on disk)...', flush=True)
        conn.execute('CREATE INDEX images_identity ON images(identity,source_id,rec_idx)')
        conn.commit()

        def negative_rows():
            for line, row in csv_rows(hard_dir / 'hard_class_neighbors.csv',
                                      ['class_id', 'neighbor_class_id', 'score']):
                a, b = nonnegative(row['class_id']), nonnegative(row['neighbor_class_id'])
                if a == b:
                    raise ValueError(f'hard_class_neighbors.csv:{line}: self-neighbor {a}')
                yield min(a, b), max(a, b), score_value(row['score'])
        for batch in batches(negative_rows()):
            conn.executemany('''INSERT INTO hard_neg VALUES(?,?,?) ON CONFLICT(a,b)
                DO UPDATE SET score=MAX(hard_neg.score,excluded.score)''', batch)
            conn.commit()

        def positive_rows():
            for line, row in csv_rows(hard_dir / 'hard_images_review.csv',
                                      ['file_prefix', 'rec_idx', 'label', 'genuine_score']):
                prefix = prefix_name(row['file_prefix'])
                if prefix not in source_ids:
                    raise ValueError(f'hard_images_review.csv:{line}: unknown source {prefix}')
                yield (source_ids[prefix], nonnegative(row['rec_idx']),
                       nonnegative(row['label']), score_value(row['genuine_score']))
        for batch in batches(positive_rows()):
            conn.executemany('''INSERT INTO hard_pos VALUES(?,?,?,?) ON CONFLICT(source_id,rec_idx)
                DO UPDATE SET label=CASE WHEN hard_pos.label=excluded.label
                THEN hard_pos.label ELSE NULL END,score=MIN(hard_pos.score,excluded.score)''', batch)
            conn.commit()

        conn.execute('CREATE INDEX hard_neg_score ON hard_neg(score DESC,a,b)')
        conn.execute('CREATE INDEX hard_pos_score ON hard_pos(score ASC,source_id,rec_idx)')

        bad_neg = conn.execute('''SELECT a,b FROM hard_neg h WHERE
            NOT EXISTS(SELECT 1 FROM images i WHERE i.identity=h.a) OR
            NOT EXISTS(SELECT 1 FROM images i WHERE i.identity=h.b) LIMIT 10''').fetchall()
        bad_pos = conn.execute('''SELECT s.prefix,h.rec_idx,h.label,i.identity FROM hard_pos h
            JOIN sources s ON s.id=h.source_id LEFT JOIN images i
            ON i.source_id=h.source_id AND i.rec_idx=h.rec_idx
            WHERE i.identity IS NULL OR i.identity<>h.label LIMIT 10''').fetchall()
        if bad_neg or bad_pos:
            raise ValueError(f'CSV references mismatch Parquet: negative examples={bad_neg}; '
                             f'positive examples={bad_pos}')
        summary = {
            'version': VERSION, 'index_id': uuid.uuid4().hex,
            'embeddings_dir': str(embed_dir), 'rec_dir': str(rec_dir),
            'hard_dir': str(hard_dir), 'metadata_rows': total,
            'unique_images': conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],
            'identities': conn.execute('SELECT COUNT(DISTINCT identity) FROM images').fetchone()[0],
            'negative_pairs': conn.execute('SELECT COUNT(*) FROM hard_neg').fetchone()[0],
            'hard_images': conn.execute('SELECT COUNT(*) FROM hard_pos').fetchone()[0],
            'prefixes': sorted(source_ids),
            'parquet_files': [{'path': str(p.relative_to(embed_dir)),
                               'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                              for p in files],
            'csv_files': [{'path': str(hard_dir / n), 'sha256': file_hash(hard_dir / n)}
                for n in ['hard_class_neighbors.csv', 'hard_images_review.csv']],
        }
        for key, value in summary.items():
            conn.execute('INSERT INTO meta VALUES(?,?)', (key, json.dumps(value)))
        conn.commit()
        conn.close()
        temporary.replace(output)
        output.with_suffix(output.suffix + '.json').write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k:v for k,v in summary.items() if k not in ['parquet_files','csv_files']},
                         indent=2, ensure_ascii=False))
        print(f'Index ready: {output}', flush=True)
    except BaseException:
        conn.close()
        temporary.unlink(missing_ok=True)
        Path(str(temporary) + '-journal').unlink(missing_ok=True)
        raise


def stable_rng(seed, token):
    digest = hashlib.sha256(f'{seed}:{token}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def pool_stats(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a fresh --output-dir')
    output.mkdir(parents=True,exist_ok=True)
    conn = open_db(args.db)
    try:
        # Temporary tables live separately from the read-only source DB.
        conn.execute('PRAGMA temp_store=FILE')
        conn.execute('PRAGMA cache_size=-65536')
        print('Counting hard cases and identity concentration (no image reads)...',flush=True)
        conn.execute('''CREATE TEMP TABLE neg_degrees AS SELECT identity,COUNT(*) AS degree
            FROM (SELECT a AS identity FROM hard_neg UNION ALL SELECT b FROM hard_neg)
            GROUP BY identity''')
        neg_count = conn.execute('SELECT COUNT(*) FROM hard_neg').fetchone()[0]
        pos_count = conn.execute('SELECT COUNT(*) FROM hard_pos').fetchone()[0]
        top_neg = [dict(r) for r in conn.execute('''SELECT identity,degree FROM neg_degrees
            ORDER BY degree DESC,identity LIMIT ?''',(args.top_identities,))]
        top_pos = [dict(r) for r in conn.execute('''SELECT label AS identity,COUNT(*) AS hard_images
            FROM hard_pos GROUP BY label ORDER BY hard_images DESC,label LIMIT ?''',
                                              (args.top_identities,))]
        for row in top_neg + top_pos:
            row['dataset_images'] = conn.execute('SELECT COUNT(*) FROM images WHERE identity=?',
                                                 (row['identity'],)).fetchone()[0]
        for row in top_pos:
            row['hard_image_fraction'] = row['hard_images']/row['dataset_images']
        by_source = [dict(r) for r in conn.execute('''SELECT s.prefix AS file_prefix,COUNT(*) AS hard_images,
            COUNT(DISTINCT h.label) AS affected_identities FROM hard_pos h JOIN sources s
            ON s.id=h.source_id GROUP BY s.id ORDER BY hard_images DESC,s.prefix''')]
        edges = [-1.00001,0,.2,.3,.4,.5,.6,.7,.8,.9,.95,.98,1.00001]
        def histogram(table):
            expressions = ','.join(f'SUM(CASE WHEN score>=? AND score<? THEN 1 ELSE 0 END) AS b{i}'
                                  for i in range(len(edges)-1))
            params = [v for a,b in zip(edges,edges[1:]) for v in (a,b)]
            counts = conn.execute(f'SELECT {expressions} FROM {table}',params).fetchone()
            return [{'lower_inclusive':a,'upper_exclusive':b,'count':counts[i] or 0}
                    for i,(a,b) in enumerate(zip(edges,edges[1:]))]
        def score_summary(table):
            row = conn.execute(f'SELECT MIN(score) AS min,MAX(score) AS max,AVG(score) AS mean FROM {table}').fetchone()
            return dict(row)
        summary = {
            'db':str(Path(args.db).resolve()),
            'negative_unique_pairs':neg_count,
            'negative_affected_identities':conn.execute('SELECT COUNT(*) FROM neg_degrees').fetchone()[0],
            'positive_unique_hard_images':pos_count,
            'positive_affected_identities':conn.execute('SELECT COUNT(DISTINCT label) FROM hard_pos').fetchone()[0],
            'negative_scores':score_summary('hard_neg'),'positive_scores':score_summary('hard_pos'),
            'negative_score_histogram':histogram('hard_neg'),'positive_score_histogram':histogram('hard_pos'),
            'top_negative_identities':top_neg,'top_positive_identities':top_pos,
            'positive_by_source':by_source,
            'top_negative_endpoint_fraction':sum(r['degree'] for r in top_neg)/(2*neg_count) if neg_count else 0,
            'top_positive_image_fraction':sum(r['hard_images'] for r in top_pos)/pos_count if pos_count else 0,
            'notes':[
                'Negative CSV symmetric rows are counted once per unordered identity pair.',
                'Negative endpoint fraction counts two endpoints/pair, not unique pair coverage.',
                'Identity degree or extreme score is a review priority, not proof of label noise.',
                'Source counts apply to hard images; identities may span multiple sources.',
            ],
        }
        (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        for name,rows,columns in [
            ('top_negative_identities.csv',top_neg,['identity','degree','dataset_images']),
            ('top_positive_identities.csv',top_pos,['identity','hard_images','dataset_images','hard_image_fraction']),
            ('positive_by_source.csv',by_source,['file_prefix','hard_images','affected_identities']),
        ]:
            with (output/name).open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)
        lines = [
            f'Negative unique identity pairs: {neg_count:,}',
            f'Negative affected identities: {summary["negative_affected_identities"]:,}',
            f'Positive hard images: {pos_count:,}',
            f'Positive affected identities: {summary["positive_affected_identities"]:,}',
            f'Top {len(top_neg)} identities share of negative endpoints: {summary["top_negative_endpoint_fraction"]:.2%}',
            f'Top {len(top_pos)} identities share of positive hard images: {summary["top_positive_image_fraction"]:.2%}',
            '', 'Top negative identities (id / number of neighbor identities / dataset images):',
        ] + [f'{r["identity"]} / {r["degree"]:,} / {r["dataset_images"]:,}' for r in top_neg]
        lines += ['', 'Top positive identities (id / hard images / dataset images):']
        lines += [f'{r["identity"]} / {r["hard_images"]:,} / {r["dataset_images"]:,}' for r in top_pos]
        lines += ['', 'Concentration identifies groups to inspect; it does not automatically exclude them.']
        text='\n'.join(lines)+'\n'
        (output/'summary.txt').write_text(text,encoding='utf-8')
        print(text,flush=True)
        print(f'Statistics ready: {output}',flush=True)
    finally:
        conn.close()


def select_cases(conn, kind, limit, random_count, seed):
    table = 'hard_neg' if kind == 'negative' else 'hard_pos'
    order = 'score DESC,a,b' if kind == 'negative' else 'score ASC,source_id,rec_idx'
    top = [dict(r) for r in conn.execute(f'SELECT * FROM {table} ORDER BY {order} LIMIT ?', (limit,))]
    key = (lambda r:(r['a'],r['b'])) if kind=='negative' else (lambda r:(r['source_id'],r['rec_idx']))
    top_keys = {key(r) for r in top}
    # Reservoir sampling: scan remaining rows with bounded memory, no ORDER BY RANDOM().
    rng, reservoir, count = stable_rng(seed, kind), [], 0
    if random_count:
        for row in conn.execute(f'SELECT * FROM {table}'):
            row = dict(row)
            if key(row) in top_keys:
                continue
            count += 1
            if len(reservoir) < random_count:
                reservoir.append(row)
            else:
                j = rng.randrange(count)
                if j < random_count:
                    reservoir[j] = row
    for row in top:
        row['selection'] = 'extreme_score'
    for row in reservoir:
        row['selection'] = 'random_sample'
    return top + reservoir


def select_sampling_plan(conn, plan_path, seed):
    plan=json.loads(Path(plan_path).read_text(encoding='utf-8'))
    if not isinstance(plan,list):
        raise ValueError('Sampling plan must be a JSON list of strata')
    result={'negative':[],'positive':[]}
    counts=[];seen=set()
    for i,stratum in enumerate(plan):
        kind=stratum.get('kind');name=stratum.get('name',f'stratum_{i}')
        if kind not in result:
            raise ValueError(f'{name}: kind must be negative or positive')
        limit=nonnegative(stratum['count'])
        lower=score_value(stratum.get('score_min',-1))
        upper=float(stratum.get('score_max',1.00001))
        if not math.isfinite(upper) or not lower<upper<=1.00001:
            raise ValueError(f'{name}: invalid score interval')
        source=stratum.get('source_glob')
        if source is not None and kind!='positive':
            raise ValueError(f'{name}: source_glob only supported for positive images')
        table='hard_neg' if kind=='negative' else 'hard_pos'
        join=' JOIN sources s ON s.id=h.source_id' if source is not None else ''
        condition=' AND s.prefix GLOB ?' if source is not None else ''
        order='h.score DESC,h.a,h.b' if kind=='negative' else 'h.score ASC,h.source_id,h.rec_idx'
        params=[lower,upper]+([source] if source is not None else [])
        query=f'SELECT h.* FROM {table} h{join} WHERE h.score>=? AND h.score<?{condition} ORDER BY {order}'
        rng=stable_rng(seed,f'plan-{i}-{name}');pool=[];eligible=0
        if limit:
            for row in conn.execute(query,params):
                row=dict(row)
                key=(kind,row['a'],row['b']) if kind=='negative' else (kind,row['source_id'],row['rec_idx'])
                if key in seen:continue
                eligible+=1
                if len(pool)<limit:pool.append(row)
                else:
                    j=rng.randrange(eligible)
                    if j<limit:pool[j]=row
        for row in pool:
            key=(kind,row['a'],row['b']) if kind=='negative' else (kind,row['source_id'],row['rec_idx'])
            seen.add(key);row['selection']=f'stratum:{name}'
            result[kind].append(row)
        counts.append({'name':name,'kind':kind,'requested':limit,'eligible':eligible,'selected':len(pool)})
        print(f'Stratum {name}: selected {len(pool)}/{eligible:,} eligible cases',flush=True)
    return result,counts


def class_refs(conn, identity, count, seed, exclude=None):
    rng, result, seen = stable_rng(seed, f'identity-{identity}'), [], 0
    for row in conn.execute('''SELECT s.prefix AS file_prefix,i.rec_idx,i.identity FROM images i
        JOIN sources s ON s.id=i.source_id WHERE i.identity=? ORDER BY i.source_id,i.rec_idx''', (identity,)):
        row = dict(row)
        if exclude == (row['file_prefix'],row['rec_idx']):
            continue
        seen += 1
        if len(result) < count:
            result.append(row)
        else:
            j = rng.randrange(seen)
            if j < count:
                result[j] = row
    return result


class RecordReader:
    """Read only requested .idx offsets, not a full MXNet index in memory.

    DMLC RecordIO framing (recordio.h): 29-bit lengths, 0/1/2/3 continuation
    flags. Magic words removed by the writer are reinserted between chunks.
    MXImageRecord payload header: little-endian IfQQ, optional float labels.
    """
    def __init__(self, rec_dir, requests, max_open=4):
        self.rec_dir, self.max_open = Path(rec_dir), max_open
        self.offsets, self.handles = {}, OrderedDict()
        for prefix, indices in sorted(requests.items()):
            prefix_name(prefix)
            remaining = set(indices)
            offsets = {}
            with (self.rec_dir / f'{prefix}.idx').open() as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    key, offset = map(int, line.split())
                    if key in remaining:
                        if offset < 0:
                            raise ValueError(f'Negative offset in {prefix}.idx')
                        offsets[key] = offset
                        remaining.remove(key)
                        if not remaining:
                            break
            self.offsets[prefix] = offsets
            if remaining:
                print(f'WARNING {prefix}: {len(remaining)} requested records absent from .idx',
                      file=sys.stderr, flush=True)

    def close(self):
        for stream in self.handles.values():
            stream.close()
        self.handles.clear()

    def payload(self, prefix, index):
        if index not in self.offsets[prefix]:
            raise ValueError(f'{prefix}: record {index} missing from .idx')
        if prefix not in self.handles:
            if len(self.handles) >= self.max_open:
                _, stream = self.handles.popitem(last=False)
                stream.close()
            self.handles[prefix] = (self.rec_dir / f'{prefix}.rec').open('rb')
        self.handles.move_to_end(prefix)
        stream = self.handles[prefix]
        stream.seek(self.offsets[prefix][index])
        chunks, size, started = [], 0, False
        while True:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError('Truncated RecordIO framing header')
            magic, encoded = struct.unpack('<II', header)
            flag, length = encoded >> 29, encoded & ((1 << 29) - 1)
            if magic != MAGIC or flag > 3:
                raise ValueError('Invalid RecordIO magic/continuation flag')
            if (not started and flag not in (0,1)) or (started and flag not in (2,3)):
                raise ValueError('Invalid RecordIO continuation sequence')
            size += length + (4 if started else 0)
            if size > 32 * 1024 * 1024:
                raise ValueError('Image record exceeds 32 MiB safety bound')
            data = stream.read(length)
            pad = stream.read((-length) % 4)
            if len(data) != length or len(pad) != (-length) % 4:
                raise ValueError('Truncated RecordIO payload')
            if started:
                chunks.append(struct.pack('<I', MAGIC))
            chunks.append(data)
            if flag in (0,3):
                return b''.join(chunks)
            started = True

    def image(self, ref):
        from PIL import Image, ImageOps
        payload = self.payload(ref['file_prefix'],ref['rec_idx'])
        if len(payload) < 24:
            raise ValueError('Truncated MXImageRecord header')
        count, scalar_label, _, _ = struct.unpack('<IfQQ', payload[:24])
        start = 24 + count * 4
        if start > len(payload):
            raise ValueError('Truncated MXImageRecord label array')
        label = struct.unpack('<f',payload[24:28])[0] if count else scalar_label
        if not math.isfinite(label) or label != ref['identity']:
            raise ValueError(f"Record label {label} != Parquet identity {ref['identity']}")
        with Image.open(io.BytesIO(payload[start:])) as img:
            return ImageOps.exif_transpose(img).convert('RGB').copy()


def write_sheet(reader, groups, path):
    from PIL import Image, ImageDraw, ImageOps
    cell_w, cell_h, label_h = 190, 230, 44
    cols = max(1,max(len(refs) for _,refs in groups))
    canvas = Image.new('RGB',(cols*cell_w,len(groups)*(cell_h+label_h)), '#f2f4f7')
    draw, errors = ImageDraw.Draw(canvas), []
    for row,(title,refs) in enumerate(groups):
        y = row*(cell_h+label_h)
        draw.text((8,y+8),title,fill='#111111')
        for col,ref in enumerate(refs):
            x = col*cell_w
            try:
                img = reader.image(ref)
                thumb = ImageOps.contain(img,(cell_w-12,cell_h-55))
                canvas.paste(thumb,(x+(cell_w-thumb.width)//2,y+label_h))
            except Exception as exc:
                message = f"{ref['file_prefix']}:{ref['rec_idx']}: {exc}"
                errors.append(message)
                draw.text((x+8,y+label_h+30),'READ / LABEL ERROR',fill='#b00020')
            draw.text((x+6,y+label_h+cell_h-46),ref['file_prefix'],fill='#222222')
            draw.text((x+6,y+label_h+cell_h-27),
                      f"idx={ref['rec_idx']} id={ref['identity']}",fill='#222222')
    canvas.save(path,quality=90)
    return errors


PAGE_JS = r'''
const KEY='hard-review-'+PROJECT;
let decisions={};
try { decisions=JSON.parse(localStorage.getItem(KEY)||'{}'); } catch(e) {}
function paint(id) {
  const c=document.getElementById(id); if(!c)return;
  const value=decisions[id]||'pending'; c.dataset.status=value;
  c.querySelector('.status').textContent=value;
}
function decide(id,status) {
  const c=ALL.find(x=>x.id===id);
  if(status==='keep'&&c.errors.length){alert('Cannot approve a case with read/label errors.');return;}
  decisions[id]=status;
  try{localStorage.setItem(KEY,JSON.stringify(decisions));}
  catch(e){alert('Browser storage unavailable. Export decisions before leaving this page.');}
  paint(id);updateCount();
}
function updateCount(){document.getElementById('counts').textContent=
 ALL.length+' cases; '+ALL.filter(c=>decisions[c.id]==='keep').length+' approved; '+
 ALL.filter(c=>['exclude','unsure'].includes(decisions[c.id])).length+' excluded/unsure';}
function esc(v){return '"'+String(v??'').replaceAll('"','""')+'"';}
function download(name,headers,rows){
 const csv='\ufeff'+[headers,...rows].map(r=>r.map(esc).join(',')).join('\r\n')+'\r\n';
 const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
 const a=document.createElement('a');a.href=url;a.download=name;a.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function exportFile(name){
 const chosen=status=>ALL.filter(c=>status.includes(decisions[c.id]||'pending'));
 if(name==='review_decisions.csv'){
  download(name,['case_id','kind','status','class_a','class_b','file_prefix','rec_idx','label','score'],
   ALL.map(c=>[c.id,c.kind,decisions[c.id]||'pending',c.a,c.b,c.file_prefix,c.rec_idx,c.label,c.score]));return;
 }
 const approved=name.startsWith('approved');
 const rows=chosen(approved?['keep']:['exclude','unsure']);
 if(name.includes('identity_pairs')){
  download(name,['class_a','class_b','score','status'],rows.filter(c=>c.kind==='negative')
    .map(c=>[c.a,c.b,c.score,decisions[c.id]]));
 }else{
  download(name,['file_prefix','rec_idx','label','genuine_score','status'],rows.filter(c=>c.kind==='positive')
    .map(c=>[c.file_prefix,c.rec_idx,c.label,c.score,decisions[c.id]]));
 }
}
document.querySelectorAll('.case').forEach(c=>paint(c.id));updateCount();
'''


def write_html(cases, output, project_id, page_size):
    pages = max(1, math.ceil(len(cases)/page_size))
    all_json = json.dumps(cases,ensure_ascii=False).replace('<','\\u003c')
    for page in range(pages):
        name = 'index.html' if page==0 else f'page_{page+1:03d}.html'
        nav = ' '.join(f'<a href="{"index.html" if p==0 else f"page_{p+1:03d}.html"}">{p+1}</a>'
                       for p in range(pages))
        cards = []
        for case in cases[page*page_size:(page+1)*page_size]:
            title = (f"Different identities: {case['a']} / {case['b']}" if case['kind']=='negative'
                     else f"Hard image: {case['file_prefix']}:{case['rec_idx']} — identity {case['label']}")
            errors = ''.join(f'<li>{html.escape(e)}</li>' for e in case['errors'])
            buttons = ''.join(f'<button onclick="decide(\'{case["id"]}\',\'{status}\')">{label}</button>'
                              for status,label in [('keep','Approve labels'),('exclude','Exclude'),('unsure','Unsure')])
            cards.append(f'''<article class="case" id="{case['id']}">
              <h2>{html.escape(title)}</h2><p>Score: {case['score']:.8f} · {case['selection']}
              · Status: <strong class="status">pending</strong></p>
              <img loading="lazy" src="{case['sheet']}" alt="{html.escape(title,quote=True)}">
              <ul class="errors">{errors}</ul><div>{buttons}</div></article>''')
        exports = ''.join(f'<button onclick="exportFile(\'{n}\')">Download {n}</button>' for n in
            ['review_decisions.csv','approved_identity_pairs.csv','approved_images.csv',
             'excluded_identity_pairs.csv','excluded_images.csv'])
        document = f'''<!doctype html><html lang="vi"><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1"><title>Hard-case label review</title>
        <style>body{{font:16px system-ui;margin:24px auto;max-width:1250px;padding:0 16px;background:#fafafa}}
        header{{background:#fff;padding:18px;border:1px solid #ddd}}button{{padding:9px;margin:4px;cursor:pointer}}
        .case{{background:white;border:2px solid #ddd;border-radius:8px;padding:16px;margin:22px 0}}
        .case[data-status=keep]{{border-color:#198754}}.case[data-status=exclude]{{border-color:#c33}}
        .case[data-status=unsure]{{border-color:#bd8b00}}img{{max-width:100%;height:auto}}
        h2{{font-size:20px}}.errors{{color:#b00020}}nav a{{margin:8px;display:inline-block}}
        </style><header><h1>Hard-case label review</h1>
        <p>Negative: xác nhận hai identity khác người. Positive: xác nhận ảnh khó đúng người của identity.
        Ảnh reference lấy mẫu; score gốc từ mining dựa vào medoid, không phải score của mọi cặp ảnh hiển thị.</p>
        <p>Approve chỉ xác nhận nhãn của case đã xem. Pending/Unsure không nằm trong danh sách approved.
        Dùng HTTP localhost để quyết định được lưu chung giữa các trang; tải CSV trước khi chuyển máy.</p>
        <p id="counts"></p>{exports}<nav>{nav}</nav></header>{''.join(cards)}
        <nav>{nav}</nav><script>const PROJECT={json.dumps(project_id)};const ALL={all_json};{PAGE_JS}</script></html>'''
        (output/name).write_text(document,encoding='utf-8')


def review(args):
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; use a fresh --output-dir')
    output.mkdir(parents=True,exist_ok=True)
    (output/'sheets').mkdir(exist_ok=True)
    conn = open_db(args.db)
    meta = {r['key']:json.loads(r['value']) for r in conn.execute('SELECT * FROM meta')}
    rec_dir = args.rec_dir or meta['rec_dir']
    sources = {r['id']:r['prefix'] for r in conn.execute('SELECT * FROM sources')}
    cases, requests = [], {}
    if getattr(args,'sampling_plan',None):
        selected,sampling_counts=select_sampling_plan(conn,args.sampling_plan,args.seed)
    else:
        selected={kind:select_cases(conn,kind,limit,args.random_each,args.seed)
                  for kind,limit in [('negative',args.negative_limit),('positive',args.positive_limit)]}
        sampling_counts=None
    for row in selected['negative']:
        groups = [(f"Identity {row['a']}",class_refs(conn,row['a'],args.images_per_class,args.seed)),
                  (f"Identity {row['b']}",class_refs(conn,row['b'],args.images_per_class,args.seed))]
        cases.append(dict(row,kind='negative',id=f"negative_{row['a']}_{row['b']}",groups=groups))
    for row in selected['positive']:
        prefix = sources[row['source_id']]
        focus = {'file_prefix':prefix,'rec_idx':row['rec_idx'],'identity':row['label']}
        refs = class_refs(conn,row['label'],args.images_per_class,args.seed,
                          exclude=(prefix,row['rec_idx']))
        groups = [('Flagged hard image',[focus]),(f"Same-identity references: {row['label']}",refs)]
        case_id = 'positive_' + hashlib.sha256(f'{prefix}:{row["rec_idx"]}'.encode()).hexdigest()[:20]
        cases.append(dict(row,kind='positive',id=case_id,file_prefix=prefix,groups=groups))
    conn.close()
    for case in cases:
        for _, refs in case['groups']:
            for ref in refs:
                requests.setdefault(ref['file_prefix'],set()).add(ref['rec_idx'])
    print(f'Selected {len(cases)} cases, {sum(map(len,requests.values()))} unique image references.',flush=True)
    reader = RecordReader(rec_dir,requests,args.max_open_files)
    try:
        for i,case in enumerate(cases,1):
            sheet = f"sheets/{case['id']}.jpg"
            case['sheet'] = sheet
            case['errors'] = write_sheet(reader,case['groups'],output/sheet)
            if case['kind']=='positive' and not case['groups'][1][1]:
                case['errors'].append('No other image in this identity; cannot verify against references.')
            print(f'Rendered {i}/{len(cases)}: {case["id"]}',flush=True)
    finally:
        reader.close()
    project_id = meta['index_id'] + '-' + uuid.uuid4().hex
    manifest = {'version':VERSION,'project_id':project_id,'db':str(Path(args.db).resolve()),
                'seed':args.seed,'cases':cases,
                'sampling_counts':sampling_counts,
                'sampling_plan':json.loads(Path(args.sampling_plan).read_text()) if getattr(args,'sampling_plan',None) else None,
                'read_errors':sum(len(c['errors']) for c in cases)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    write_html(cases,output,project_id,args.page_size)
    print(f'Review ready: {output / "index.html"}; read/label errors: {manifest["read_errors"]}')


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest='command',required=True)
    index = sub.add_parser('index',help='Stream Parquet metadata into SQLite; validate both hard CSVs')
    index.add_argument('--embeddings-dir',required=True)
    index.add_argument('--rec-dir',required=True)
    index.add_argument('--hard-dir',required=True)
    index.add_argument('--db',required=True)
    index.add_argument('--batch-size',type=int,default=50000)
    index.set_defaults(run=build_index)
    stats = sub.add_parser('stats',help='Count hard pool, score distributions and identity concentration; no images')
    stats.add_argument('--db',required=True)
    stats.add_argument('--output-dir',required=True)
    stats.add_argument('--top-identities',type=int,default=30)
    stats.set_defaults(run=pool_stats)
    viewer = sub.add_parser('review',help='Render sampled hard cases and browser decision/export UI')
    viewer.add_argument('--db',required=True)
    viewer.add_argument('--rec-dir',help='Override RecordIO directory recorded in the index')
    viewer.add_argument('--output-dir',required=True)
    viewer.add_argument('--negative-limit',type=int,default=100)
    viewer.add_argument('--positive-limit',type=int,default=100)
    viewer.add_argument('--random-each',type=int,default=25)
    viewer.add_argument('--images-per-class',type=int,default=6)
    viewer.add_argument('--page-size',type=int,default=25)
    viewer.add_argument('--max-open-files',type=int,default=4)
    viewer.add_argument('--seed',type=int,default=2048)
    viewer.add_argument('--sampling-plan',help='JSON strata; overrides limits/random-each. Random bounded-memory sampling per stratum.')
    viewer.set_defaults(run=review)
    return root


def main():
    args = parser().parse_args()
    for name in ['batch_size','images_per_class','page_size','max_open_files','top_identities']:
        if hasattr(args,name) and getattr(args,name)<=0:
            raise SystemExit(f'--{name.replace("_","-")} must be positive')
    for name in ['negative_limit','positive_limit','random_each']:
        if hasattr(args,name) and getattr(args,name)<0:
            raise SystemExit(f'--{name.replace("_","-")} must be nonnegative')
    try:
        args.run(args)
    except (ValueError,OSError,sqlite3.Error) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc


if __name__=='__main__':
    main()
