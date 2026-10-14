import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sqlite3
import struct
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

script=Path(__file__).with_name('hard_case_review.py')
if not script.is_file():
    script=Path(__file__).resolve().parents[1]/'hard_case_review.py'
spec=importlib.util.spec_from_file_location('review_tool',script)
tool=importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def frame(payload,flag=0):
    return struct.pack('<II',tool.MAGIC,(flag<<29)|len(payload))+payload+b'\0'*((-len(payload))%4)


def image_payload(identity,color,array_label=False):
    buf=io.BytesIO()
    Image.new('RGB',(112,112),color).save(buf,format='JPEG')
    if array_label:
        return struct.pack('<IfQQ',1,0,1,0)+struct.pack('<f',identity)+buf.getvalue()
    return struct.pack('<IfQQ',0,identity,1,0)+buf.getvalue()


def fixture(root):
    rec=root/'rec_out';embed=root/'embeddings';hard=root/'hard';rec.mkdir();embed.mkdir();hard.mkdir()
    rows={
        'train_synthetic_clean':[(1,0,'red'),(2,0,'orange'),(3,1,'blue')],
        'train_public_clean':[(5,1,'cyan'),(8,2,'green')],
        'train_crawl_000':[(0,2,'lime'),(17,3,'purple')],
        'train_crawl_031':[(3,3,'pink'),(99,4,'yellow')],
    }
    for prefix,records in rows.items():
        offset=0
        with (rec/f'{prefix}.rec').open('wb') as data,(rec/f'{prefix}.idx').open('w') as idx:
            for key,identity,color in records:
                blob=frame(image_payload(identity,color,array_label=(key==8)))
                idx.write(f'{key}\t{offset}\n');data.write(blob);offset+=len(blob)
        part=embed/f'file_prefix={prefix}';part.mkdir()
        pq.write_table(pa.table({
            'identity':[r[1] for r in records], 'rec_idx':[r[0] for r in records],
            'embedding':pa.array([[1.0]+[0.0]*511 for r in records],type=pa.list_(pa.float32(),512))
        }),part/'part-rank0-chunk0-0.parquet')
    # Root artifacts must not be interpreted as Parquet.
    (embed/'false_accept.csv').write_text('class_a,class_b,score\n0,1,0.8\n')
    (hard/'hard_class_neighbors.csv').write_text(
        'class_id,neighbor_class_id,score\n0,1,0.90\n1,0,0.90\n2,3,0.88\n3,2,0.88\n0,4,0.70\n')
    (hard/'hard_images_review.csv').write_text(
        'file_prefix,rec_idx,label,genuine_score\ntrain_synthetic_clean,2,0,0.20\n'
        'train_crawl_000,17,3,0.25\ntrain_crawl_031,99,4,0.10\n')
    return rec,embed,hard


def index_args(root,rec,embed,hard):
    return argparse.Namespace(rec_dir=str(rec),embeddings_dir=str(embed),hard_dir=str(hard),
                              db=str(root/'index.sqlite'),batch_size=2)


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='face-review-test-',dir='/tmp')
        self.root=Path(self.tmp.name)
        self.rec,self.embed,self.hard=fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        args=index_args(self.root,self.rec,self.embed,self.hard)
        with contextlib.redirect_stdout(io.StringIO()):tool.build_index(args)
        return Path(args.db)

    def test_index_partition_metadata_and_symmetric_dedup(self):
        db=self.build()
        with tool.open_db(db) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM images').fetchone()[0],9)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM hard_neg').fetchone()[0],3)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM hard_pos').fetchone()[0],3)
            # Noncontiguous rec index is preserved as a reference, never used as dataset index.
            self.assertEqual(conn.execute('SELECT identity FROM images WHERE rec_idx=99').fetchone()[0],4)

    def test_conflicting_label_ref_fails_and_no_partial_db(self):
        (self.hard/'hard_images_review.csv').write_text(
            'file_prefix,rec_idx,label,genuine_score\ntrain_synthetic_clean,2,9,0.2\n')
        args=index_args(self.root,self.rec,self.embed,self.hard)
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'mismatch'):
            tool.build_index(args)
        self.assertFalse(Path(args.db).exists())
        self.assertEqual(list(self.root.glob('*.building-*')),[])

    def test_unknown_negative_identity_fails(self):
        (self.hard/'hard_class_neighbors.csv').write_text('class_id,neighbor_class_id,score\n0,999,0.9\n')
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'mismatch'):
            tool.build_index(index_args(self.root,self.rec,self.embed,self.hard))

    def test_reader_checks_labels_and_array_labels(self):
        reader=tool.RecordReader(self.rec,{'train_public_clean':{8}},max_open=1)
        try:
            self.assertEqual(reader.image({'file_prefix':'train_public_clean','rec_idx':8,'identity':2}).size,(112,112))
            with self.assertRaisesRegex(ValueError,'Record label'):
                reader.image({'file_prefix':'train_public_clean','rec_idx':8,'identity':9})
        finally:reader.close()

    def test_multichunk_reinserts_magic(self):
        payload=b'abcd'+struct.pack('<I',tool.MAGIC)+b'efgh'+struct.pack('<I',tool.MAGIC)+b'ijkl'
        prefix='chunk_test'
        (self.rec/f'{prefix}.rec').write_bytes(frame(b'abcd',1)+frame(b'efgh',2)+frame(b'ijkl',3))
        (self.rec/f'{prefix}.idx').write_text('7\t0\n')
        reader=tool.RecordReader(self.rec,{prefix:{7}})
        try:self.assertEqual(reader.payload(prefix,7),payload)
        finally:reader.close()

    def test_complete_review_zero_based_record_and_singleton_block(self):
        db=self.build();out=self.root/'review'
        args=argparse.Namespace(db=str(db),rec_dir=None,output_dir=str(out),negative_limit=1,
                               positive_limit=1,random_each=2,seed=7,images_per_class=3,
                               max_open_files=1,page_size=2)
        with contextlib.redirect_stdout(io.StringIO()):tool.review(args)
        manifest=json.loads((out/'manifest.json').read_text())
        self.assertEqual(len(manifest['cases']),6)
        self.assertEqual(manifest['read_errors'],1) # Singleton has no comparison reference.
        for case in manifest['cases']:
            self.assertTrue((out/case['sheet']).is_file())
        self.assertTrue((out/'page_003.html').exists())
        text=(out/'index.html').read_text()
        self.assertIn('approved_identity_pairs.csv',text)
        self.assertIn("Cannot approve a case with read/label errors",text)
        with self.assertRaises(FileExistsError):tool.review(args)

    def test_review_shows_actual_rec_label_mismatch(self):
        db=self.build()
        # Rewrite a flagged image with a conflicting actual RecordIO label.
        p=self.rec/'train_crawl_031.rec'
        p.write_bytes(frame(image_payload(3,'pink'))+frame(image_payload(999,'yellow')))
        # .idx offsets remain valid: both JPEGs were same dimensions and constant color,
        # but rewrite offset explicitly to avoid depending on JPEG byte sizes.
        first=frame(image_payload(3,'pink'))
        (self.rec/'train_crawl_031.idx').write_text(f'3\t0\n99\t{len(first)}\n')
        args=argparse.Namespace(db=str(db),rec_dir=None,output_dir=str(self.root/'bad_review'),
            negative_limit=0,positive_limit=1,random_each=0,seed=7,images_per_class=3,
            max_open_files=1,page_size=2)
        with contextlib.redirect_stdout(io.StringIO()):tool.review(args)
        manifest=json.loads((Path(args.output_dir)/'manifest.json').read_text())
        self.assertIn('Record label',manifest['cases'][0]['errors'][0])


if __name__=='__main__':unittest.main()
