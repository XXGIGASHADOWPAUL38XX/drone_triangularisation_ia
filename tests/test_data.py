import csv
import json
import wave
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from edth.data import (COLS, FIR, LOGMEL_FIELD, LOGMEL_LEN, MEL_CENTERS, N_FRAMES, N_MELS, VERSION,
                       Decimator48to24, _events, assign_splits, features_from_pcm24, features_from_pcm48, prepare)


def tone(hz, rate, seconds, level=0.5):
    return np.rint(32767 * level * np.sin(2*np.pi*hz*np.arange(int(seconds*rate))/rate)).astype('<i2')


def logmel(pcm24):
    return features_from_pcm24(pcm24)[LOGMEL_FIELD]


def logmel48(pcm48):
    return list(features_from_pcm48(pcm48))[0][LOGMEL_FIELD]


def test_dbfs_parity_and_antialias():
    pcm = tone(6100, 48000, 2)
    rows = list(features_from_pcm48(pcm))
    assert [r['start_sample_24k'] for r in rows] == [12000, 18000, 24000]
    p24 = np.rint(np.convolve(pcm, FIR)[:len(pcm):2]).astype('int16')
    r = features_from_pcm24(p24[12000:36000]); assert set(r) == set(COLS + ['b4_8', 'rms_db', LOGMEL_FIELD])
    for k in set(r) - {LOGMEL_FIELD}:
        assert rows[0][k] == r[k]
    # Offline 48 kHz path and 24 kHz replay must share the exact same logmel.
    np.testing.assert_array_equal(rows[0][LOGMEL_FIELD], r[LOGMEL_FIELD])
    for k in ['b4_8', 'rms_db']:
        assert r[k] == pytest.approx(-9.03, abs=.02)
    alias = list(features_from_pcm48(tone(18000, 48000, 2)))[0]
    assert alias['rms_db'] < -85
    with pytest.raises(ValueError): features_from_pcm24(pcm.astype(float))


def test_logmel_contract_is_documented_and_flat():
    arr = logmel48(tone(3000, 48000, 2))
    assert arr.dtype == np.float32 and arr.shape == (LOGMEL_LEN,) and LOGMEL_LEN == 704
    # Mel-major flatten: channel m occupies the contiguous block m*22..m*22+21.
    assert arr.reshape(N_MELS, N_FRAMES).shape == (32, 22)
    assert np.isfinite(arr).all()


def test_logmel_keeps_time_order_in_mel_major_layout():
    n = np.arange(24000)
    pcm = np.rint(20000 * (n / 24000) * np.sin(2*np.pi*1000*n/24000)).astype('<i2')
    arr = logmel(pcm)
    def frame_energy(a):
        return 10*np.log10((10**(a/10)).sum(axis=0))  # per frame, summed over mels
    ordered = np.diff(frame_energy(arr.reshape(N_MELS, N_FRAMES)))
    assert np.all(ordered > 0), 'rising window must produce monotonically rising frames'
    # The same bytes read as frame-major would not be time-ordered: this pins the layout.
    scrambled = np.diff(frame_energy(arr.reshape(N_FRAMES, N_MELS).T))
    assert not np.all(scrambled > 0)


def test_logmel_gain_is_additive_db():
    rng = np.random.default_rng(7)
    pcm = np.rint(rng.normal(0, 9000, 24000)).clip(-32768, 32767).astype('<i2')
    loud, quiet = logmel(pcm), logmel(np.rint(pcm/2).astype('<i2'))
    active = loud > -90  # ignore quantisation noise in near-silent bins
    assert active.sum() > 600
    np.testing.assert_allclose((quiet-loud)[active], 20*np.log10(.5), atol=.01)
    assert np.all(logmel(np.zeros(24000, '<i2')) == -120.)


def test_triangular_filterbank_orders_frequency_at_mel_centres():
    for target in (489., 3023., 8261.):
        k = int(np.argmin(abs(MEL_CENTERS-target)))
        spectrum = np.median(logmel48(tone(MEL_CENTERS[k], 48000, 2)).reshape(N_MELS, N_FRAMES), axis=1)
        assert int(np.argmax(spectrum)) == k  # a tone peaks in its own mel channel


def test_fir_block_invariance_including_phase_and_startup():
    rng = np.random.default_rng(31)
    x = rng.integers(-32768, 32768, 5201, dtype=np.int16)
    reference = np.clip(np.rint(np.convolve(x, FIR, mode='full')[:len(x):2]), -32768, 32767).astype('int16')
    for size in [1, 39, 128, 161, 1025, 2400]:
        state = Decimator48to24()
        blocks = [state.process(x[i:i+size]) for i in range(0, len(x), size)]
        np.testing.assert_array_equal(np.concatenate(blocks), reference)


def test_qc_keeps_quiet_drops_digital_interruptions_and_recovers():
    x = np.full(4*48000, 1, dtype='int16')
    x[2*48000:2*48000+1920] = 0
    starts = [r['start_sample_24k']/24000 for r in features_from_pcm48(x)]
    assert starts == [.5, .75, 1., 2.75, 3.]
    assert not list(features_from_pcm48(x*0))


def test_transitive_cross_device_overlap_and_infeasible_cv():
    rows = [dict(sample_id=str(i), start_utc=s, duration_s=5.) for i,s in enumerate([0.,5.5,11.,20.])]
    _events(rows)
    assert len({r['group_id'] for r in rows[:3]}) == 1
    assert rows[2]['group_id'] != rows[3]['group_id']
    with pytest.raises(ValueError, match='too few groups'): assign_splits([])


def test_collector_audiofolder_equivalence(tmp_path):
    local = tmp_path/'collector'; folder = tmp_path/'snapshot'/'data'; folder.mkdir(parents=True)
    rows = []
    for i in range(12):
        sid = str(i); dist = [None, 1., 3.][i%3]
        # Four non-overlapping events; three simultaneous phones in each.
        dt = datetime(2026,1,1,tzinfo=timezone.utc) + timedelta(seconds=10*(i//3))
        m = dict(sample_id=sid, captured_at=dt.isoformat(), device_id=f'd{i%3}', mode='hand',
                 capture_settings={'capture_contract':'edth.web-pcm.v1'}, drone_present=dist is not None,
                 distance_m=dist, duration_seconds=2)
        sub = local/'samples'/sid; sub.mkdir(parents=True)
        (sub/'metadata.json').write_text(json.dumps(m)); pcm = tone(1500+i, 48000, 2)
        with wave.open(str(sub/'audio.wav'), 'wb') as w:
            w.setparams((1,2,48000,0,'NONE','not compressed')); w.writeframes(pcm.tobytes())
        wav = folder/f'{sid}.wav'; wav.write_bytes((sub/'audio.wav').read_bytes())
        rows.append({**m,'capture_settings':json.dumps(m['capture_settings']), 'file_name':wav.name})
    with (folder/'metadata.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(reversed(rows))
    a = prepare(local, tmp_path/'a'); b = prepare(folder.parent, tmp_path/'b')
    with pytest.raises(FileExistsError):
        prepare(local, tmp_path/'a')
    # Array-valued column cannot go through assert_frame_equal: compare separately.
    pd.testing.assert_frame_equal(a.drop(columns=LOGMEL_FIELD).reset_index(drop=True),
                                  b.drop(columns=LOGMEL_FIELD).reset_index(drop=True))
    np.testing.assert_array_equal(np.stack(a[LOGMEL_FIELD]), np.stack(b[LOGMEL_FIELD]))
    assert a[LOGMEL_FIELD].map(lambda v: len(v) == LOGMEL_LEN).all()
    stored = pd.read_parquet(tmp_path/'a'/'windows.parquet')
    fixture = stored[LOGMEL_FIELD].iloc[0]
    assert np.asarray(fixture).dtype == np.float32 and len(fixture) == LOGMEL_LEN
    np.testing.assert_array_equal(np.asarray(fixture), a[LOGMEL_FIELD].iloc[0])
    manifest = json.loads((tmp_path/'a'/'manifest.json').read_text())
    assert manifest['version'] == VERSION == 'edth.audio.24k.v2'
    assert manifest['logmel']['frames'] == N_FRAMES and manifest['logmel']['mels'] == N_MELS
    assert manifest['logmel']['length'] == LOGMEL_LEN and 'mel*22 + frame' in manifest['logmel']['layout']
    assert manifest['logmel']['filterbank_sha256'] and manifest['logmel']['mel_edges_hz'][-1] == 12000
    assert len(a)==36 and set(a.fold)=={-1,0,1,2}
    assert a.groupby('group_id').fold.nunique().max()==1
    assert a.groupby('fold').drone_present.nunique().eq(2).all()
    assert (a['split'].eq('test') == a.fold.eq(-1)).all()
    assert len(json.loads((tmp_path/'a'/'manifest.json').read_text())['takes'])==12
