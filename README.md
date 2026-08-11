# lyric-align

[![PyPI](https://img.shields.io/pypi/v/lyric-align)](https://pypi.org/project/lyric-align/)
[![Python](https://img.shields.io/pypi/pyversions/lyric-align)](https://pypi.org/project/lyric-align/)
[![Tests](https://github.com/Zennmn/lyric-align/actions/workflows/ci.yml/badge.svg)](https://github.com/Zennmn/lyric-align/actions/workflows/ci.yml)
[![License](https://img.shields.io/pypi/l/lyric-align)](LICENSE)

GPU-assisted lyric alignment for Japanese, Chinese, and mixed Japanese/English
singing. Give the program an audio file and the correct lyrics; it estimates
the timestamps without replacing the lyrics with an ASR transcript.

The current singing path is built around **Mel-Band RoFormer + Qwen3-ASR +
HubertFA**. Japanese is emitted character by character and English is emitted
word by word, which makes the result suitable for TTML, enhanced LRC, and
karaoke subtitles.

## Current pipeline

```text
song.flac + lyrics.txt
        │
        ▼
Mel-Band RoFormer vocal separation       CUDA
        │
        ▼
vocals.wav (cached on D:)
        │
        ▼
Qwen3-ASR-1.7B coarse text windows       CUDA
        │       Qwen3-ForcedAligner is disabled in HubertFA mode
        ▼
CJK lyric-align fuzzy matching            CPU
        │
        ▼
HubertFA global blocks: 10–15 lines,
padded audio limited to about 100 seconds
        │
        ▼
Japanese-context G2P + English CMU G2P
        │
        ▼
HubertFA ONNX singing alignment           CUDAExecutionProvider
        │
        ▼
Japanese character / English word spans
        │
        ▼
retry on anomalies → close tiny gaps → TTML/LRC/subtitle output
```

The pipeline keeps one model stage resident at a time to fit a 16 GB RTX 5070
Ti. Neural-network inference is requested on `cuda:0`; audio I/O, fuzzy
matching, G2P, and formatting remain CPU-side. A CUDA request fails loudly if
HubertFA cannot obtain `CUDAExecutionProvider`; it does not silently run the
aligner on CPU.

## Recommended GPU setup on Windows

The tested setup is Windows, Python 3.12, CUDA PyTorch, and an RTX 5070 Ti.
The setup script keeps the Python runtime, package caches, model caches, audio
stems, and outputs under the project root on D:.

Install `uv`, clone the repository, and run:

```powershell
git clone https://github.com/Zennmn/lyric-align.git D:\project\lyric-align
Set-Location D:\project\lyric-align

powershell -ExecutionPolicy Bypass -File .\scripts\setup_d_env.ps1
```

The script creates `.venv`, installs CUDA PyTorch/torchaudio, installs the
`pipeline`, `hubertfa`, and `dev` extras, and creates the D-drive model/cache
directories. It does not commit or download model weights into Git.

### Model layout

The project resolves models from the following locations:

```text
D:\project\lyric-align\
├─ .venv\
├─ .cache\
├─ models\
│  ├─ qwen\
│  ├─ melband\
│  └─ hubertfa\
│     └─ 1218_hfa_model_new_dict\
│        ├─ model.onnx
│        ├─ config.json
│        ├─ vocab.json
│        ├─ VERSION
│        ├─ japanese_dict_full.txt
│        └─ ds_cmudict-07b.txt
├─ audio_cache\
└─ outputs\
```

Qwen and Mel-Band weights can be downloaded on the first run, or pre-provisioned
under `models\qwen` and `models\melband`. The HubertFA weight is intentionally
not stored in Git because it is hundreds of megabytes. Download the official
`1218_hfa_model_new_dict` bundle from the
[HubertFA releases](https://github.com/wolfgitpr/HubertFA/releases) and extract
it so that `models\hubertfa\1218_hfa_model_new_dict\model.onnx` exists.

The repository vendors only the lightweight HubertFA runtime files needed by
the adapter and keeps the upstream [Apache-2.0 license](third_party/HubertFA-v0.0.7/LICENSE).
Model weights, evaluation audio, and training files are not part of the
repository.

## Run the recommended HubertFA path

Use UTF-8 lyrics with one lyric line per line:

```powershell
Set-Location D:\project\lyric-align

& .\.venv\Scripts\python.exe -m lyric_align.cli `
  .\song.flac .\lyrics.txt `
  --pipeline qwen `
  --aligner-backend hubertfa `
  --project-root . `
  --qwen-device cuda:0 `
  --dtype bf16 `
  --language ja `
  --asr-window 8 `
  --asr-overlap 1 `
  --pairing auto `
  --threshold 0.35 `
  --local-files-only `
  --output .\outputs\song.ttml
```

Remove `--local-files-only` on the first run if Qwen or Mel-Band weights still
need to be downloaded. Use `--overwrite-vocals` only when the cached
`audio_cache\*_vocals.wav` must be regenerated.

### Important options

| Option | Purpose |
|---|---|
| `--aligner-backend hubertfa` | Recommended singing backend; does not load Qwen3-ForcedAligner |
| `--qwen-device cuda:0` | Device for Qwen ASR and Mel-Band |
| `--dtype bf16` | GPU model precision used in the tested setup |
| `--asr-window 8 --asr-overlap 1` | A practical starting point for singing windows |
| `--pairing auto` | Let the matcher estimate lyric lines per ASR window |
| `--threshold 0.35` | Starting fuzzy-match threshold for sung mixed lyrics |
| `--local-files-only` | Refuse network/model downloads |
| `--overwrite-vocals` | Re-run Mel-Band instead of using the cached stem |
| `--interpolate` | Fill unmatched lines with guesses; use only when gaps are acceptable |

For HubertFA, `--min-lines` and `--max-lines` still exist for API/legacy
compatibility, but the singing path expands them to global blocks of roughly
10–15 lines and splits earlier when the padded audio would exceed the tested
100-second safety limit. The full 259-second song must not be sent to HubertFA
as one ONNX call.

## Alternative backends and legacy mode

### Qwen forced-alignment baseline

```powershell
& .\.venv\Scripts\python.exe -m lyric_align.cli `
  .\song.flac .\lyrics.txt `
  --pipeline qwen `
  --aligner-backend qwen `
  --project-root . `
  --qwen-device cuda:0 `
  --dtype bf16 `
  --language ja `
  --output .\outputs\song-qwen.ttml
```

This path loads Qwen3-ForcedAligner-0.6B for coarse timestamps and final
alignment. It is retained as a comparison/fallback path, but speech-oriented
Qwen models are less reliable on singing and repeated English hooks than
HubertFA.

### Existing non-GPU path

Without `--pipeline qwen`, the original faster-whisper/segments workflow stays
available:

```bash
# Transcribe and align with the legacy ASR path
lyric-align song.wav lyrics.txt -o out.lrc

# Use precomputed segments and skip ASR
lyric-align --segments segments.json lyrics.txt -f ttml -o out.ttml

# Convert an Audacity label track after manual correction
lyric-align --from-labels fixed.labels.txt -f lrc -o corrected.lrc
```

## Lyrics and mixed-language tokenization

`lyrics.txt` is UTF-8 plain text, one lyric line per line. Blank lines, lines
starting with `#`, and standalone ASCII section markers such as `[Verse 1]` are
skipped. Keep a mixed lyric line as one line; do not split Japanese and English
into separate model calls:

```text
今夜 stay with me
君だけ I need you
```

HubertFA converts continuous Japanese text with contextual `pykakasi` G2P and
then maps the aligned word interval back to Japanese characters. English words
are looked up in the CMU dictionary included in the HubertFA model bundle and
remain one output unit each:

```text
君と stay with me ずっと
↓
君  と  stay  with  me  ず っ と
```

Parentheses are punctuation, not a voice-role instruction. Inline text such as
`主句 (yeah)` can be aligned if the pronunciation exists in the dictionary,
but HubertFA does not know that it is harmony. A standalone `(yeah)` line is
treated like a section marker by the current lyric reader and may be skipped.
Lead and harmony are currently one mixed vocal stream; separate harmony tracks
would require a separate vocal stem and a second alignment layer.

## Output formats

The format is inferred from `-o`, or can be selected with `-f`:

| Format | Typical use | Fine-grained timing |
|---|---|---|
| `ttml` | AMLL/Apple-style rich lyrics | Japanese characters / English words |
| `elrc` | Enhanced LRC karaoke | Japanese characters / English words |
| `lrc` | Standard LRC | Line-level |
| `ass` | Video subtitles and effects | Add `--karaoke` for `\\k` timing |
| `srt` | Standard subtitles | Line-level |
| `vtt` | WebVTT / HTML5 track | Line-level |
| `json` | Programmatic post-processing | Full alignment records |
| `aud` | Audacity labels | Line-level |

TTML is the most complete output for the HubertFA path:

```powershell
# Enhanced LRC
lyric-align song.flac lyrics.txt -f elrc -o outputs\song.elrc

# Karaoke ASS
lyric-align song.flac lyrics.txt -f ass --karaoke -o outputs\song.ass

# Write every supported format next to the requested base name
lyric-align song.flac lyrics.txt -f all -o outputs\song.ttml
```

When a line cannot be aligned reliably, the pipeline keeps an honest gap and
returns a non-zero status after writing the output. `--interpolate` is opt-in;
it fills line-level gaps with guesses and does not create real character
timestamps.

## Project layout

```text
src/lyric_align/
├─ cli.py                 CLI and format dispatch
├─ pipeline.py            Mel-Band → ASR → matching → alignment orchestration
├─ melband_backend.py     CUDA vocal separation
├─ qwen_backend.py        Qwen ASR and optional Qwen FA adapter
├─ hubertfa_backend.py    HubertFA ONNX/G2P adapter
├─ chunking.py            bounded alignment chunks
├─ gapfix.py              local timestamp/gap repair
└─ formats.py             LRC/ELRC/SRT/VTT/ASS/TTML/JSON/Audacity output

third_party/HubertFA-v0.0.7/
└─ runtime-only upstream files used by the ONNX adapter

scripts/setup_d_env.ps1   D-drive Windows environment setup
tests/                    unit, adapter, pipeline, and offline-path tests
```

## Verification

The current implementation has been validated with:

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

The checked-in branch passes **97 tests**. A real mixed Japanese/English song
run on an RTX 5070 Ti produced 38/38 lyric lines, used
`CUDAExecutionProvider`, and emitted TTML with no zero-duration spans or
parent/child timestamp violations. Exact line boundaries still depend on the
quality of the supplied lyrics and the ASR coarse window; HubertFA cannot
recover a line that was matched to the wrong repeated section.

## License

The main project is MIT-licensed. The vendored HubertFA runtime retains its
upstream license in
[`third_party/HubertFA-v0.0.7/LICENSE`](third_party/HubertFA-v0.0.7/LICENSE).
