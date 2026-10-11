#!/usr/bin/env bash
# get_dataset.sh — download the raw HaGRID pieces this use case needs.
#
# Run from anywhere:
#   custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
#
# Everything lands in datasets/_hagrid_raw/ (gitignored, shared across use
# cases) and is cached — a re-run re-uses whatever is already there and only
# fetches what is missing. build_dataset.py then converts it into
# datasets/detect/hagrid_gestures/.
#
# Two sources, selected with HAGRID_SOURCE:
#
#   subsample (default)  ONE 1.01 GB zip — 31,833 images / 19 classes / 384p,
#                        images AND annotations together, hosted on Hugging Face
#                        by cj-mills, no auth needed. About 1,768 images per
#                        class, which comfortably exceeds MAX_PER_CLASS=2000 for
#                        the five gestures we keep. Verified reachable: HTTP 200,
#                        application/zip, 1,009,688,861 bytes.
#
#   full512              upstream's per-gesture zips from the 512p release plus a
#                        separate annotations zip. The full 512p release is ~100 GB
#                        across all 18 classes, so even five gestures is a big
#                        transfer. Only worth it if the subsample underfits.
#
# Why not upstream for the default: hukenovs/hagrid only offers the full-size
# releases (HaGRID 723 GB FullHD, HaGRIDv2 1.5 TB, or ~100 GB at 512p). The
# 384p subsample is the same data and annotations at a size that actually fits
# this project — we train at 288 px, so 384p source images lose us nothing.
#
# Every URL is overridable and is HEAD-checked BEFORE any large transfer starts,
# because HaGRID re-hosts its archives between releases:
#
#   HAGRID_SUBSAMPLE_URL=<url> custom_dataset_usecases/detect/hagrid_gestures/get_dataset.sh
#
# Alternatives if the default ever dies:
#   cj-mills/hagrid-sample-500k-384p   13.4 GB, 509k images, same layout
#   Kaggle innominate817/hagrid-sample-30k-384p   the same 30k zip, needs a login
#   upstream download table: https://github.com/hukenovs/hagrid#downloads
#
# The five gestures are duplicated in build_dataset.py (GESTURES). Change both
# together if you swap one out.

set -euo pipefail

GESTURES=(call fist like palm peace)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RAW="$REPO_ROOT/datasets/_hagrid_raw"

HAGRID_SOURCE="${HAGRID_SOURCE:-subsample}"

HF_BASE="https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main"
HAGRID_SUBSAMPLE_URL="${HAGRID_SUBSAMPLE_URL:-$HF_BASE/hagrid-sample-30k-384p.zip}"

# Upstream (SberCloud) — used only by HAGRID_SOURCE=full512. These are the paths
# from the hukenovs/hagrid README and are NOT verified; preflight will tell you.
SBER_BASE="https://n-usr-2uzac.s3pd12.sbercloud.ru/b-usr-2uzac-mv4/hagrid"
HAGRID_FULL_BASE_URL="${HAGRID_FULL_BASE_URL:-$SBER_BASE/hagrid_dataset_512}"
HAGRID_ANN_URL="${HAGRID_ANN_URL:-$SBER_BASE/annotations/ann_train_val.zip}"

mkdir -p "$RAW"

# ── Helpers ───────────────────────────────────────────────────────────────────

# HEAD-check a URL and report its size. Fails the script on a non-2xx/3xx.
preflight() {
  local url="$1" name="$2" code len
  local head
  head="$(curl -sIL --max-time 30 "$url" || true)"
  code="$(printf '%s' "$head" | awk '/^HTTP/ {c=$2} END {print c}')"
  len="$(printf '%s' "$head" | awk 'tolower($1)=="content-length:" {l=$2} END {print l+0}')"
  if [[ "$code" != 2* ]]; then
    echo "  ✗ $name — HTTP ${code:-no-response}" >&2
    echo "" >&2
    echo "    $url" >&2
    echo "" >&2
    echo "    HaGRID re-hosts its archives between releases. Find the current link in" >&2
    echo "    https://github.com/hukenovs/hagrid#downloads and pass it in, e.g." >&2
    echo "      HAGRID_SUBSAMPLE_URL=<url> $0" >&2
    return 1
  fi
  if (( len > 0 )); then
    printf '  ✓ %-28s %6.2f GB\n' "$name" "$(echo "$len" | awk '{print $1/1e9}')"
  else
    printf '  ✓ %-28s (size unknown)\n' "$name"
  fi
}

# Download to a .part file and rename on success, so an interrupted transfer is
# never mistaken for a cached complete one. -C - resumes a partial .part.
fetch() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    printf '  cached  %-28s %6.2f GB\n' "$(basename "$dest")" \
      "$(echo "$(wc -c <"$dest")" | awk '{print $1/1e9}')"
    return 0
  fi
  echo "  downloading $(basename "$dest") ..."
  curl -L --progress-bar -C - -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

# Unzip only if the expected marker directory is absent, so re-runs are cheap.
unpack() {
  local archive="$1" marker="$2"
  if [[ -e "$marker" ]]; then
    echo "  already unpacked: $(basename "$marker")"
    return 0
  fi
  echo "  unpacking $(basename "$archive") ..."
  unzip -q "$archive" -d "$RAW"
}

# ── Fetch ─────────────────────────────────────────────────────────────────────

echo "HaGRID raw cache: $RAW"
echo "Source: $HAGRID_SOURCE"
echo ""

case "$HAGRID_SOURCE" in
subsample)
  echo "[1/3] Checking URLs..."
  preflight "$HAGRID_SUBSAMPLE_URL" "hagrid-sample-30k-384p"
  echo ""
  echo "[2/3] Downloading..."
  fetch "$HAGRID_SUBSAMPLE_URL" "$RAW/hagrid-sample-30k-384p.zip"
  echo ""
  echo "[3/3] Unpacking..."
  # The subsample zip carries both the images and their annotations. Its exact
  # top-level directory name has changed between releases, so key the
  # already-unpacked check off the archive's own first entry rather than a
  # hardcoded path.
  # Take the first entry WITHOUT piping into head/sed -q: those close the pipe
  # early, unzip dies of SIGPIPE, and `set -o pipefail` turns that into exit 141
  # even though nothing actually went wrong. Capture the whole listing (it is
  # only a few MB of text) and slice it with parameter expansion instead.
  listing="$(unzip -Z1 "$RAW/hagrid-sample-30k-384p.zip")"
  top="${listing%%$'\n'*}"   # first line
  top="${top%%/*}"           # its top-level directory component
  unpack "$RAW/hagrid-sample-30k-384p.zip" "$RAW/$top"
  ;;

full512)
  echo "[1/3] Checking URLs (${#GESTURES[@]} gestures + annotations)..."
  preflight "$HAGRID_ANN_URL" "ann_train_val"
  for g in "${GESTURES[@]}"; do
    preflight "$HAGRID_FULL_BASE_URL/$g.zip" "$g.zip"
  done
  echo ""
  echo "  Note: the 512p release is ~100 GB across 18 classes; five gestures is still"
  echo "  tens of GB. Sizes reported above are the real ones — HAGRID_SOURCE=subsample"
  echo "  is 1.01 GB for the same annotations at 384p, and 288 is our train size anyway."
  echo ""
  echo "[2/3] Downloading..."
  fetch "$HAGRID_ANN_URL" "$RAW/ann_train_val.zip"
  for g in "${GESTURES[@]}"; do
    fetch "$HAGRID_FULL_BASE_URL/$g.zip" "$RAW/$g.zip"
  done
  echo ""
  echo "[3/3] Unpacking..."
  unpack "$RAW/ann_train_val.zip" "$RAW/ann_train_val"
  for g in "${GESTURES[@]}"; do
    unpack "$RAW/$g.zip" "$RAW/$g"
  done
  ;;

*)
  echo "Unknown HAGRID_SOURCE='$HAGRID_SOURCE' (expected 'subsample' or 'full512')" >&2
  exit 1
  ;;
esac

echo ""
echo "Raw HaGRID ready in $RAW"
echo ""
echo "Next:"
echo "  python3 custom_dataset_usecases/detect/hagrid_gestures/build_dataset.py"
