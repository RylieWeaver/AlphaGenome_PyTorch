#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
#
# Downloads publicly available data from:
# - UCSC GRCh38 chromosome 1:
#   https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/README.txt
# - ENCODE file ENCFF877MMK from experiment ENCSR601DZY:
#   https://www.encodeproject.org/files/ENCFF877MMK/
#
# Data-use and citation policies:
# - UCSC: https://genome.ucsc.edu/license/
# - ENCODE: https://www.encodeproject.org/help/citing-encode/
#
# The SPDX identifier applies to this script. Downloaded data remain subject
# to their source attribution and citation policies.

"""Download and index the data used by the proceedings training curves."""

import gzip
import shutil
from pathlib import Path
from urllib.request import urlretrieve

import pysam

from utils import DATA_DIR, DEFAULT_BIGWIG, DEFAULT_FASTA


FASTA_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr1.fa.gz"
BIGWIG_URL = "https://www.encodeproject.org/files/ENCFF877MMK/@@download/ENCFF877MMK.bigWig"


def download(url: str, destination: Path) -> None:
    if destination.exists():
        print(f"Using {destination}")
        return
    print(f"Downloading {destination.name}")
    urlretrieve(url, destination)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DEFAULT_FASTA.exists():
        compressed_fasta = DEFAULT_FASTA.with_suffix(".fa.gz")
        download(FASTA_URL, compressed_fasta)
        print(f"Decompressing {compressed_fasta.name}")
        with gzip.open(compressed_fasta, "rb") as source:
            with DEFAULT_FASTA.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        compressed_fasta.unlink()
    else:
        print(f"Using {DEFAULT_FASTA}")

    fasta_index = Path(f"{DEFAULT_FASTA}.fai")
    if not fasta_index.exists():
        print(f"Indexing {DEFAULT_FASTA.name}")
        pysam.faidx(str(DEFAULT_FASTA))
    else:
        print(f"Using {fasta_index}")

    download(BIGWIG_URL, DEFAULT_BIGWIG)
    print(f"Data are ready in {DATA_DIR}")


if __name__ == "__main__":
    main()
