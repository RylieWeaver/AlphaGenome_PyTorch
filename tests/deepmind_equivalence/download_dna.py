"""Download deterministic chromosome-1 DNA chunks for equivalence tests."""

# External
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


CHUNK_LENGTH = 2**20
SEARCH_LENGTH = 2**20
ASSEMBLIES = {
    "human": "hg38",
    "mouse": "mm39",
}


def _sequence(assembly, start, end):
    url = (
        "https://api.genome.ucsc.edu/getData/sequence"
        f"?genome={assembly};chrom=chr1;start={start};end={end}"
    )
    request = Request(
        url,
        headers={"User-Agent": "AlphaGenome-PyTorch-equivalence-tests"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)["dna"].upper()


def _first_sequence_base(assembly):
    """Return the first chr1 coordinate that is not an assembly-gap N."""
    start = 0
    while True:
        sequence = _sequence(assembly, start, start + SEARCH_LENGTH)
        if not sequence:
            raise RuntimeError(f"{assembly} chr1 returned no sequence")
        for offset, base in enumerate(sequence):
            if base != "N":
                return start + offset
        start += len(sequence)


def _write_fasta(path, name, assembly, start, sequence):
    end = start + len(sequence)
    lines = [
        f">{name} {assembly}:chr1:{start}-{end}",
        *(sequence[index:index + 80] for index in range(0, len(sequence), 80)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_directory",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().with_name("dna"),
    )
    return parser.parse_args(argv)


def main(argv=None):
    output_directory = parse_args(argv).output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    for organism, assembly in ASSEMBLIES.items():
        start = _first_sequence_base(assembly)
        sequence = _sequence(assembly, start, start + CHUNK_LENGTH)
        if len(sequence) != CHUNK_LENGTH:
            raise RuntimeError(
                f"{assembly} returned {len(sequence)} of {CHUNK_LENGTH} bases"
            )
        output = output_directory / f"{organism}_{assembly}_chr1.fa"
        _write_fasta(output, organism, assembly, start, sequence)
        print(output)


if __name__ == "__main__":
    main()
