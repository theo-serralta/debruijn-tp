"""Tests for graph build"""
import pytest
import os
import networkx as nx
from pathlib import Path
from .context import debruijn
from debruijn import read_fastq
from debruijn import cut_kmer
from debruijn import build_kmer_dict
from debruijn import build_graph


def test_read_fastq():
    """Test fastq reading"""
    fastq_reader = read_fastq(Path(__file__).parent / "test_two_reads.fq")
    assert next(fastq_reader) == "TCAGAGCTCTAGAGTTGGTTCTGAGAGAGATCGGTTACTCGGAGGAGGCTGTGTCACTCATAGAAGGGATCAATCACACCCACCACGTGTACCGAAACAA"
    assert next(fastq_reader) == "TTTGAATTACAACATCCATATGTTCTTGATGCTGGAATTCCAATATCTCAGTTGACAGTGTGCCCTCACCAGTGGATCAATTTACGAACCAACAATTGTG"


def test_cut_kmer():
    """Test cut_kmer"""
    kmer_reader = cut_kmer("TCAGA", 3)
    assert next(kmer_reader) == "TCA"
    assert next(kmer_reader) == "CAG"
    assert next(kmer_reader) == "AGA"


def test_build_kmer_dict():
    """Test kmer dict"""
    kmer_dict = build_kmer_dict(Path(__file__).parent / "test_build.fq", 3)
    assert(len(kmer_dict.keys()) == 4)
    assert "TCA" in kmer_dict
    assert "CAG" in kmer_dict
    assert "AGA" in kmer_dict
    assert "GAG" in kmer_dict
    assert kmer_dict["AGA"] == 2

def test_build_graph():
    """Test build graph"""
    kmer_dict = {'GAG': 1, 'CAG': 1, 'AGA': 2, 'TCA': 1}
    graph = build_graph(kmer_dict)
    #TCAGAGA
    #TCA TC CA
    #CAG CA AG
    #AGA AG GA
    #GAG GA AG
    #AGA AG GA
    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 4
    assert "AG" in graph
    assert "GA" in graph
    assert graph.edges["AG", "GA"]['weight'] == 2