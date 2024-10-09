#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

import argparse
import os
import sys
import random
import statistics
import textwrap
from pathlib import Path
from random import randint
from typing import Iterator, Dict, List, Any
import matplotlib
import matplotlib.pyplot as plt
from networkx import (
    DiGraph,
    all_simple_paths,
    lowest_common_ancestor,
    has_path,
    random_layout,
)
from networkx.drawing.nx_pylab import draw_networkx_nodes, draw_networkx_edges


matplotlib.use("Agg")
random.seed(9001)

__author__ = "Theo Serralta"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Theo Serralta"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Theo Serralta"
__email__ = "theo.serralta@gmail.com"
__status__ = "Developpement"



def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description=__doc__, usage="{0} -h".format(sys.argv[0])
    )
    parser.add_argument(
        "-i", dest="fastq_file", type=isfile, required=True, help="Fastq file"
    )
    parser.add_argument(
        "-k", dest="kmer_size", type=int, default=22, help="k-mer size (default 22)"
    )
    parser.add_argument(
        "-o",
        dest="output_file",
        type=Path,
        default=Path(os.curdir + os.sep + "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)",
    )
    parser.add_argument(
        "-f", dest="graphimg_file", type=Path, help="Save graph as an image (png)"
    )
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with fastq_file.open() as f:
        while True:
            # Lire 4 lignes du fichier pour chaque bloc
            f.readline()  # Ignore l'identifiant
            sequence = f.readline().strip()  # Lecture
            f.readline()  # Ignore la ligne avec '+'
            f.readline()  # Ignore la qualité
            if not sequence:  # Fin du fichier
                break
            yield sequence


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    # Lire les séquences du fichier FASTQ
    for read in read_fastq(fastq_file):
        # Pour chaque k-mer dans la séquence de lecture
        for kmer in cut_kmer(read, kmer_size):
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()
    # Construire les arcs avec poids basés sur les occurrences de chaque k-mer
    for kmer, count in kmer_dict.items():
        prefix = kmer[:-1]  # Préfixe du k-mer
        suffix = kmer[1:]   # Suffixe du k-mer
        graph.add_edge(prefix, suffix, weight=count)
    return graph

def remove_paths(
    graph: DiGraph,
    path_list: List[List[str]],
    delete_entry_node: bool,
    delete_sink_node: bool,
) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if delete_entry_node and delete_sink_node:
            # Supprimer tous les nœuds
            nodes_to_remove = path
        elif delete_entry_node:
            # Supprimer tous les nœuds sauf le dernier
            nodes_to_remove = path[:-1]
        elif delete_sink_node:
            # Supprimer tous les nœuds sauf le premier
            nodes_to_remove = path[1:]
        else:
            # Supprimer tous les nœuds sauf le premier et le dernier
            nodes_to_remove = path[1:-1]

        # Supprimer les nœuds spécifiés du graphe
        graph.remove_nodes_from(nodes_to_remove)
    return graph


def select_best_path(
    graph: DiGraph,
    path_list: List[List[str]],
    path_length: List[int],
    weight_avg_list: List[float],
    delete_entry_node: bool = False,
    delete_sink_node: bool = False,
) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Étape 1 : Comparer les poids moyens
    if statistics.stdev(weight_avg_list) > 0:
        # Sélectionner l'indice du chemin avec le poids moyen le plus élevé
        best_index = weight_avg_list.index(max(weight_avg_list))
    elif statistics.stdev(path_length) > 0:
        # Étape 2 : Comparer les longueurs si les poids moyens sont identiques
        best_index = path_length.index(max(path_length))
    else:
        # Étape 3 : Choix aléatoire si tous les critères sont identiques
        best_index = randint(0, len(path_list) - 1)

    # Identifier les chemins à supprimer
    paths_to_remove = [path for i, path in enumerate(path_list) if i != best_index]

    # Supprimer les chemins non sélectionnés du graphe
    graph = remove_paths(graph, paths_to_remove, delete_entry_node, delete_sink_node)

    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)]
    )


def solve_bubble(graph: DiGraph, ancestor_node: str, descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Trouver tous les chemins simples entre l'ancêtre et le descendant
    path_list = list(all_simple_paths(graph, ancestor_node, descendant_node))

    # Calculer la longueur de chaque chemin et le poids moyen
    path_length = [len(path) for path in path_list]
    weight_avg_list = [path_average_weight(graph, path) for path in path_list]

    # Appeler select_best_path pour conserver le meilleur chemin
    graph = select_best_path(
        graph,
        path_list,
        path_length,
        weight_avg_list,
        delete_entry_node=False,
        delete_sink_node=False
    )

    return graph

def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    # Parcourir chaque noeud du graphe pour identifier les bulles
    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        # Si le noeud a plus d'un prédécesseur, il pourrait y avoir une bulle
        if len(predecessors) > 1:
            # Générer les combinaisons de prédécesseurs sans itertools
            for i in range(len(predecessors)):
                for j in range(i + 1, len(predecessors)):
                    pred1, pred2 = predecessors[i], predecessors[j]
                    # Trouver l'ancêtre commun le plus proche des deux prédécesseurs
                    ancestor = lowest_common_ancestor(graph, pred1, pred2)
                    # Si un ancêtre commun est trouvé, nous avons détecté une bulle
                    if ancestor is not None:
                        # Résoudre la bulle entre cet ancêtre et le noeud
                        graph = solve_bubble(graph, ancestor, node)
                        # Appel récursif pour vérifier si d'autres bulles subsistent
                        return simplify_bubbles(graph)
    # Retourner le graphe une fois toutes les bulles résolues
    return graph

def solve_entry_tips(graph: DiGraph, starting_nodes: List[Any]) -> DiGraph:
    """Remove entry tips from the graph.

    :param graph: (nx.DiGraph) A directed graph object.
    :param starting_nodes: (list) A list of starting nodes.
    :return: (nx.DiGraph) A directed graph object without entry tips.
    """
    # For all nodes in the graph
    for node in graph.nodes:
        predecessors = list(graph.predecessors(node))
        # If node has more than one predecessor
        if len(predecessors) > 1:
            # Initialize lists to collect paths
            path_list = []
            path_length = []
            weight_avg_list = []
            valid_paths = False
            # For each starting node
            for start_node in starting_nodes:
                # Check if there is a path from start_node to node
                if has_path(graph, start_node, node):
                    # Get all simple paths from start_node to node
                    paths = list(all_simple_paths(graph, start_node, node))
                    for path in paths:
                        if len(path) >= 2:
                            path_list.append(path)
                            path_length.append(len(path))
                            weight_avg_list.append(path_average_weight(graph, path))
                            valid_paths = True
            if valid_paths and len(path_list) > 1:
                # We have multiple paths from starting nodes to this node
                # Select the best path and remove the others
                graph = select_best_path(
                    graph,
                    path_list,
                    path_length,
                    weight_avg_list,
                    delete_entry_node=True,
                    delete_sink_node=False
                )
                # Update starting nodes
                starting_nodes = get_starting_nodes(graph)
                # Restart the process
                return solve_entry_tips(graph, starting_nodes)
    # No more entry tips to remove
    return graph

def solve_out_tips(graph: DiGraph, ending_nodes: List[str]) -> DiGraph:
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :param ending_nodes: (list) A list of ending nodes
    :return: (nx.DiGraph) A directed graph object
    """
    # For all nodes in the graph
    for node in graph.nodes:
        successors = list(graph.successors(node))
        # If node has more than one successor
        if len(successors) > 1:
            # Initialize lists to collect paths
            path_list = []
            path_length = []
            weight_avg_list = []
            valid_paths = False
            # For each ending node
            for end_node in ending_nodes:
                # Check if there is a path from current node to end_node
                if has_path(graph, node, end_node):
                    # Get all simple paths from current node to end_node
                    paths = list(all_simple_paths(graph, node, end_node))
                    for path in paths:
                        if len(path) >= 2:
                            path_list.append(path)
                            path_length.append(len(path))
                            weight_avg_list.append(path_average_weight(graph, path))
                            valid_paths = True
            if valid_paths and len(path_list) > 1:
                # We have multiple paths from this node to ending nodes
                # Select the best path and remove the others
                graph = select_best_path(
                    graph,
                    path_list,
                    path_length,
                    weight_avg_list,
                    delete_entry_node=False,
                    delete_sink_node=True
                )
                # Update ending nodes
                ending_nodes = get_sink_nodes(graph)
                # Restart the process
                return solve_out_tips(graph, ending_nodes)
    # No more out tips to remove
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node in graph.nodes if not list(graph.predecessors(node))]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node in graph.nodes if not list(graph.successors(node))]

def get_contigs(
    graph: DiGraph, starting_nodes: List[str], ending_nodes: List[str]
) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []
    for start in starting_nodes:
        for end in ending_nodes:
            if has_path(graph, start, end):
                for path in all_simple_paths(graph, start, end):
                    contig = path[0]  # Initialise avec le premier k-mer
                    for node in path[1:]:
                        contig += node[-1]  # Ajouter le dernier caractère de chaque noeud
                    contigs.append((contig, len(contig)))
    return contigs

def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with output_file.open("w") as f:
        for i, (contig, length) in enumerate(contigs_list):
            f.write(f">contig_{i} len={length}\n")
            f.write(textwrap.fill(contig, width=80) + "\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:
    """Draw the graph

    :param graph: (DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] > 3]
    esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d["weight"] <= 3]

    pos = random_layout(graph)
    draw_networkx_nodes(graph, pos, node_size=6, ax=ax)
    draw_networkx_edges(graph, pos, edgelist=elarge, width=6, ax=ax)
    draw_networkx_edges(
        graph,
        pos,
        edgelist=esmall,
        width=6,
        alpha=0.5,
        edge_color="b",
        style="dashed",
        ax=ax,
    )
    # Save the figure associated with 'fig'
    fig.savefig(graphimg_file.resolve())

# ==============================================================
# Main program
# ==============================================================
def main() -> None:
    """
    Main program function.
    """
    print("Starting program")
    # Retrieve arguments
    args = get_arguments()

    # Step 1: Build k-mer dictionary
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)

    # Step 2: Build De Bruijn graph
    graph = build_graph(kmer_dict)

    # Step 3: Identify starting and ending nodes
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)

    # Step 4: Simplify the graph
    # Resolve bubbles
    graph = simplify_bubbles(graph)

    # Update starting and ending nodes after simplification
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)

    # Resolve entry tips
    graph = solve_entry_tips(graph, starting_nodes)

    # Update ending nodes after resolving entry tips
    ending_nodes = get_sink_nodes(graph)

    # Resolve out tips
    graph = solve_out_tips(graph, ending_nodes)

    # Step 5: Extract contigs
    contigs = get_contigs(graph, starting_nodes, ending_nodes)

    # Step 6: Save contigs to a FASTA file
    save_contigs(contigs, args.output_file)

    # Optional: Generate an image of the graph
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)

if __name__ == "__main__":  # pragma: no cover
    main()
