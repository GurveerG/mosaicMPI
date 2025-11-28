
from . import cpus_available

from types import SimpleNamespace
from typing import Union, Optional, Literal
from collections.abc import Collection, Iterable

import pandas as pd
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from tqdm import tqdm


def order_genesets(df: pd.DataFrame, ascending=False):
    """Order genesets by the column with highest significance, followed by the max significance value.

    :param df: A geneset × program/sample matrix with -log10(pvals) as values.
    :type df: pd.DataFrame
    """
    if df.shape[0] > 0:
        ordered = []
        if ascending:
            # sort gene sets by lowest column and then lowest value of that column
            stats = pd.DataFrame({"col": df.idxmin(axis=1), "min": df.min(axis=1)})
            for col in df.columns:
                ordered.append(stats[stats["col"] == col].sort_values("min", ascending=ascending))
        else:
            # sort gene sets by highest column and then highest value of that column
            stats = pd.DataFrame({"col": df.idxmax(axis=1), "max": df.max(axis=1)})
            for col in df.columns:
                ordered.append(stats[stats["col"] == col].sort_values("max", ascending=ascending))
        ordered = pd.concat(ordered)
        ordered_df = df.loc[ordered.index]
    else:
        ordered_df = df
    return ordered_df


def read_gmt(filepath: str) -> pd.DataFrame:
    rows = []
    seen_pathways = set()

    with open(filepath, "r") as f:
        for line in f:
            entries = line.strip().split("\t")
            pathway, description, genes = entries[0], entries[1], entries[2:]

            # Ensure unique pathway names
            base_name = pathway
            suffix = 1

            while pathway in seen_pathways:
                pathway = f"{base_name}_{suffix}"
                suffix += 1
            
            seen_pathways.add(pathway)
            rows.append({"pathway": pathway,
                         "description": description,
                         "genes": genes})

    return pd.DataFrame(rows)


def program_ssgsea(program_df: pd.DataFrame,
                      gene_sets: str = "GO_Biological_Process_2023",
                      min_intersection: int = 5,
                      max_intersection: int = 500,
                      cpus: int = cpus_available) -> SimpleNamespace:

    from gseapy import ssgsea
    result = SimpleNamespace()

    # map column MultiIndex to a range index for gseapy compatibility
    prog_ids = program_df.columns
    output_colnames = program_df.columns.names
    program_df.columns = [str(i) for i in range(prog_ids.size)]
    prog_id_mapper = {col: id for col, id in zip(program_df.columns, prog_ids)}
    result.gseapy_output = ssgsea(data=program_df, gene_sets=gene_sets, min_size=min_intersection, max_size=max_intersection, threads=cpus)
    # extract NES scores
    result.prog_nes = result.gseapy_output.res2d.pivot(index="Term", columns="Name", values="NES")
    result.prog_nes.columns = result.prog_nes.columns.map(prog_id_mapper).rename(output_colnames)
    # order scores by column and significance
    result.prog_nes = order_genesets(result.prog_nes).astype(np.float32)
    return result

# TODO: Add support for ordered genesets
# TODO: Add support for some of the current functionality in the R code
def program_gprofiler(program_df: pd.DataFrame,
                      species: Literal["hsapiens", "mmusculus"],
                      n_hsg: int = 1000,
                      gene_sets: Collection[str] = [],
                      no_iea: bool = True,
                      min_intersection: int = 5,
                      max_intersection: int = 500,
                      ordered: bool = False,
                      batch_size: int = 20,
                      show_progress_bar: bool = True
                      ) -> SimpleNamespace:
    
    from gprofiler import GProfiler

    result = SimpleNamespace()
    result.background = program_df.dropna(how="all").index.to_list()  # all genes in program_df
    result.hsg = program_df.rank(ascending=False) <= n_hsg  # bool dataframe of high scoring genes across programs
    prog_names_str = program_df.columns.map(str)  # gProfiler multi-query only supports string names for queries
    # but we can decode them
    if program_df.columns.nlevels == 1:
        prog_names_str_decoder = {progstr: [prog] for progstr, prog in zip(prog_names_str, program_df.columns)}
    else:
        prog_names_str_decoder = {progstr: list(prog) for progstr, prog in zip(prog_names_str, program_df.columns)}
    prog_level_names = program_df.columns.names
    result.query = {prog_str: genes[genes].index.to_list() for (prog, genes), prog_str in zip(result.hsg.items(), prog_names_str)}
    result.gene_sets = gene_sets
    result.no_iea = no_iea

    gp = GProfiler(return_dataframe=True)

    result.gprofiler_output = []
    batch_query = []
    for i, query in enumerate(tqdm(result.query.keys(), total=len(result.query), unit="program", desc="Querying g:Profiler", disable=not show_progress_bar), start=1):
        batch_query.append(query)
        if i % batch_size == 0 or i == len(result.query):
            batch_result = gp.profile(organism=species, query={q: result.query[q] for q in batch_query},
                                    sources=gene_sets, no_iea=no_iea, background=result.background)
            if ordered:
                batch_result = gp.profile(organism=species, query={q: result.query[q] for q in batch_query},
                                          sources=gene_sets, no_iea=no_iea, background=result.background)
            else:
                batch_result = gp.profile(organism=species, query={q: result.query[q] for q in batch_query},
                                          sources=gene_sets, no_iea=no_iea, background=result.background)
            result.gprofiler_output.append(batch_result)
            batch_query = []
        
    result.gprofiler_output = pd.concat(result.gprofiler_output)
    result.gprofiler_output["-log10pval"] = -np.log10(result.gprofiler_output["p_value"])
    subset = ((result.gprofiler_output["intersection_size"] <= max_intersection) &
              (result.gprofiler_output["intersection_size"] >= min_intersection))
    result.summary = result.gprofiler_output[subset].pivot(index=["source", "native", "name", "description", "term_size"], columns="query")
    stats = ["-log10pval", "query_size", "intersection_size"]
    result.summary = result.summary[stats]
    result.summary.columns = pd.MultiIndex.from_tuples([([c[0]] + prog_names_str_decoder[c[1]]) for c in result.summary.columns], names=["stat"] + prog_level_names)

    # conform column order to input dataframe
    if program_df.columns.nlevels == 1:
        sorted_cols = pd.MultiIndex.from_tuples([(stat, prog) for stat in stats for prog in program_df.columns])
    else:
        sorted_cols = pd.MultiIndex.from_tuples([tuple([stat] + list(prog)) for stat in stats for prog in program_df.columns])
    result.summary = result.summary.reindex(columns=sorted_cols)
    return result


def program_marker_gene_scores(program_df: pd.DataFrame,
                               n_hsg: int = 1000,
                               gene_sets: Collection[str] = [],
                               scale_output: bool = False,
                               filter_output: bool = False,
                               detailed: bool = True) -> SimpleNamespace:
    # Create a result namespace to store the results
    result = SimpleNamespace()
    result.input_genes = program_df.dropna(how="all").index.to_list()
    result.n_geps = program_df.shape[1]
    result.n_genes = program_df.shape[0]

    # Load and preprocess gene sets
    gene_sets_df = read_gmt(gene_sets)
    gene_sets = {row["pathway"]: {gene.upper() for gene in row["genes"]} for _, row in gene_sets_df.iterrows()}

    # Standardize program_df gene names
    program_df.index = program_df.index.str.upper()

    # Sort top scoring genes per GEP
    top_rank_genes = {gep: program_df[gep].sort_values(ascending=False).head(n_hsg).index.tolist() for gep in program_df.columns}

    all_pathways_df = []
    for pathway, marker_genes in tqdm(gene_sets.items(), desc="Scoring Gene Sets"):
        result_data = {"pathway": pathway, "n_markers": len(marker_genes)}

        for gep in program_df.columns:
            top_genes = top_rank_genes[gep]
            top_genes_set = set(top_genes)
            gene_pos = {gene: idx + 1 for idx, gene in enumerate(top_genes)}

            intersect_genes = marker_genes & top_genes_set
            int_N = len(intersect_genes)

            if int_N == 0:
                score = 0
                intersect_genes_sorted = []
                intersect_ranks = []
            else:
                intersect_genes_sorted = sorted(intersect_genes, key=lambda g: gene_pos[g])
                intersect_ranks = [gene_pos[g] for g in intersect_genes_sorted]
                ranks_sum = np.sum(1 / np.array(intersect_ranks))
                score = (int_N * ranks_sum) / np.log10(len(marker_genes)) if len(marker_genes) > 1 else 0

            gep_id = f"K{gep[0]}_GEP{gep[1]}"
            if detailed:
                result_data.update({
                    f"{gep_id}_n_Int": int_N,
                    f"{gep_id}_genes_Int": '|'.join(intersect_genes_sorted),
                    f"{gep_id}_ranks_Int": '|'.join(map(str, intersect_ranks)),
                    f"{gep_id}_top{len(top_genes)}_marker_score": score,
                })
            else:
                result_data[f"{gep_id}_marker_score"] = score

        all_pathways_df.append(result_data)

    final_df = pd.DataFrame(all_pathways_df).set_index("pathway")

    # Identify score columns and compute max score
    score_cols = [col for col in final_df.columns if col.endswith("_marker_score")]
    final_df = final_df.copy()
    final_df["maxScore"] = final_df[score_cols].max(axis=1)

    # TODO: Determine how to scale the scores the same way as R does it
    if filter_output:
        final_df = final_df.loc[:, final_df.any()]
    if scale_output:
        final_df = (final_df - final_df.mean()) / final_df.std()
    
    result.summary = final_df

    return result

# TODO: Add an automatic function for all program correlations
# TODO: Add an automatic function for all program enrichment bar plots using the metadata
# TODO: Add an automatic function for the number of programs per observation with usage above some threshold
# TODO: Add a function to plot the program usage density for every program