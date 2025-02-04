#!/usr/bin/env python3
"""
Git Graph Visualizer with Multi–Level Visual Grouping

This script operates in two modes:

1. Git Object Graph Mode (default):
   Parses a Git repository (its .git folder) and generates a Graphviz diagram
   of its commits, trees, blobs, and branches. Various options control colors,
   edge styling, metadata, and now grouping of repetitive elements.

2. Combined Repository Base Mode (--repo-base):
   Recursively scans a base folder for all Git repositories (directories containing
   a “.git” folder) and produces one PDF that shows the overall project directory
   structure plus, side by side, subgraphs for each repository’s Git object graph.

New grouping functionality (when “--group” is enabled) lets you “condense” nodes
that share common text (for example, many blobs with the same name) into a single
record–shaped node. The record’s left field shows the common text and the right
field lists the differences (for example, a short hash). Unique, visible colors are
generated per group and applied to both the node and connecting edges. The degree
of visual condensation is adjustable with “--group-condense-level” and the types
of nodes to group are given by “--group-types” (a comma–separated list of types among:
blob, tree, commit, branch).

Other options include:
  --colors             Enable unique colors for node types and add a legend.
  -o, --output-only    Save the output to file only (don’t open the viewer).
  --local              Show only local branches (skip remote branches).
  -P N, --predecessors N
                       Limit the number of predecessor (parent) edges shown per commit.
                       Extra edges are drawn dotted.
  -S N, --successors N
                       Limit the number of successor (child) edges shown per commit.
                       Extra edges are drawn dotted.
  --dotted [remote/local/all]
                       Force the given group(s) of branch edges to be drawn dotted.
  -M META, --metadata META
                       Comma–separated list of metadata keys to include in commit labels.
                       Available keys: flags, author, short, gitstat. Default: all.
  --group              Enable multi–level grouping of nodes with similar text.
  --group-types TYPE   Comma–separated list of node types to group.
                       Options: blob, tree, commit, branch. Default: blob,tree.
  --group-condense-level N
                       An integer (default 1) controlling the amount of detail in grouped nodes.
  --repo-base PATH     Instead of a single Git repo, scan the given base folder for
                       repositories and generate one PDF with subgraphs for each repo plus
                       an overall project directory structure.
  --verbose            Enable verbose output (debug messages).

Examples:
  $ python3 git-graph.py
      Generate the Git object graph for the current repository.

  $ python3 git-graph.py --colors -o --local -P 1 -S 2 --dotted remote -M author,flags,short
      Generate a colored Git graph of the current repo with limited parent/child edges,
      forcing remote branch edges dotted, and including only author, flags and short hash.

  $ python3 git-graph.py --group --group-types blob,tree --group-condense-level 1
      Generate the Git graph for the current repo with grouping of blobs and trees.
      
  $ python3 git-graph.py --repo-base /path/to/project --colors --group --group-types blob,tree --verbose
      Recursively scan /path/to/project for Git repositories and produce one PDF that shows
      the overall project structure plus a subgraph for each repository with grouping enabled.

Requirements:
  - Python 3.5+
  - Graphviz (the “dot” command must be installed and available in PATH)
"""

import argparse
import collections
import colorsys
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from graphviz import Digraph
from typing import Dict, Set, List

# --- Git object types ---
Branch = collections.namedtuple('Branch', 'name commit remote')
Commit = collections.namedtuple('Commit', 'hash tree parents author')
Tree = collections.namedtuple('Tree', 'hash name trees blobs')
Blob = collections.namedtuple('Blob', 'hash name')
Hash = str

# --- Helper Functions ---

def sanitize_id(s: str) -> str:
    """Sanitize a string for use as a Graphviz node ID."""
    return re.sub(r'\W+', '_', s)

def make_node_id(prefix: str, typ: str, identifier: str) -> str:
    """Generate a namespaced node ID."""
    return f"{prefix}_{typ}_{sanitize_id(identifier)}"

def get_distinct_color(index: int, total: int) -> str:
    """
    Generate a distinct hex color using HLS.
    Returns a hex string (e.g. '#aabbcc').
    """
    if total <= 0:
        total = 1
    hue = (index / total) % 1.0
    # Use fixed lightness and saturation for visibility.
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def draw_grouped_nodes(subgraph: Digraph, nodes: List[dict], node_type: str,
                       prefix: str, config: dict) -> None:
    """
    Given a list of node dictionaries (each with keys: id, common, differentiator, full_label, attrs),
    group them by the common text. If a group contains only one element, draw it normally.
    Otherwise, draw a single record–shaped node that shows the common part on the left and
    the differentiators (one per variant) on the right. A unique group color is used.
    """
    groups = defaultdict(list)
    for node in nodes:
        groups[node["common"]].append(node)
    total_groups = len(groups)
    group_index = 0
    for common, group_nodes in groups.items():
        if len(group_nodes) == 1:
            n = group_nodes[0]
            subgraph.node(n['id'], label=n['full_label'], **n.get('attrs', {}))
        else:
            # Build a record label: {common | diff1 \l diff2 \l ...}
            diffs = "\\l".join([node["differentiator"] for node in group_nodes])
            record_label = f"{{{common}|{diffs}\\l}}"
            group_node_id = make_node_id(prefix, node_type, common)
            group_color = get_distinct_color(group_index, total_groups)
            # Override attributes for grouped node.
            attrs = {'shape': 'record', 'style': 'filled', 'color': group_color,
                     'fontsize': str(10 + 2 * (1 - config.get('group_condense_level', 1)))}
            subgraph.node(group_node_id, label=record_label, **attrs)
            group_index += 1

# --- Git Repository Data Model ---
class GitRepo:
    def __init__(self, git_repo_path: str, local_only: bool = False):
        self.git_repo_path = git_repo_path
        self.dot_git_dir = os.path.join(git_repo_path, '.git')
        self.local_only = local_only
        self.cache: Dict[Hash, object] = {}
        self.branches: List[Branch] = []
        self.branch_to_commit: Dict[str, Hash] = {}
        self.commit_to_branches: Dict[Hash, List[str]] = defaultdict(list)
        self.commit_to_parents: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.commit_to_tree: Dict[Hash, Hash] = {}
        self.tree_to_trees: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.tree_to_blobs: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.blobs: Dict[Hash, Blob] = {}
        self.commit_to_children: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.commit_gitstat: Dict[Hash, str] = {}

    def parse_dot_git_dir(self) -> None:
        logging.info("Parsing .git directory at '%s'", self.dot_git_dir)
        self.branches = self.list_branches()
        for branch in self.branches:
            self.branch_to_commit[branch.name] = branch.commit
            self.commit_to_branches[branch.commit].append(branch.name)
        visited: Set[Hash] = set()
        for branch in self.branches:
            try:
                self.traverse_history(branch.commit, visited)
            except Exception as e:
                logging.error("Error traversing history for branch %s: %s", branch.name, e)
        self.build_commit_children()

    def list_branches(self) -> List[Branch]:
        branches = []
        heads_dir = os.path.join(self.dot_git_dir, 'refs', 'heads')
        if os.path.isdir(heads_dir):
            try:
                for f in os.listdir(heads_dir):
                    full_path = os.path.join(heads_dir, f)
                    commit = self.read_txt(full_path)
                    branches.append(Branch(name=f, commit=commit, remote=False))
                    logging.debug("Found local branch '%s' -> %s", f, commit)
            except Exception as e:
                logging.error("Error listing local branches: %s", e)
                raise
        else:
            logging.warning("Local heads directory '%s' not found", heads_dir)
        if not self.local_only:
            remotes_dir = os.path.join(self.dot_git_dir, 'refs', 'remotes')
            if os.path.isdir(remotes_dir):
                try:
                    for remote in os.listdir(remotes_dir):
                        remote_path = os.path.join(remotes_dir, remote)
                        for root, _, files in os.walk(remote_path):
                            for f in files:
                                branch_name = remote + '/' + f
                                file_path = os.path.join(root, f)
                                commit = self.read_txt(file_path)
                                branches.append(Branch(name=branch_name, commit=commit, remote=True))
                                logging.debug("Found remote branch '%s' -> %s", branch_name, commit)
                except Exception as e:
                    logging.error("Error listing remote branches: %s", e)
            else:
                logging.info("No remote branches directory found at '%s'", remotes_dir)
        return branches

    def traverse_history(self, commit_hash: Hash, visited: Set[Hash]) -> None:
        if commit_hash in visited:
            return
        visited.add(commit_hash)
        try:
            commit_obj = self.get_commit(commit_hash)
        except Exception as e:
            logging.error("Failed to get commit %s: %s", commit_hash, e)
            return
        for parent_hash in commit_obj.parents:
            self.commit_to_parents[commit_hash].add(parent_hash)
            self.traverse_history(parent_hash, visited)

    def build_commit_children(self) -> None:
        for child, parents in self.commit_to_parents.items():
            for parent in parents:
                self.commit_to_children[parent].add(child)

    def get_commit(self, hash: Hash) -> Commit:
        if hash not in self.cache:
            try:
                content = self.git_cat_file(hash)
                commit_obj = self.parse_commit(hash, content)
            except Exception as e:
                logging.error("Error retrieving commit %s: %s", hash, e)
                raise
            self.cache[hash] = commit_obj
            self.commit_to_tree[commit_obj.hash] = commit_obj.tree
            try:
                self.get_tree(commit_obj.tree)
            except Exception as e:
                logging.error("Error retrieving tree %s for commit %s: %s", commit_obj.tree, hash, e)
                raise
        return self.cache[hash]

    def parse_commit(self, hash: Hash, content: str) -> Commit:
        commit_data = {'hash': hash, 'tree': None, 'parents': []}
        author_found = False
        for line in content.splitlines():
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'tree':
                commit_data['tree'] = parts[1]
            elif parts[0] == 'parent':
                commit_data['parents'].append(parts[1])
            elif parts[0] == 'author' and not author_found:
                commit_data['author'] = ' '.join(parts[1:])
                author_found = True
        if 'author' not in commit_data:
            commit_data['author'] = "Unknown"
        if not commit_data['tree']:
            raise ValueError(f"Commit {hash} missing tree pointer")
        return Commit(**commit_data)

    def get_tree(self, hash: Hash, name: str = '/') -> Tree:
        if hash not in self.cache:
            try:
                content = self.git_cat_file(hash)
                tree_obj = self.parse_tree(hash, name, content)
            except Exception as e:
                logging.error("Error retrieving tree %s: %s", hash, e)
                raise
            for child_hash in tree_obj.blobs:
                self.tree_to_blobs[hash].add(child_hash)
            for child_hash in tree_obj.trees:
                self.tree_to_trees[hash].add(child_hash)
            self.cache[hash] = tree_obj
        return self.cache[hash]

    def parse_tree(self, hash: Hash, name: str, content: str) -> Tree:
        trees_list = []
        blobs_list = []
        for line in content.splitlines():
            if not line:
                continue
            try:
                mode, obj_type, child_hash, child_name = line.split(None, 3)
            except ValueError:
                logging.warning("Skipping malformed tree entry in tree %s: '%s'", hash, line)
                continue
            if obj_type == 'tree':
                try:
                    self.get_tree(child_hash, child_name)
                except Exception as e:
                    logging.error("Error processing subtree %s: %s", child_hash, e)
                    continue
                trees_list.append(child_hash)
            elif obj_type == 'blob':
                blobs_list.append(child_hash)
                self.blobs[child_hash] = Blob(hash=child_hash, name=child_name)
        return Tree(hash=hash, name=name, trees=trees_list, blobs=blobs_list)

    def git_cat_file(self, hash: Hash) -> str:
        try:
            result = subprocess.run(
                ['git', 'cat-file', '-p', hash],
                cwd=self.git_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8').strip()
            logging.error("git cat-file failed for hash %s: %s", hash, err)
            raise Exception(f"Object {hash} not found.")
        except Exception as e:
            logging.error("Unexpected error in git_cat_file for hash %s: %s", hash, e)
            raise

    def read_txt(self, file_path: str) -> str:
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logging.error("Error reading file '%s': %s", file_path, e)
            raise

    def get_git_stat(self, commit_hash: Hash) -> str:
        if commit_hash in self.commit_gitstat:
            return self.commit_gitstat[commit_hash]
        try:
            result = subprocess.run(
                ['git', 'show', '--stat', '--oneline', '-s', commit_hash],
                cwd=self.git_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            stat = result.stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8').strip()
            logging.error("git show --stat failed for commit %s: %s", commit_hash, err)
            stat = ""
        except Exception as e:
            logging.error("Unexpected error in get_git_stat for commit %s: %s", commit_hash, e)
            stat = ""
        self.commit_gitstat[commit_hash] = stat
        return stat

# --- Formatting for commit labels ---
def format_commit_label(commit: Commit, git_repo: GitRepo, config: dict) -> str:
    label = commit.hash[:6]
    meta = config['metadata']
    if meta == "all" or ("author" in meta):
        label += "\n" + commit.author
    if meta == "all" or ("flags" in meta):
        flags = []
        if commit.hash in git_repo.commit_to_branches:
            for b in git_repo.commit_to_branches[commit.hash]:
                flags.append("R" if ('/' in b) else "L")
            flags = sorted(set(flags))
            label += "\n(" + "/".join(flags) + ")"
    if meta == "all" or ("gitstat" in meta):
        stat = git_repo.get_git_stat(commit.hash)
        if stat:
            label += "\n" + stat.strip()
    return label

# --- Graph Generation for a Single Repository (as a subgraph) ---
def add_git_repo_subgraph(master: Digraph, git_repo: GitRepo, config: dict,
                          prefix: str, repo_label: str) -> None:
    """
    Add a subgraph (cluster) for one repository’s Git object graph.
    All node IDs are prefixed with the given prefix.
    This routine now supports grouping of repetitive nodes.
    """
    drawn_edges = set()
    repo_cluster = master.subgraph(name=f"cluster_repo_{sanitize_id(prefix)}")
    repo_cluster.attr(label=repo_label, color='blue')
    
    # --- Blobs ---
    if config.get('group_enabled', False) and 'blob' in config.get('group_types', []):
        nodes_list = []
        for blob in git_repo.blobs.values():
            node_id = make_node_id(prefix, "blob", blob.hash)
            common = blob.name  # group by blob name
            differentiator = blob.hash[:6]
            full_label = f"{blob.name} {blob.hash[:6]}"
            attrs = {'shape': 'ellipse', 'style': 'filled'}
            if config['colors']:
                attrs['color'] = config['colors_map']['blob']
            nodes_list.append({"id": node_id, "common": common,
                               "differentiator": differentiator, "full_label": full_label, "attrs": attrs})
        # Create a dedicated subgraph for grouped blobs.
        group_blob_sg = repo_cluster.subgraph(name=f"cluster_{prefix}_grouped_blobs")
        group_blob_sg.attr(label="Blobs", color="gray")
        draw_grouped_nodes(group_blob_sg, nodes_list, "blob", prefix, config)
    else:
        with repo_cluster.subgraph(name=f"cluster_{prefix}_blobs") as blob_sg:
            blob_sg.attr(label='Blobs', color='gray')
            for blob in git_repo.blobs.values():
                nid = make_node_id(prefix, "blob", blob.hash)
                label = f"{blob.name} {blob.hash[:6]}"
                attrs = {'shape': 'ellipse', 'style': 'filled'}
                if config['colors']:
                    attrs['color'] = config['colors_map']['blob']
                blob_sg.node(nid, label=label, **attrs)
                
    # --- Trees ---
    if config.get('group_enabled', False) and 'tree' in config.get('group_types', []):
        nodes_list = []
        tree_keys = set(git_repo.tree_to_blobs.keys()).union(set(git_repo.tree_to_trees.keys()))
        for tree_hash in tree_keys:
            try:
                tree_obj = git_repo.get_tree(tree_hash)
            except Exception as e:
                logging.error("Skipping tree %s: %s", tree_hash, e)
                continue
            node_id = make_node_id(prefix, "tree", tree_hash)
            common = tree_obj.name  # group by tree name
            differentiator = tree_hash[:6]
            full_label = f"{tree_obj.name} {tree_obj.hash[:6]}"
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            nodes_list.append({"id": node_id, "common": common,
                               "differentiator": differentiator, "full_label": full_label, "attrs": attrs})
        group_tree_sg = repo_cluster.subgraph(name=f"cluster_{prefix}_grouped_trees")
        group_tree_sg.attr(label="Trees", color="gray")
        draw_grouped_nodes(group_tree_sg, nodes_list, "tree", prefix, config)
    else:
        with repo_cluster.subgraph(name=f"cluster_{prefix}_trees") as tree_sg:
            tree_sg.attr(label='Trees', color='gray')
            for tree_hash, blob_hashes in git_repo.tree_to_blobs.items():
                try:
                    tree_obj = git_repo.get_tree(tree_hash)
                except Exception as e:
                    logging.error("Skipping tree %s due to error: %s", tree_hash, e)
                    continue
                tree_node = f"{tree_obj.name} {tree_obj.hash[:6]}"
                attrs = {'shape': 'triangle'}
                if config['colors']:
                    attrs['color'] = config['colors_map']['tree']
                tree_sg.node(tree_node, **attrs)
                for blob_hash in blob_hashes:
                    try:
                        blob_node = f"{git_repo.blobs[blob_hash].name} {blob_hash[:6]}"
                    except KeyError:
                        logging.warning("Blob %s not found in cache", blob_hash)
                        continue
                    tree_sg.edge(tree_node, blob_node, minlen='2', constraint='true')
            for tree_hash, subtree_hashes in git_repo.tree_to_trees.items():
                try:
                    tree_obj = git_repo.get_tree(tree_hash)
                except Exception as e:
                    logging.error("Skipping tree %s: %s", tree_hash, e)
                    continue
                tree_node = f"{tree_obj.name} {tree_obj.hash[:6]}"
                attrs = {'shape': 'triangle'}
                if config['colors']:
                    attrs['color'] = config['colors_map']['tree']
                tree_sg.node(tree_node, **attrs)
                for subtree_hash in subtree_hashes:
                    try:
                        subtree_obj = git_repo.get_tree(subtree_hash)
                    except Exception as e:
                        logging.error("Skipping subtree %s: %s", subtree_hash, e)
                        continue
                    subtree_node = f"{subtree_obj.name} {subtree_obj.hash[:6]}"
                    tree_sg.edge(tree_node, subtree_node, minlen='2', constraint='true')
                    
    # --- Commits ---
    with repo_cluster.subgraph(name=f"cluster_{prefix}_commits") as commit_sg:
        commit_sg.attr(label='Commits', color='gray')
        for commit_hash, tree_hash in git_repo.commit_to_tree.items():
            try:
                commit_obj = git_repo.get_commit(commit_hash)
            except Exception as e:
                logging.error("Skipping commit %s: %s", commit_hash, e)
                continue
            commit_nid = make_node_id(prefix, "commit", commit_hash)
            label = format_commit_label(commit_obj, git_repo, config)
            attrs = {'shape': 'rectangle', 'style': 'filled', 'label': label}
            if config['colors']:
                attrs['color'] = config['colors_map']['commit']
            commit_sg.node(commit_nid, **attrs)
            try:
                tree_obj = git_repo.get_tree(tree_hash)
                tree_nid = make_node_id(prefix, "tree", tree_hash)
                edge_key = (commit_nid, tree_nid)
                if edge_key not in drawn_edges:
                    commit_sg.edge(commit_nid, tree_nid, minlen='2')
                    drawn_edges.add(edge_key)
            except Exception as e:
                logging.error("Error linking commit %s to tree %s: %s", commit_hash, tree_hash, e)
        for commit_hash in git_repo.commit_to_tree.keys():
            try:
                commit_obj = git_repo.get_commit(commit_hash)
            except Exception:
                continue
            if not commit_obj.parents:
                continue
            commit_nid = make_node_id(prefix, "commit", commit_hash)
            limit = config['predecessor_limit']
            normal_preds = commit_obj.parents if (limit is None) else commit_obj.parents[:limit]
            extra_preds = [] if (limit is None) else commit_obj.parents[limit:]
            for parent in normal_preds:
                parent_nid = make_node_id(prefix, "commit", parent)
                edge_key = (commit_nid, parent_nid)
                if edge_key not in drawn_edges:
                    commit_sg.edge(commit_nid, parent_nid, minlen='2')
                    drawn_edges.add(edge_key)
            for parent in extra_preds:
                parent_nid = make_node_id(prefix, "commit", parent)
                edge_key = (commit_nid, parent_nid)
                if edge_key not in drawn_edges:
                    commit_sg.edge(commit_nid, parent_nid, style='dotted', minlen='2')
                    drawn_edges.add(edge_key)
        for commit_hash, children in git_repo.commit_to_children.items():
            commit_nid = make_node_id(prefix, "commit", commit_hash)
            children_list = list(children)
            limit = config['successor_limit']
            normal_children = children_list if (limit is None) else children_list[:limit]
            extra_children = [] if (limit is None) else children_list[limit:]
            for child in normal_children:
                child_nid = make_node_id(prefix, "commit", child)
                edge_key = (commit_nid, child_nid)
                if edge_key not in drawn_edges:
                    commit_sg.edge(commit_nid, child_nid, minlen='2')
                    drawn_edges.add(edge_key)
            for child in extra_children:
                child_nid = make_node_id(prefix, "commit", child)
                edge_key = (commit_nid, child_nid)
                if edge_key not in drawn_edges:
                    commit_sg.edge(commit_nid, child_nid, style='dotted', minlen='2')
                    drawn_edges.add(edge_key)
                    
    # --- Branches ---
    with repo_cluster.subgraph(name=f"cluster_{prefix}_branches") as branch_sg:
        branch_sg.attr(label='Branches', color='gray')
        for branch_name, commit_hash in git_repo.branch_to_commit.items():
            branch_nid = make_node_id(prefix, "branch", branch_name)
            attrs = {'shape': 'parallelogram'}
            if config['colors']:
                attrs['color'] = config['colors_map']['branch']
            branch_sg.node(branch_nid, label=branch_name, **attrs)
            style = 'solid'
            if (( '/' in branch_name and ('remote' in config['dotted'] or 'all' in config['dotted'])) or
                ( '/' not in branch_name and ('local' in config['dotted'] or 'all' in config['dotted']))):
                style = 'dotted'
            commit_nid = make_node_id(prefix, "commit", commit_hash)
            branch_sg.edge(branch_nid, commit_nid, style=style, minlen='2')

    if config['colors']:
        with repo_cluster.subgraph(name=f"cluster_{prefix}_legend") as legend:
            legend.attr(label='Legend', color='black')
            for nodetype, col in config['colors_map'].items():
                node_id = f"legend_{nodetype}"
                legend.node(node_id, label=nodetype.capitalize(), shape='box', style='filled', color=col)

# --- Combined Repository Base Mode ---
def generate_combined_repo_graph(repo_base_path: str, config: dict) -> None:
    """
    Recursively scan repo_base_path for Git repositories (directories with a .git folder)
    and generate one PDF document containing:
      1. A cluster for the overall project directory structure.
      2. A cluster for all detected Git repositories (each added as a subgraph).
    """
    logging.info("Scanning for Git repositories under '%s'", repo_base_path)
    git_repos = []
    for root, dirs, _ in os.walk(repo_base_path):
        if '.git' in dirs:
            repo_abs = os.path.join(root)
            rel_path = os.path.relpath(repo_abs, repo_base_path)
            git_repos.append((repo_abs, rel_path))
            dirs[:] = []  # do not descend further into repos
    if not git_repos:
        logging.error("No Git repositories found under '%s'", repo_base_path)
        sys.exit(1)
    logging.info("Found %d Git repositories.", len(git_repos))

    master = Digraph(comment='Combined Repository Graph', format='pdf')
    master.attr(compound='true', splines='true', overlap='false')

    # Project structure cluster.
    proj_cluster = master.subgraph(name="cluster_project_structure")
    proj_cluster.attr(label="Project Structure", color="black")
    node_counter = [0]
    def process_directory(dir_path: str, parent_id: str = None):
        try:
            entries = os.listdir(dir_path)
        except Exception as e:
            logging.error("Cannot list directory '%s': %s", dir_path, e)
            return
        current_id = sanitize_id(os.path.abspath(dir_path))
        label = os.path.basename(dir_path) if os.path.basename(dir_path) else dir_path
        proj_cluster.node(current_id, label=label, shape='folder', style='filled', color='lightblue')
        if parent_id:
            proj_cluster.edge(parent_id, current_id)
        for entry in sorted(entries):
            full_path = os.path.join(dir_path, entry)
            if os.path.isdir(full_path):
                process_directory(full_path, current_id)
            else:
                file_id = sanitize_id(os.path.abspath(full_path)) + f"_{node_counter[0]}"
                node_counter[0] += 1
                proj_cluster.node(file_id, label=entry, shape='note', style='filled', color='lightyellow')
                proj_cluster.edge(current_id, file_id)
    process_directory(repo_base_path)

    # Git repositories cluster.
    repos_cluster = master.subgraph(name="cluster_git_repos")
    repos_cluster.attr(label="Git Repositories", color="blue")
    for idx, (repo_abs, rel_path) in enumerate(git_repos):
        prefix = f"repo{idx}_{sanitize_id(rel_path)}"
        logging.info("Processing repository at '%s' (relative: '%s')", repo_abs, rel_path)
        try:
            repo_obj = GitRepo(repo_abs, local_only=config['local_only'])
            repo_obj.parse_dot_git_dir()
        except Exception as e:
            logging.error("Error processing repository '%s': %s", repo_abs, e)
            continue
        add_git_repo_subgraph(repos_cluster, repo_obj, config, prefix, repo_label=rel_path)

    try:
        logging.info("Rendering combined graph to file 'combined_repo.gv.pdf'")
        logging.debug("Graph source:\n%s", master.source)
        master.render('combined_repo.gv', view=(not config['output_only']))
    except Exception as e:
        logging.error("Error rendering combined graph: %s", e)
        sys.exit(1)

# --- Graph Generation for Single Repository Mode ---
class GraphGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_graph(self, git_repo: GitRepo) -> None:
        logging.info("Generating Git object graph (single repository mode)")
        # Instead of building the graph here from scratch, call add_git_repo_subgraph
        master = Digraph(comment='Git graph', format='pdf')
        master.attr(compound='true', splines='true', overlap='false')
        add_git_repo_subgraph(master, git_repo, self.config, prefix="single", repo_label="Repository")
        try:
            logging.info("Rendering graph to file 'git.gv.pdf'")
            logging.debug("Graph source:\n%s", master.source)
            master.render('git.gv', view=(not self.config['output_only']))
        except Exception as e:
            logging.error("Error rendering graph: %s", e)
            sys.exit(1)

# --- Utility Functions ---
def check_dependencies() -> None:
    if not shutil.which('dot'):
        logging.error('Command "dot" was not found. Please install Graphviz.')
        sys.exit(1)

def get_git_repo_path(path: str) -> str:
    if not os.path.isdir(path):
        logging.error("Invalid git repo path: '%s'", path)
        sys.exit(1)
    dot_git_dir = os.path.join(path, '.git')
    if not os.path.isdir(dot_git_dir):
        logging.error("No .git directory found in '%s'", path)
        sys.exit(1)
    return path

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description='Generate a Graphviz diagram of a Git repository or a combined view of multiple repos.\n\n'
                    'Examples:\n'
                    '  python3 git-graph.py --colors -o --local -P 1 -S 2 --dotted remote -M author,flags,short\n'
                    '  python3 git-graph.py --repo-base /path/to/project --colors --group --group-types blob,tree --verbose',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('path', nargs='?', default=os.getcwd(),
                        help='Path to the Git repository (default: current directory)')
    parser.add_argument('--colors', action='store_true',
                        help='Enable unique colors for nodes and add a legend')
    parser.add_argument('-o', '--output-only', action='store_true',
                        help='Save the output to file only without opening the viewer')
    parser.add_argument('--local', action='store_true',
                        help='Display only local branches (skip remote branches)')
    parser.add_argument('-P', '--predecessors', type=int, default=None,
                        help='Maximum number of predecessor (parent) edges to show normally per commit\n'
                             '(extra edges will be drawn dotted)')
    parser.add_argument('-S', '--successors', type=int, default=None,
                        help='Maximum number of successor (child) edges to show normally per commit\n'
                             '(extra edges will be drawn dotted)')
    parser.add_argument('--dotted', nargs='*', choices=['remote', 'local', 'all'], default=[],
                        help='Force the given group(s) of branch edges to be drawn in dotted style.\n'
                             'Choices: remote, local, all')
    parser.add_argument('-M', '--metadata', type=str, default='all',
                        help='Comma-separated list of metadata keys to include in commit node labels.\n'
                             'Available keys: flags, author, short, gitstat. Default: all')
    parser.add_argument('--group', action='store_true',
                        help='Enable multi-level grouping of nodes with similar labels')
    parser.add_argument('--group-types', type=str, default='blob,tree',
                        help='Comma-separated list of node types to group (options: blob, tree, commit, branch). Default: blob,tree')
    parser.add_argument('--group-condense-level', type=int, default=1,
                        help='Level of visual condensation for grouped nodes (higher means more condensed; default: 1)')
    parser.add_argument('--repo-base', type=str,
                        help='Path to the repository base folder to generate a combined diagram of all Git repos found')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    args = parser.parse_args()

    meta = args.metadata.strip()
    if meta != "all":
        meta = {m.strip() for m in meta.split(',')}
    config = {
        'colors': args.colors,
        'output_only': args.output_only,
        'predecessor_limit': args.predecessors,
        'successor_limit': args.successors,
        'dotted': args.dotted,
        'metadata': meta,
        'local_only': args.local,
        'colors_map': {
            'blob': 'lightyellow',
            'tree': 'lightgreen',
            'commit': 'lightblue',
            'branch': 'orange'
        },
        'group_enabled': args.group,
        'group_types': [x.strip().lower() for x in args.group_types.split(',')] if args.group_types else [],
        'group_condense_level': args.group_condense_level
    }
    return {'repo_path': args.path, 'config': config, 'repo_base': args.repo_base, 'verbose': args.verbose}

def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args['verbose'] else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    check_dependencies()
    try:
        if args['repo_base']:
            if not os.path.isdir(args['repo_base']):
                logging.error("Invalid repository base path: '%s'", args['repo_base'])
                sys.exit(1)
            generate_combined_repo_graph(args['repo_base'], args['config'])
        else:
            repo_path = get_git_repo_path(args['repo_path'])
            git_repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            git_repo.parse_dot_git_dir()
            GraphGenerator(args['config']).generate_graph(git_repo)
    except Exception as e:
        logging.exception("An unrecoverable error occurred: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
