#!/usr/bin/env python3
"""
Git Graph Visualizer with Multi–Level Visual Grouping and Dump/Read–Dump Functionality

This script operates in several modes:

1. Git Object Graph Mode (default):
   Parses a Git repository (its .git folder) and generates a Graphviz diagram
   of its commits, trees, blobs, and branches.

2. Combined Repository Base Mode (--repo-base):
   Recursively scans a base folder for Git repositories (directories containing
   a .git folder) and produces one PDF that shows the overall project directory
   structure plus subgraphs for each repository’s Git object graph.

3. Dump Mode (--dump):
   Instead of generating a graph, this mode produces a JSON dump of the internal
   Git repository model (branches, commits, trees, blobs, and relationships).

4. Read Dump Mode (--read-dump):
   Reads a previously generated dump file and produces a graph based on the data.

Other options include node grouping, colorization, metadata filtering, edge‐limits,
and safety limits when scanning file trees.

Requirements:
  - Python 3.5+
  - Graphviz (the “dot” command must be installed and available in PATH)
"""

import argparse
import collections
import colorsys
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from graphviz import Digraph
from types import SimpleNamespace
from typing import Dict, Set, List, Optional

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
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def draw_grouped_nodes(subgraph: Digraph, nodes: List[dict], node_type: str,
                       prefix: str, config: dict) -> None:
    """
    Given a list of node dictionaries (each with keys: id, common, differentiator,
    full_label, attrs), group them by the 'common' text. Single–element groups are drawn
    normally; groups with multiple elements are drawn as a record–shaped node.
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
            diffs = "\\l".join(node["differentiator"] for node in group_nodes)
            record_label = f"{{{common}|{diffs}\\l}}"
            group_node_id = make_node_id(prefix, node_type, common)
            group_color = get_distinct_color(group_index, total_groups)
            attrs = {
                'shape': 'record',
                'style': 'filled',
                'color': group_color,
                'fontsize': str(10 + 2 * (1 - config.get('group_condense_level', 1)))
            }
            subgraph.node(group_node_id, label=record_label, **attrs)
            group_index += 1

def create_subgraph(parent: Digraph, name: str, label: str = None, color: str = None) -> Digraph:
    """
    Helper function to create a subgraph, then set its attributes.
    This version handles the Graphviz subgraph context manager by calling __enter__() immediately.
    """
    sub = parent.subgraph(name=name)
    if hasattr(sub, '__enter__'):
        sub = sub.__enter__()
    if label:
        sub.attr(label=label)
    if color:
        sub.attr(color=color)
    return sub

# --- Git Repository Data Model ---
class GitRepo:
    """
    Represents a parsed Git repository, with commits, trees, blobs, and branches.
    """
    def __init__(self, git_repo_path: str, local_only: bool = False):
        self.git_repo_path = os.path.abspath(git_repo_path)
        self.dot_git_dir = os.path.join(self.git_repo_path, '.git')
        self.local_only = local_only
        
        # Caches and indices
        self.cache: Dict[Hash, object] = {}  # commits or trees
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
        """
        Parses the Git repo data: references, commits, and trees.
        """
        logging.info("Parsing .git directory at '%s'", self.dot_git_dir)

        # Gather references from HEAD, loose refs, and packed-refs
        self._gather_references()

        # Convert references to Branch objects.
        # (We treat any reference in refs/remotes/* as remote.)
        self.branches = []
        for ref_name, commit_hash in self._all_references.items():
            if not commit_hash:
                continue
            # Distinguish remote vs local by naming convention
            is_remote = ref_name.startswith("refs/remotes/")
            # Also store a short branch name for display
            if is_remote:
                short_name = ref_name.replace("refs/remotes/", "", 1)
            else:
                short_name = ref_name.replace("refs/heads/", "", 1)
            # If local_only, skip remote references
            if self.local_only and is_remote:
                continue
            self.branches.append(Branch(name=short_name, commit=commit_hash, remote=is_remote))
            self.branch_to_commit[short_name] = commit_hash
            self.commit_to_branches[commit_hash].append(short_name)

        visited: Set[Hash] = set()
        for branch in self.branches:
            try:
                self.traverse_history(branch.commit, visited)
            except Exception as e:
                logging.error("Error traversing history for branch %s: %s", branch.name, e)

        self.build_commit_children()

    def list_branches(self) -> List[Branch]:
        """
        (Deprecated old function to keep the code structure unchanged.)
        Now this returns the final list of branches found by parse_dot_git_dir().
        """
        return self.branches

    def traverse_history(self, commit_hash: Hash, visited: Set[Hash]) -> None:
        """
        Recursively traverse commit parents, building up commit_to_parents
        and storing each commit in self.cache.
        """
        if not commit_hash or commit_hash in visited:
            return
        visited.add(commit_hash)
        commit_obj: Optional[Commit] = None
        try:
            commit_obj = self.get_commit(commit_hash)
        except Exception as e:
            # Possibly a shallow or missing object; log and skip.
            logging.warning("Skipping missing commit %s: %s", commit_hash, e)

        if not commit_obj:
            return

        for parent_hash in commit_obj.parents:
            self.commit_to_parents[commit_hash].add(parent_hash)
            self.traverse_history(parent_hash, visited)

    def build_commit_children(self) -> None:
        """
        Invert commit_to_parents to populate commit_to_children for graph edges.
        """
        for child, parents in self.commit_to_parents.items():
            for parent in parents:
                self.commit_to_children[parent].add(child)

    def get_commit(self, hash: Hash) -> Commit:
        """
        Return a commit object from cache or by reading it via git cat-file.
        """
        if hash in self.cache and isinstance(self.cache[hash], Commit):
            return self.cache[hash]  # type: ignore

        content = self.git_cat_file(hash)
        commit_obj = self.parse_commit(hash, content)
        self.cache[hash] = commit_obj
        self.commit_to_tree[commit_obj.hash] = commit_obj.tree

        # Also load the tree to populate tree info
        try:
            self.get_tree(commit_obj.tree)
        except Exception as e:
            logging.warning("Error retrieving tree %s for commit %s: %s", commit_obj.tree, hash, e)

        return commit_obj

    def parse_commit(self, hash: Hash, content: str) -> Commit:
        """
        Parse the raw text of a commit object into a Commit namedtuple.
        """
        commit_data = {'hash': hash, 'tree': None, 'parents': [], 'author': "Unknown"}
        author_found = False
        for line in content.splitlines():
            line = line.rstrip("\n")
            if not line.strip():
                # Commit message or blank line
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == 'tree' and len(parts) >= 2:
                commit_data['tree'] = parts[1]
            elif key == 'parent' and len(parts) >= 2:
                commit_data['parents'].append(parts[1])
            elif key == 'author' and not author_found:
                commit_data['author'] = ' '.join(parts[1:]) if len(parts) > 1 else "Unknown"
                author_found = True

        if not commit_data['tree']:
            raise ValueError(f"Commit {hash} missing tree pointer.")
        return Commit(**commit_data)  # type: ignore

    def get_tree(self, hash: Hash, name: str = '/') -> Tree:
        """
        Return a Tree object from cache or by reading it via git cat-file.
        """
        if hash in self.cache and isinstance(self.cache[hash], Tree):
            return self.cache[hash]  # type: ignore

        content = self.git_cat_file(hash)
        tree_obj = self.parse_tree(hash, name, content)

        # Update internal mappings
        for b_hash in tree_obj.blobs:
            self.tree_to_blobs[hash].add(b_hash)
        for t_hash in tree_obj.trees:
            self.tree_to_trees[hash].add(t_hash)

        self.cache[hash] = tree_obj
        return tree_obj

    def parse_tree(self, hash: Hash, name: str, content: str) -> Tree:
        """
        Parse the raw text of a tree object into a Tree namedtuple.
        """
        trees_list = []
        blobs_list = []

        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Typical line format: "<mode> <type> <hash>\t<filename>"
            # git cat-file -p tree can also show e.g.: "100644 blob f3bfc etc"
            parts = re.split(r'\s+', line, maxsplit=3)
            if len(parts) < 4:
                logging.debug("Skipping malformed tree entry in %s: %r", hash, line)
                continue

            _, obj_type, child_hash, child_name = parts
            if obj_type == 'tree':
                trees_list.append(child_hash)
                # Recursively parse subtree (on-demand)
                try:
                    self.get_tree(child_hash, child_name)
                except Exception as e:
                    logging.warning("Error processing subtree %s: %s", child_hash, e)
                    continue
            elif obj_type == 'blob':
                blobs_list.append(child_hash)
                self.blobs[child_hash] = Blob(hash=child_hash, name=child_name)

        return Tree(hash=hash, name=name, trees=trees_list, blobs=blobs_list)

    def git_cat_file(self, hash: Hash) -> str:
        """
        Runs 'git cat-file -p <hash>' and returns the stdout as a string.
        """
        try:
            result = subprocess.run(
                ['git', 'cat-file', '-p', hash],
                cwd=self.git_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return result.stdout.decode('utf-8', errors='replace')
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8', errors='replace').strip()
            logging.warning("git cat-file failed for hash %s: %s", hash, err)
            raise Exception(f"Object {hash} not found.")
        except Exception as e:
            logging.warning("Unexpected error in git_cat_file for hash %s: %s", hash, e)
            raise

    def read_txt(self, file_path: str) -> str:
        """
        Return the stripped text content of a file.
        """
        try:
            with open(file_path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logging.error("Error reading file '%s': %s", file_path, e)
            return ""

    def get_git_stat(self, commit_hash: Hash) -> str:
        """
        Runs 'git show --stat --oneline -s <commit>' to get the short stats,
        caches them, and returns them.
        """
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
            stat = result.stdout.decode('utf-8', errors='replace')
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode('utf-8', errors='replace').strip()
            logging.warning("git show --stat failed for commit %s: %s", commit_hash, err)
            stat = ""
        except Exception as e:
            logging.warning("Unexpected error in get_git_stat for commit %s: %s", commit_hash, e)
            stat = ""
        self.commit_gitstat[commit_hash] = stat
        return stat

    # ----------------------------------------------------------
    #      Reference Gathering from HEAD, Refs, and Packed-Refs
    # ----------------------------------------------------------

    def _gather_references(self) -> None:
        """
        Internal method to gather references from HEAD, loose refs in .git/refs,
        and .git/packed-refs. Populates self._all_references with refname->commit.
        """
        self._all_references: Dict[str, str] = {}

        # 1) HEAD (could be a direct commit or symbolic ref)
        head_ref = os.path.join(self.dot_git_dir, 'HEAD')
        if os.path.isfile(head_ref):
            head_content = self.read_txt(head_ref)
            # Usually: "ref: refs/heads/main" or just a commit hash
            if head_content.startswith("ref: "):
                ref_path = head_content[5:].strip()
                commit_hash = self._try_resolve_ref(ref_path)
                if commit_hash:
                    self._all_references[ref_path] = commit_hash
            else:
                # HEAD might be a direct hash (detached)
                # Validate that it looks like a commit
                if re.match(r'^[0-9a-fA-F]{4,40}$', head_content):
                    # We'll store it as a "HEAD" ref
                    self._all_references["HEAD"] = head_content

        # 2) Loose references in .git/refs/* 
        refs_dir = os.path.join(self.dot_git_dir, 'refs')
        if os.path.isdir(refs_dir):
            for root, _, files in os.walk(refs_dir):
                for f in files:
                    full_path = os.path.join(root, f)
                    # The "canonical" ref name would be root[len(.git/)+1..] + "/" + f
                    # E.g.: .git/refs/heads/master => "refs/heads/master"
                    ref_name = os.path.relpath(full_path, self.dot_git_dir).replace("\\", "/")
                    commit_hash = self.read_txt(full_path)
                    # Validate commit
                    if re.match(r'^[0-9a-fA-F]{4,40}$', commit_hash):
                        self._all_references[ref_name] = commit_hash
                    else:
                        logging.debug("Skipping non-commit ref %s => %s", ref_name, commit_hash)

        # 3) Packed-refs in .git/packed-refs
        packed_refs = os.path.join(self.dot_git_dir, 'packed-refs')
        if os.path.isfile(packed_refs):
            try:
                with open(packed_refs, 'r') as pf:
                    for line in pf:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith('^'):
                            # ^ means it's the object hash of an annotated tag,
                            # we skip because we only want the commit references.
                            # If you want tags, handle them here.
                            continue
                        # Typically: "<hash> <refname>"
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            ref_hash, ref_name = parts
                            if re.match(r'^[0-9a-fA-F]{4,40}$', ref_hash):
                                # We only gather heads/remotes if local_only=False. 
                                # But let's store them all; filter later in parse_dot_git_dir.
                                self._all_references[ref_name] = ref_hash
            except Exception as e:
                logging.warning("Error reading packed-refs: %s", e)

    def _try_resolve_ref(self, ref_name: str) -> Optional[str]:
        """
        Attempt to read the reference by its file path under .git, or from packed-refs.
        Return a commit hash or None.
        """
        # 1) Check if there's a loose ref file
        ref_path = os.path.join(self.dot_git_dir, ref_name)
        if os.path.isfile(ref_path):
            commit_hash = self.read_txt(ref_path)
            if re.match(r'^[0-9a-fA-F]{4,40}$', commit_hash):
                return commit_hash

        # 2) Check packed-refs
        packed_refs = os.path.join(self.dot_git_dir, 'packed-refs')
        if os.path.isfile(packed_refs):
            try:
                with open(packed_refs, 'r') as pf:
                    for line in pf:
                        line = line.strip()
                        if line.startswith('#') or not line:
                            continue
                        if line.startswith('^'):
                            continue
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            ref_hash, name = parts
                            if name == ref_name and re.match(r'^[0-9a-fA-F]{4,40}$', ref_hash):
                                return ref_hash
            except Exception as e:
                logging.warning("Error reading packed-refs: %s", e)

        return None

# --- Dump/Load Support for Repository Data ---
def dump_git_repo(git_repo: GitRepo, dump_file: str) -> None:
    """Serialize the GitRepo’s data into a JSON dump file."""
    data = {
        "branches": [branch._asdict() for branch in git_repo.branches],
        "commits": {
            h: commit._asdict() for h, commit in git_repo.cache.items()
            if isinstance(commit, Commit)
        },
        "trees": {
            h: tree._asdict() for h, tree in git_repo.cache.items()
            if isinstance(tree, Tree)
        },
        "blobs": {
            h: blob._asdict() for h, blob in git_repo.blobs.items()
        },
        "commit_to_tree": git_repo.commit_to_tree,
        "commit_to_branches": git_repo.commit_to_branches,
        "branch_to_commit": git_repo.branch_to_commit,
        "commit_to_parents": {k: list(v) for k, v in git_repo.commit_to_parents.items()},
        "commit_to_children": {k: list(v) for k, v in git_repo.commit_to_children.items()},
        "tree_to_blobs": {k: list(v) for k, v in git_repo.tree_to_blobs.items()},
        "tree_to_trees": {k: list(v) for k, v in git_repo.tree_to_trees.items()},
        "commit_gitstat": git_repo.commit_gitstat,
    }
    try:
        with open(dump_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info("Dumped repository data to '%s'", dump_file)
    except Exception as e:
        logging.error("Failed to write dump file '%s': %s", dump_file, e)
        sys.exit(1)

class DumpedRepo:
    """
    A lightweight repository–like class constructed from a dump.
    Provides the attributes and methods needed by the graph–generation routines.
    """
    def __init__(self, data: dict):
        self.branches = [SimpleNamespace(**b) for b in data.get("branches", [])]
        self.commits = data.get("commits", {})
        self.trees = data.get("trees", {})
        self.blobs = {
            h: SimpleNamespace(**b) for h, b in data.get("blobs", {}).items()
        }
        self.commit_to_tree = data.get("commit_to_tree", {})
        self.commit_to_branches = data.get("commit_to_branches", {})
        self.branch_to_commit = data.get("branch_to_commit", {})
        self.commit_to_children = data.get("commit_to_children", {})
        self.tree_to_blobs = data.get("tree_to_blobs", {})
        self.tree_to_trees = data.get("tree_to_trees", {})
        self.commit_gitstat = data.get("commit_gitstat", {})

    def get_tree(self, tree_hash: str) -> SimpleNamespace:
        tree_data = self.trees.get(tree_hash)
        if tree_data is None:
            raise Exception(f"Tree {tree_hash} not found in dump.")
        return SimpleNamespace(**tree_data)

    def get_commit(self, commit_hash: str) -> SimpleNamespace:
        commit_data = self.commits.get(commit_hash)
        if commit_data is None:
            raise Exception(f"Commit {commit_hash} not found in dump.")
        return SimpleNamespace(**commit_data)

def load_git_repo_dump(dump_file: str) -> DumpedRepo:
    """Load the JSON dump file and return a DumpedRepo instance."""
    try:
        with open(dump_file, 'r') as f:
            data = json.load(f)
        logging.info("Loaded dump from '%s'", dump_file)
        return DumpedRepo(data)
    except Exception as e:
        logging.error("Failed to load dump file '%s': %s", dump_file, e)
        sys.exit(1)

# --- Formatting for commit labels ---
def format_commit_label(commit, git_repo, config: dict) -> str:
    """
    Build a text label for a commit, given the user-chosen metadata in config.
    """
    label = commit.hash[:6]
    meta = config['metadata']
    # If meta is a set, check membership. If it's 'all', show everything.
    def want(key: str) -> bool:
        return meta == 'all' or (isinstance(meta, set) and key in meta)

    # Show author
    if want("author"):
        label += "\n" + commit.author

    # Show local/remote flags
    if want("flags"):
        flags = []
        if commit.hash in git_repo.commit_to_branches:
            for b in git_repo.commit_to_branches[commit.hash]:
                # If there's a slash, call it remote
                flags.append("R" if '/' in b else "L")
            flags = sorted(set(flags))
            label += "\n(" + "/".join(flags) + ")"

    # Show short stats
    if want("gitstat"):
        stat = git_repo.get_git_stat(commit.hash)
        if stat:
            label += "\n" + stat.strip()

    return label

# --- Graph Generation for a Single Repository (as a subgraph) ---
def add_git_repo_subgraph(master: Digraph, git_repo, config: dict,
                          prefix: str, repo_label: str) -> None:
    """
    Add a subgraph (cluster) for one repository’s Git object graph.
    All node IDs are prefixed with the given prefix.
    Supports grouping of repetitive nodes.
    """
    drawn_edges = set()
    repo_cluster = create_subgraph(
        master, f"cluster_repo_{sanitize_id(prefix)}",
        label=repo_label, color='blue'
    )
    
    # --- Blobs ---
    if config.get('group_enabled', False) and 'blob' in config.get('group_types', []):
        nodes_list = []
        for blob in git_repo.blobs.values():
            blob_name = getattr(blob, 'name', str(blob))
            blob_hash = getattr(blob, 'hash', '')
            node_id = make_node_id(prefix, "blob", blob_hash)
            common = blob_name  # group by blob name
            differentiator = blob_hash[:6]
            full_label = f"{blob_name} {blob_hash[:6]}"
            attrs = {'shape': 'ellipse', 'style': 'filled'}
            if config['colors']:
                attrs['color'] = config['colors_map']['blob']
            nodes_list.append({
                "id": node_id, "common": common,
                "differentiator": differentiator,
                "full_label": full_label, "attrs": attrs
            })
        group_blob_sg = create_subgraph(
            repo_cluster, f"cluster_{prefix}_grouped_blobs",
            label="Blobs", color="gray"
        )
        draw_grouped_nodes(group_blob_sg, nodes_list, "blob", prefix, config)
    else:
        blob_sg = create_subgraph(
            repo_cluster, f"cluster_{prefix}_blobs",
            label="Blobs", color="gray"
        )
        for blob in git_repo.blobs.values():
            blob_name = getattr(blob, 'name', str(blob))
            blob_hash = getattr(blob, 'hash', '')
            nid = make_node_id(prefix, "blob", blob_hash)
            label = f"{blob_name} {blob_hash[:6]}"
            attrs = {'shape': 'ellipse', 'style': 'filled'}
            if config['colors']:
                attrs['color'] = config['colors_map']['blob']
            blob_sg.node(nid, label=label, **attrs)
                
    # --- Trees ---
    if config.get('group_enabled', False) and 'tree' in config.get('group_types', []):
        nodes_list = []
        tree_keys = set(git_repo.tree_to_blobs.keys()).union(
            set(git_repo.tree_to_trees.keys())
        )
        for tree_hash in tree_keys:
            try:
                tree_obj = git_repo.get_tree(tree_hash)
            except Exception as e:
                logging.error("Skipping tree %s: %s", tree_hash, e)
                continue
            common = tree_obj.name
            differentiator = tree_obj.hash[:6]
            full_label = f"{tree_obj.name} {tree_obj.hash[:6]}"
            node_id = make_node_id(prefix, "tree", common)
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            nodes_list.append({
                "id": node_id, "common": common,
                "differentiator": differentiator,
                "full_label": full_label, "attrs": attrs
            })
        group_tree_sg = create_subgraph(
            repo_cluster, f"cluster_{prefix}_grouped_trees",
            label="Trees", color="gray"
        )
        draw_grouped_nodes(group_tree_sg, nodes_list, "tree", prefix, config)
    else:
        tree_sg = create_subgraph(
            repo_cluster, f"cluster_{prefix}_trees",
            label="Trees", color="gray"
        )
        # For each tree, draw its node and connect to blobs/subtrees
        for tree_hash, blob_hashes in git_repo.tree_to_blobs.items():
            try:
                tree_obj = git_repo.get_tree(tree_hash)
            except Exception as e:
                logging.error("Skipping tree %s due to error: %s", tree_hash, e)
                continue
            tree_nid = (make_node_id(prefix, "tree", tree_obj.name) if
                        (config.get('group_enabled', False) and
                         'tree' in config.get('group_types', []))
                        else make_node_id(prefix, "tree", tree_hash))
            label = f"{tree_obj.name} {tree_obj.hash[:6]}"
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            tree_sg.node(tree_nid, label=label, **attrs)

            for blob_hash in blob_hashes:
                try:
                    blob = git_repo.blobs[blob_hash]
                except KeyError:
                    logging.warning("Blob %s not found in cache", blob_hash)
                    continue
                blob_nid = make_node_id(prefix, "blob", blob_hash)
                tree_sg.edge(tree_nid, blob_nid, minlen='2', constraint='true')

        for tree_hash, subtree_hashes in git_repo.tree_to_trees.items():
            try:
                tree_obj = git_repo.get_tree(tree_hash)
            except Exception as e:
                logging.error("Skipping tree %s: %s", tree_hash, e)
                continue
            tree_nid = (make_node_id(prefix, "tree", tree_obj.name) if
                        (config.get('group_enabled', False) and
                         'tree' in config.get('group_types', []))
                        else make_node_id(prefix, "tree", tree_hash))
            label = f"{tree_obj.name} {tree_obj.hash[:6]}"
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            tree_sg.node(tree_nid, label=label, **attrs)

            for subtree_hash in subtree_hashes:
                try:
                    subtree_obj = git_repo.get_tree(subtree_hash)
                except Exception as e:
                    logging.error("Skipping subtree %s: %s", subtree_hash, e)
                    continue
                subtree_nid = (make_node_id(prefix, "tree", subtree_obj.name) if
                               (config.get('group_enabled', False) and
                                'tree' in config.get('group_types', []))
                               else make_node_id(prefix, "tree", subtree_hash))
                subtree_label = f"{subtree_obj.name} {subtree_obj.hash[:6]}"
                tree_sg.node(subtree_nid, label=subtree_label, **attrs)
                tree_sg.edge(tree_nid, subtree_nid, minlen='2', constraint='true')
                    
    # --- Commits ---
    commit_sg = create_subgraph(
        repo_cluster, f"cluster_{prefix}_commits",
        label="Commits", color="gray"
    )
    for commit_hash, tree_hash in git_repo.commit_to_tree.items():
        try:
            commit_obj = git_repo.get_commit(commit_hash)
        except Exception as e:
            logging.warning("Skipping commit %s: %s", commit_hash, e)
            continue
        commit_nid = make_node_id(prefix, "commit", commit_hash)
        label = format_commit_label(commit_obj, git_repo, config)
        attrs = {'shape': 'rectangle', 'style': 'filled', 'label': label}
        if config['colors']:
            attrs['color'] = config['colors_map']['commit']
        commit_sg.node(commit_nid, **attrs)

        # Edge to its tree
        try:
            tree_obj = git_repo.get_tree(tree_hash)
            tree_nid = (make_node_id(prefix, "tree", tree_obj.name) if
                        (config.get('group_enabled', False) and
                         'tree' in config.get('group_types', []))
                        else make_node_id(prefix, "tree", tree_hash))
            edge_key = (commit_nid, tree_nid)
            if edge_key not in drawn_edges:
                commit_sg.edge(commit_nid, tree_nid, minlen='2')
                drawn_edges.add(edge_key)
        except Exception as e:
            logging.warning("Error linking commit %s to tree %s: %s",
                            commit_hash, tree_hash, e)

    # Commit parent edges
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

    # Commit child edges
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
    branch_sg = create_subgraph(
        repo_cluster, f"cluster_{prefix}_branches",
        label="Branches", color="gray"
    )
    for branch_name, commit_hash in git_repo.branch_to_commit.items():
        branch_nid = make_node_id(prefix, "branch", branch_name)
        attrs = {'shape': 'parallelogram'}
        if config['colors']:
            attrs['color'] = config['colors_map']['branch']
        branch_sg.node(branch_nid, label=branch_name, **attrs)

        # dotted or solid edges for local/remote branches
        style = 'solid'
        # If slash in name => likely remote
        is_remote = ('/' in branch_name)
        if (is_remote and ('remote' in config['dotted'] or 'all' in config['dotted'])) \
           or (not is_remote and ('local' in config['dotted'] or 'all' in config['dotted'])):
            style = 'dotted'

        commit_nid = make_node_id(prefix, "commit", commit_hash)
        branch_sg.edge(branch_nid, commit_nid, style=style, minlen='2')
        
    # --- Legend (optional) ---
    if config['colors']:
        legend = create_subgraph(
            repo_cluster, f"cluster_{prefix}_legend",
            label="Legend", color="black"
        )
        for nodetype, col in config['colors_map'].items():
            node_id = f"legend_{nodetype}"
            legend.node(node_id, label=nodetype.capitalize(),
                        shape='box', style='filled', color=col)

# --- Combined Repository Base Mode ---
def generate_combined_repo_graph(repo_base_path: str, config: dict) -> None:
    """
    Recursively scan repo_base_path for Git repositories (directories with a .git folder)
    and generate one PDF document containing:
      1. A cluster for the overall project directory structure.
      2. A cluster for all detected Git repositories (each added as a subgraph).
    """
    abs_repo_base = os.path.abspath(repo_base_path)
    if abs_repo_base in ['/', os.path.sep]:
        logging.error("Refusing to scan system root '%s'", abs_repo_base)
        sys.exit(1)
    logging.info("Scanning for Git repositories under '%s'", repo_base_path)
    git_repos = []
    for root, dirs, _ in os.walk(repo_base_path, topdown=True, followlinks=False):
        # Skip directories that are outside the base path
        if os.path.abspath(root) != abs_repo_base and \
           not os.path.commonpath([abs_repo_base, os.path.abspath(root)]) == abs_repo_base:
            continue

        if '.git' in dirs:
            repo_abs = os.path.join(root)
            rel_path = os.path.relpath(repo_abs, repo_base_path)
            git_repos.append((repo_abs, rel_path))
            # do not descend further into repos
            dirs[:] = []

    if not git_repos:
        logging.error("No Git repositories found under '%s'", repo_base_path)
        sys.exit(1)
    logging.info("Found %d Git repositories.", len(git_repos))

    master = Digraph(comment='Combined Repository Graph', format='pdf')
    master.attr(compound='true', splines='true', overlap='false')

    # Project structure cluster
    proj_cluster = create_subgraph(
        master, "cluster_project_structure",
        label="Project Structure", color="black"
    )
    node_counter = [0]

    def process_directory(dir_path: str, parent_id: str = None,
                          depth: int = 0,
                          max_depth: int = config.get('max_depth', 10)):
        try:
            entries = os.listdir(dir_path)
        except Exception as e:
            logging.error("Cannot list directory '%s': %s", dir_path, e)
            return

        if depth > max_depth:
            return

        current_id = sanitize_id(os.path.abspath(dir_path))
        label = os.path.basename(dir_path) if os.path.basename(dir_path) else dir_path
        proj_cluster.node(current_id, label=label, shape='folder',
                          style='filled', color='lightblue')
        if parent_id:
            proj_cluster.edge(parent_id, current_id)

        for entry in sorted(entries):
            full_path = os.path.join(dir_path, entry)
            if not os.path.abspath(full_path).startswith(abs_repo_base):
                continue
            if os.path.isdir(full_path):
                process_directory(full_path, current_id, depth + 1, max_depth)
            else:
                file_id = sanitize_id(os.path.abspath(full_path)) + f"_{node_counter[0]}"
                node_counter[0] += 1
                proj_cluster.node(file_id, label=entry, shape='note',
                                  style='filled', color='lightyellow')
                proj_cluster.edge(current_id, file_id)

    process_directory(repo_base_path)

    # Git repositories cluster
    repos_cluster = create_subgraph(
        master, "cluster_git_repos",
        label="Git Repositories", color="blue"
    )
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
        output_file = 'combined_repo.gv'
        logging.info("Rendering combined graph to file '%s.pdf'", output_file)
        logging.debug("Graph source:\n%s", master.source)
        master.render(output_file, view=(not config['output_only']))
    except Exception as e:
        logging.error("Error rendering combined graph: %s", e)
        sys.exit(1)

# --- Graph Generation for Single Repository Mode ---
class GraphGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_graph(self, git_repo) -> None:
        logging.info("Generating Git object graph (single repository mode)")
        master = Digraph(comment='Git graph', format='pdf')
        master.attr(compound='true', splines='true', overlap='false')
        try:
            add_git_repo_subgraph(master, git_repo, self.config,
                                  prefix="single", repo_label="Repository")
        except Exception as e:
            logging.error("Error generating subgraph for repository: %s", e)
            sys.exit(1)

        try:
            output_file = 'git.gv'
            logging.info("Rendering graph to file '%s.pdf'", output_file)
            logging.debug("Graph source:\n%s", master.source)
            master.render(output_file, view=(not self.config['output_only']))
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
                    '  python3 git-graph.py --repo-base /path/to/project --colors --group --group-types blob,tree --verbose\n'
                    '  python3 git-graph.py --dump\n'
                    '  python3 git-graph.py --read-dump my_dump.json',
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
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth when scanning directories in repo-base mode (default: 10)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dump', action='store_true',
                       help='Generate a JSON dump of the repository data (instead of a graph)')
    group.add_argument('--read-dump', type=str,
                       help='Read repository dump from file and generate a graph')
    parser.add_argument('--dump-file', type=str, default="git_repo_dump.json",
                        help='Filename for dump output (used with --dump)')
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
        'group_condense_level': args.group_condense_level,
        'max_depth': args.max_depth
    }
    return {
        'repo_path': args.path,
        'config': config,
        'repo_base': args.repo_base,
        'dump': args.dump,
        'dump_file': args.dump_file,
        'read_dump': args.read_dump,
        'verbose': args.verbose
    }

def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args['verbose'] else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    check_dependencies()

    try:
        if args['dump']:
            # Dump mode
            repo_path = get_git_repo_path(args['repo_path'])
            git_repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            git_repo.parse_dot_git_dir()
            dump_git_repo(git_repo, args['dump_file'])
        elif args['read_dump']:
            # Read a dump file and generate a graph
            dumped_repo = load_git_repo_dump(args['read_dump'])
            GraphGenerator(args['config']).generate_graph(dumped_repo)
        elif args['repo_base']:
            # Combined multi-repo mode
            if not os.path.isdir(args['repo_base']):
                logging.error("Invalid repository base path: '%s'", args['repo_base'])
                sys.exit(1)
            generate_combined_repo_graph(args['repo_base'], args['config'])
        else:
            # Single-repo graph mode
            repo_path = get_git_repo_path(args['repo_path'])
            git_repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            git_repo.parse_dot_git_dir()
            GraphGenerator(args['config']).generate_graph(git_repo)
    except Exception as e:
        logging.exception("An unrecoverable error occurred: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
