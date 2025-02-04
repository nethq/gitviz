#!/usr/bin/env python3
"""
Git Graph Visualizer with Multi–Level Visual Grouping and Dump/Read–Dump Functionality

Modes:
  1) Default single–repo mode
  2) --repo-base to combine multiple repos into one PDF
  3) --dump to write JSON data describing the repo
  4) --read-dump to read a previously dumped JSON and render a graph

Supports:
  - Local only branches (–-local)
  - Limit commit parents/children edges (–P, –S)
  - Grouping node types (–-group, –-group-types)
  - Optional color legend
  - “Dotted” edges for local or remote branches
  - Dump/Load from JSON

Requires:
  - Python 3.x
  - `pip install graphviz`
  - Graphviz’s `dot` command in PATH
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

# -------------------------------------------------------------------
#                   UTILITY / HELPER FUNCTIONS
# -------------------------------------------------------------------

def sanitize_id(s: str) -> str:
    """Sanitize a string for use as a Graphviz node ID (remove weird chars)."""
    return re.sub(r'\W+', '_', s)

def make_node_id(prefix: str, typ: str, identifier: str) -> str:
    """Create a canonical node ID for graphviz, e.g. prefix_commit_abcd."""
    return f"{prefix}_{typ}_{sanitize_id(identifier)}"

def get_distinct_color(index: int, total: int) -> str:
    """
    Generate a distinct hex color using HLS approach.
    Returns a hex string (e.g. '#aabbcc').
    """
    if total <= 0:
        total = 1
    hue = (index / total) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def create_subgraph(parent: Digraph, name: str, label: str = None, color: str = None) -> Digraph:
    """
    Creates a new subgraph inside 'parent' with the specified name and label.
    We call __enter__() to treat it like a with-block, if supported.
    """
    sub = parent.subgraph(name=name)
    if hasattr(sub, '__enter__'):
        sub = sub.__enter__()
    if label:
        sub.attr(label=label)
    if color:
        sub.attr(color=color)
    return sub

def draw_grouped_nodes(subgraph: Digraph, nodes: List[dict], node_type: str,
                       prefix: str, config: dict) -> None:
    """
    For each node dictionary in 'nodes', group them by 'common'. If a group has only 1 node,
    just draw it as usual; if multiple, we use a record shape with each differentiator.
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for node in nodes:
        groups[node["common"]].append(node)
    total_groups = len(groups)
    group_index = 0
    for common, group_nodes in groups.items():
        if len(group_nodes) == 1:
            # Single node
            n = group_nodes[0]
            subgraph.node(n['id'], label=n['full_label'], **n.get('attrs', {}))
        else:
            # Build a record-like label
            diffs = "\\l".join(nd["differentiator"] for nd in group_nodes)
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

# -------------------------------------------------------------------
#                     GIT REPO MODEL AND PARSER
# -------------------------------------------------------------------

class GitRepo:
    """
    Represents a parsed Git repository: references (branches), commits, trees, blobs.
    """
    def __init__(self, git_repo_path: str, local_only: bool = False):
        self.git_repo_path = os.path.abspath(git_repo_path)
        self.dot_git_dir = os.path.join(self.git_repo_path, '.git')
        self.local_only = local_only

        # Cache of objects read from cat-file
        self.cache: Dict[Hash, object] = {}
        # Lists/dicts for branches => commits
        self.branches: List[Branch] = []
        self.branch_to_commit: Dict[str, Hash] = {}
        self.commit_to_branches: Dict[Hash, List[str]] = defaultdict(list)
        # Commit relationships
        self.commit_to_parents: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.commit_to_children: Dict[Hash, Set[Hash]] = defaultdict(set)
        # Commit => associated tree
        self.commit_to_tree: Dict[Hash, Hash] = {}
        # Tree => subtrees, Tree => blobs
        self.tree_to_trees: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.tree_to_blobs: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.blobs: Dict[Hash, Blob] = {}
        # Optional commit stats from 'git show --stat'
        self.commit_gitstat: Dict[Hash, str] = {}
        
        # Internal store of discovered references
        self._all_refs: Dict[str, str] = {}

    def parse_dot_git_dir(self) -> None:
        """
        Main entry point: discovers references, parses commits, builds relationships.
        """
        logging.info("Parsing .git directory at '%s'", self.dot_git_dir)
        self._gather_references()

        # Turn references into Branch objects
        self.branches.clear()
        for ref_name, commit_hash in self._all_refs.items():
            if not commit_hash:
                continue
            is_remote = ref_name.startswith("refs/remotes/")
            if self.local_only and is_remote:
                continue
            # short name
            short_name = ref_name
            if ref_name.startswith("refs/heads/"):
                short_name = ref_name.replace("refs/heads/", "", 1)
            elif ref_name.startswith("refs/remotes/"):
                short_name = ref_name.replace("refs/remotes/", "", 1)
            self.branches.append(Branch(name=short_name, commit=commit_hash, remote=is_remote))
            self.branch_to_commit[short_name] = commit_hash
            self.commit_to_branches[commit_hash].append(short_name)

        # If somehow no references were found, fallback to HEAD if it’s a direct hash
        if not self.branches:
            head_file = os.path.join(self.dot_git_dir, 'HEAD')
            if os.path.isfile(head_file):
                head_val = self._read_txt(head_file)
                if re.match(r'^[0-9a-f]{4,40}$', head_val):
                    # Treat HEAD as a single 'HEAD' branch
                    logging.info("No references found; using HEAD as a detached branch.")
                    self.branches.append(Branch(name='HEAD', commit=head_val, remote=False))
                    self.branch_to_commit['HEAD'] = head_val
                    self.commit_to_branches[head_val].append('HEAD')

        # Traverse each branch's commit history
        visited: Set[Hash] = set()
        for br in self.branches:
            try:
                self._traverse_history(br.commit, visited)
            except Exception as e:
                logging.warning("Unable to traverse branch %s: %s", br.name, e)

        # Build commit children from commit_to_parents
        self._build_commit_children()

    # -------------- GATHERING REFERENCES (HEAD, loose refs, packed-refs) --------------

    def _gather_references(self) -> None:
        """
        Gathers references from HEAD, .git/refs, and .git/packed-refs.
        Fills self._all_refs with e.g. {'refs/heads/main': <hash>, 'refs/remotes/origin/main': <hash>, etc.}
        """
        self._all_refs.clear()

        # 1) HEAD
        head_file = os.path.join(self.dot_git_dir, 'HEAD')
        if os.path.isfile(head_file):
            head_content = self._read_txt(head_file)
            if head_content.startswith('ref: '):
                # symbolic ref: "ref: refs/heads/main"
                ref_path = head_content[5:].strip()
                commit_hash = self._resolve_ref(ref_path)
                if commit_hash:
                    self._all_refs[ref_path] = commit_hash
            else:
                # Possibly a direct commit hash (detached HEAD)
                if re.match(r'^[0-9a-fA-F]{4,40}$', head_content):
                    self._all_refs['HEAD'] = head_content

        # 2) Loose references in .git/refs/
        refs_base = os.path.join(self.dot_git_dir, 'refs')
        if os.path.isdir(refs_base):
            for root, dirs, files in os.walk(refs_base):
                for f in files:
                    full_path = os.path.join(root, f)
                    ref_name = os.path.relpath(full_path, self.dot_git_dir).replace("\\", "/")
                    val = self._read_txt(full_path)
                    if re.match(r'^[0-9a-fA-F]{4,40}$', val):
                        self._all_refs[ref_name] = val
                    else:
                        logging.debug("Skipping non-hash ref %s => %s", ref_name, val)

        # 3) packed-refs
        packed_refs = os.path.join(self.dot_git_dir, 'packed-refs')
        if os.path.isfile(packed_refs):
            try:
                with open(packed_refs, 'r') as pf:
                    for line in pf:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue
                        if line.startswith('^'):
                            # line with '^<object>', typically the tag’s object
                            continue
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            ref_hash, ref_full = parts
                            if re.match(r'^[0-9a-fA-F]{4,40}$', ref_hash):
                                # store it
                                self._all_refs[ref_full] = ref_hash
            except Exception as e:
                logging.warning("Error reading packed-refs: %s", e)

    def _resolve_ref(self, ref_path: str) -> Optional[str]:
        """
        Given 'refs/heads/main' or similar, resolve to a commit hash if possible.
        """
        # 1) If there's a loose ref file
        full_ref = os.path.join(self.dot_git_dir, ref_path)
        if os.path.isfile(full_ref):
            val = self._read_txt(full_ref)
            if re.match(r'^[0-9a-fA-F]{4,40}$', val):
                return val
        # 2) Check in packed-refs
        packed_refs = os.path.join(self.dot_git_dir, 'packed-refs')
        if os.path.isfile(packed_refs):
            try:
                with open(packed_refs, 'r') as pf:
                    for line in pf:
                        line = line.strip()
                        if not line or line.startswith('#') or line.startswith('^'):
                            continue
                        parts = line.split(None, 1)
                        if len(parts) == 2:
                            ref_hash, name = parts
                            if name == ref_path and re.match(r'^[0-9a-fA-F]{4,40}$', ref_hash):
                                return ref_hash
            except Exception as e:
                logging.warning("Error searching packed-refs for %s: %s", ref_path, e)
        return None

    # ----------------- HISTORY TRAVERSAL / COMMIT PARSING -------------------

    def _traverse_history(self, commit_hash: Hash, visited: Set[Hash]) -> None:
        """Recursively load commits from commit_hash back through all parents."""
        if not commit_hash or commit_hash in visited:
            return
        visited.add(commit_hash)

        try:
            cobj = self.get_commit(commit_hash)
        except Exception as e:
            logging.warning("Skipping commit %s: %s", commit_hash, e)
            return

        for parent in cobj.parents:
            self.commit_to_parents[commit_hash].add(parent)
            self._traverse_history(parent, visited)

    def _build_commit_children(self) -> None:
        """Inverts commit_to_parents so child -> parent becomes parent -> child."""
        for child, parents in self.commit_to_parents.items():
            for p in parents:
                self.commit_to_children[p].add(child)

    # ----------------- LAZY LOAD COMMIT / TREE / BLOB ---------------------

    def get_commit(self, hash: Hash) -> Commit:
        """Return commit from cache or cat-file. Raise if not found."""
        if hash in self.cache and isinstance(self.cache[hash], Commit):
            return self.cache[hash]  # type: ignore

        content = self._git_cat_file(hash)
        commit_obj = self._parse_commit(hash, content)
        self.cache[hash] = commit_obj

        # store commit->tree
        self.commit_to_tree[commit_obj.hash] = commit_obj.tree

        # load that tree (in case we want to link commits->trees->blobs)
        try:
            self.get_tree(commit_obj.tree)
        except Exception as e:
            logging.warning("Could not load tree %s for commit %s: %s", commit_obj.tree, hash, e)

        return commit_obj

    def _parse_commit(self, hash: Hash, content: str) -> Commit:
        """Parse the raw commit text to a Commit namedtuple."""
        tree_hash = None
        parents = []
        author = "Unknown"
        author_found = False

        for line in content.splitlines():
            line = line.rstrip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key == 'tree' and len(parts) >= 2:
                tree_hash = parts[1]
            elif key == 'parent' and len(parts) >= 2:
                parents.append(parts[1])
            elif key == 'author' and not author_found:
                author_found = True
                author = ' '.join(parts[1:]) if len(parts) > 1 else "Unknown"

        if not tree_hash:
            raise ValueError(f"Commit {hash} missing tree pointer.")
        return Commit(hash=hash, tree=tree_hash, parents=parents, author=author)

    def get_tree(self, hash: Hash, name: str = '/') -> Tree:
        """Return tree object from cache or cat-file."""
        if hash in self.cache and isinstance(self.cache[hash], Tree):
            return self.cache[hash]  # type: ignore

        content = self._git_cat_file(hash)
        tree_obj = self._parse_tree(hash, name, content)
        self.cache[hash] = tree_obj

        # fill tree->blobs, tree->trees
        for b in tree_obj.blobs:
            self.tree_to_blobs[hash].add(b)
        for t in tree_obj.trees:
            self.tree_to_trees[hash].add(t)
        return tree_obj

    def _parse_tree(self, hash: Hash, name: str, content: str) -> Tree:
        """Parse the raw tree text to a Tree namedtuple."""
        subtrees = []
        blobs = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # typical: "100644 blob a1b2c3d4 fileName"
            parts = re.split(r'\s+', line, maxsplit=3)
            if len(parts) < 4:
                logging.debug("Skipping malformed tree entry in %s: %r", hash, line)
                continue
            mode, obj_type, child_hash, child_name = parts
            if obj_type == 'tree':
                subtrees.append(child_hash)
                # parse that subtree on-demand
                try:
                    self.get_tree(child_hash, child_name)
                except Exception as e:
                    logging.warning("Error reading subtree %s: %s", child_hash, e)
            elif obj_type == 'blob':
                blobs.append(child_hash)
                self.blobs[child_hash] = Blob(hash=child_hash, name=child_name)
        return Tree(hash=hash, name=name, trees=subtrees, blobs=blobs)

    # ----------------- SHELL COMMANDS ----------------------

    def _git_cat_file(self, hash: Hash) -> str:
        """Runs 'git cat-file -p <hash>' and returns content."""
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
            raise Exception(f"git cat-file failed for {hash}: {err}")

    def get_git_stat(self, commit_hash: Hash) -> str:
        """Show short stats for a commit."""
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
            logging.debug("git show --stat failed for %s: %s", commit_hash, err)
            stat = ""
        self.commit_gitstat[commit_hash] = stat
        return stat

    # ----------------- MISC READ FILE ----------------------

    def _read_txt(self, path: str) -> str:
        """Return content of a text file or empty string on error."""
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logging.debug("Could not read file %s: %s", path, e)
            return ""

# -------------------------------------------------------------------
#             DUMP / LOAD REPOSITORY TO/FROM JSON
# -------------------------------------------------------------------

def dump_git_repo(git_repo: GitRepo, dump_file: str) -> None:
    """
    Serialize the repository data into a JSON file.
    """
    data = {
        "branches": [b._asdict() for b in git_repo.branches],
        "commits": {
            h: c._asdict() for h, c in git_repo.cache.items()
            if isinstance(c, Commit)
        },
        "trees": {
            h: t._asdict() for h, t in git_repo.cache.items()
            if isinstance(t, Tree)
        },
        "blobs": {h: b._asdict() for h, b in git_repo.blobs.items()},

        "commit_to_tree": git_repo.commit_to_tree,
        "commit_to_parents": {k: list(v) for k, v in git_repo.commit_to_parents.items()},
        "commit_to_children": {k: list(v) for k, v in git_repo.commit_to_children.items()},
        "commit_to_branches": git_repo.commit_to_branches,
        "branch_to_commit": git_repo.branch_to_commit,

        "tree_to_blobs": {k: list(v) for k, v in git_repo.tree_to_blobs.items()},
        "tree_to_trees": {k: list(v) for k, v in git_repo.tree_to_trees.items()},
        "commit_gitstat": git_repo.commit_gitstat
    }
    try:
        with open(dump_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info("Dumped repository data to '%s'", dump_file)
    except Exception as e:
        logging.error("Failed writing dump file '%s': %s", dump_file, e)
        sys.exit(1)

class DumpedRepo:
    """
    Lightweight repo-like object that can be used by the same graph logic.
    """
    def __init__(self, data: dict):
        self.branches = [SimpleNamespace(**b) for b in data.get("branches", [])]
        self.commits = data.get("commits", {})
        self.trees = data.get("trees", {})
        self.blobs = {
            h: SimpleNamespace(**b) for h, b in data.get("blobs", {}).items()
        }
        self.commit_to_tree = data.get("commit_to_tree", {})
        self.commit_to_parents = {k: set(v) for k, v in data.get("commit_to_parents", {}).items()}
        self.commit_to_children = {k: set(v) for k, v in data.get("commit_to_children", {}).items()}
        self.commit_to_branches = data.get("commit_to_branches", {})
        self.branch_to_commit = data.get("branch_to_commit", {})
        self.tree_to_blobs = {k: set(v) for k, v in data.get("tree_to_blobs", {}).items()}
        self.tree_to_trees = {k: set(v) for k, v in data.get("tree_to_trees", {}).items()}
        self.commit_gitstat = data.get("commit_gitstat", {})

    def get_commit(self, commit_hash: str) -> SimpleNamespace:
        c = self.commits.get(commit_hash)
        if not c:
            raise KeyError(f"DumpedRepo: commit {commit_hash} not found in dump.")
        return SimpleNamespace(**c)

    def get_tree(self, tree_hash: str) -> SimpleNamespace:
        t = self.trees.get(tree_hash)
        if not t:
            raise KeyError(f"DumpedRepo: tree {tree_hash} not found in dump.")
        return SimpleNamespace(**t)

    def get_git_stat(self, commit_hash: str) -> str:
        """Provide a similar method so the label code doesn't crash."""
        return self.commit_gitstat.get(commit_hash, "")

# -------------------------------------------------------------------
#               LOADING A REPO FROM AN EXISTING JSON DUMP
# -------------------------------------------------------------------

def load_git_repo_dump(dump_file: str) -> DumpedRepo:
    """
    Load the JSON dump file from disk into a DumpedRepo.
    """
    try:
        with open(dump_file, 'r') as f:
            data = json.load(f)
        logging.info("Loaded dump from '%s'", dump_file)
        return DumpedRepo(data)
    except Exception as e:
        logging.error("Failed loading dump file '%s': %s", dump_file, e)
        sys.exit(1)

# -------------------------------------------------------------------
#               GRAPH GENERATION (SINGLE REPO SUBGRAPH)
# -------------------------------------------------------------------

def format_commit_label(commit, repo_obj, config: dict) -> str:
    """
    Build a multi-line label for a commit node, based on user-chosen metadata.
    """
    label = commit.hash[:6]  # short hash
    meta = config['metadata']

    def want(key: str) -> bool:
        if meta == 'all':
            return True
        if isinstance(meta, set) and key in meta:
            return True
        return False

    # show author
    if want("author"):
        label += "\n" + commit.author

    # show local/remote flags
    if want("flags"):
        # e.g. L or R
        flags = []
        # commit_to_branches => hash -> [branch1, branch2]
        if commit.hash in repo_obj.commit_to_branches:
            for b in repo_obj.commit_to_branches[commit.hash]:
                # if slash => remote
                if '/' in b:
                    flags.append('R')
                else:
                    flags.append('L')
            flags = sorted(set(flags))
            if flags:
                label += "\n(" + "/".join(flags) + ")"

    # show short stats
    if want("gitstat"):
        stat = repo_obj.get_git_stat(commit.hash)
        if stat:
            label += "\n" + stat.strip()

    return label

def add_git_repo_subgraph(master: Digraph, repo_obj, config: dict,
                          prefix: str, repo_label: str) -> None:
    """
    Add all commits/branches/trees/blobs from 'repo_obj' into the given 'master' Digraph,
    inside a subgraph cluster. Node IDs are namespaced by 'prefix'.
    """
    drawn_edges = set()

    repo_cluster = create_subgraph(master,
                                   name=f"cluster_repo_{sanitize_id(prefix)}",
                                   label=repo_label,
                                   color='blue')

    # -------------- BLOB NODES --------------
    if config.get('group_enabled') and 'blob' in config.get('group_types', []):
        # group them by filename
        blob_list = []
        for blob_hash, blob in repo_obj.blobs.items():
            node_id = make_node_id(prefix, "blob", blob_hash)
            blob_label = f"{blob.name} {blob_hash[:6]}"
            blob_list.append({
                'id': node_id,
                'common': blob.name,
                'differentiator': blob_hash[:6],
                'full_label': blob_label,
                'attrs': {
                    'shape': 'ellipse',
                    'style': 'filled',
                    'color': config['colors_map']['blob'] if config['colors'] else 'black'
                }
            })
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_grouped_blobs", "Blobs", "gray")
        draw_grouped_nodes(sg, blob_list, "blob", prefix, config)
    else:
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_blobs", "Blobs", "gray")
        for blob_hash, blob in repo_obj.blobs.items():
            node_id = make_node_id(prefix, "blob", blob_hash)
            label = f"{blob.name} {blob_hash[:6]}"
            attrs = {'shape': 'ellipse', 'style': 'filled'}
            if config['colors']:
                attrs['color'] = config['colors_map']['blob']
            sg.node(node_id, label=label, **attrs)

    # -------------- TREE NODES --------------
    if config.get('group_enabled') and 'tree' in config.get('group_types', []):
        # We'll group trees by name
        all_tree_hashes = set(repo_obj.tree_to_blobs.keys()) | set(repo_obj.tree_to_trees.keys())
        tree_nodes = []
        for th in all_tree_hashes:
            try:
                tobj = repo_obj.get_tree(th)
            except Exception as e:
                logging.warning("Skipping tree %s: %s", th, e)
                continue
            node_id = make_node_id(prefix, "tree", tobj.name)
            full_label = f"{tobj.name} {tobj.hash[:6]}"
            tree_nodes.append({
                'id': node_id,
                'common': tobj.name,
                'differentiator': tobj.hash[:6],
                'full_label': full_label,
                'attrs': {
                    'shape': 'triangle',
                    'color': config['colors_map']['tree'] if config['colors'] else 'black'
                }
            })
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_grouped_trees", "Trees", "gray")
        draw_grouped_nodes(sg, tree_nodes, "tree", prefix, config)
    else:
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_trees", "Trees", "gray")
        # For each tree, connect to its blobs
        for th, blob_hashes in repo_obj.tree_to_blobs.items():
            try:
                tobj = repo_obj.get_tree(th)
            except Exception as e:
                logging.warning("Skipping tree %s: %s", th, e)
                continue
            tree_id = make_node_id(prefix, "tree",
                                   tobj.name if (config['group_enabled'] and 'tree' in config['group_types']) else th)
            label = f"{tobj.name} {tobj.hash[:6]}"
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            sg.node(tree_id, label=label, **attrs)

            for blob_h in blob_hashes:
                blob_id = make_node_id(prefix, "blob", blob_h)
                ekey = (tree_id, blob_id)
                if ekey not in drawn_edges:
                    sg.edge(tree_id, blob_id, minlen='2')
                    drawn_edges.add(ekey)

        # For each tree, connect to its subtrees
        for th, subtree_hashes in repo_obj.tree_to_trees.items():
            try:
                tobj = repo_obj.get_tree(th)
            except Exception as e:
                logging.warning("Skipping tree %s: %s", th, e)
                continue
            tree_id = make_node_id(prefix, "tree",
                                   tobj.name if (config['group_enabled'] and 'tree' in config['group_types']) else th)
            label = f"{tobj.name} {tobj.hash[:6]}"
            attrs = {'shape': 'triangle'}
            if config['colors']:
                attrs['color'] = config['colors_map']['tree']
            sg.node(tree_id, label=label, **attrs)

            for sub_h in subtree_hashes:
                try:
                    subtobj = repo_obj.get_tree(sub_h)
                except Exception as e:
                    logging.warning("Skipping subtree %s: %s", sub_h, e)
                    continue
                subtree_id = make_node_id(prefix, "tree",
                                          subtobj.name if (config['group_enabled'] and 'tree' in config['group_types']) else sub_h)
                subtree_label = f"{subtobj.name} {subtobj.hash[:6]}"
                sg.node(subtree_id, label=subtree_label, **attrs)
                ekey = (tree_id, subtree_id)
                if ekey not in drawn_edges:
                    sg.edge(tree_id, subtree_id, minlen='2')
                    drawn_edges.add(ekey)

    # -------------- COMMITS --------------
    commit_sg = create_subgraph(repo_cluster, f"cluster_{prefix}_commits", "Commits", "gray")

    # Build commit nodes and link them to their tree
    for commit_hash, tree_hash in repo_obj.commit_to_tree.items():
        try:
            cobj = repo_obj.get_commit(commit_hash)
        except Exception as e:
            logging.warning("Skipping commit %s: %s", commit_hash, e)
            continue
        commit_id = make_node_id(prefix, "commit", commit_hash)
        label = format_commit_label(cobj, repo_obj, config)
        attrs = {'shape': 'rectangle', 'style': 'filled', 'label': label}
        if config['colors']:
            attrs['color'] = config['colors_map']['commit']
        commit_sg.node(commit_id, **attrs)

        # Edge commit -> tree
        try:
            tobj = repo_obj.get_tree(tree_hash)
            tree_id = make_node_id(prefix, "tree",
                                   tobj.name if (config['group_enabled'] and 'tree' in config['group_types']) else tree_hash)
            ekey = (commit_id, tree_id)
            if ekey not in drawn_edges:
                commit_sg.edge(commit_id, tree_id, minlen='2')
                drawn_edges.add(ekey)
        except Exception as e:
            logging.warning("Could not link commit %s to tree %s: %s", commit_hash, tree_hash, e)

    # For parent edges (commit -> parent)
    limit_parents = config.get('predecessor_limit')
    for commit_hash, parents in repo_obj.commit_to_parents.items():
        commit_id = make_node_id(prefix, "commit", commit_hash)
        parent_list = list(parents)
        normal_parents = parent_list if limit_parents is None else parent_list[:limit_parents]
        extra_parents = [] if limit_parents is None else parent_list[limit_parents:]
        for p in normal_parents:
            pid = make_node_id(prefix, "commit", p)
            ekey = (commit_id, pid)
            if ekey not in drawn_edges:
                commit_sg.edge(commit_id, pid, minlen='2')
                drawn_edges.add(ekey)
        # Dotted for extras
        for p in extra_parents:
            pid = make_node_id(prefix, "commit", p)
            ekey = (commit_id, pid)
            if ekey not in drawn_edges:
                commit_sg.edge(commit_id, pid, style='dotted', minlen='2')
                drawn_edges.add(ekey)

    # For child edges (commit -> children), if we want forward edges
    limit_children = config.get('successor_limit')
    for parent_hash, children in repo_obj.commit_to_children.items():
        parent_id = make_node_id(prefix, "commit", parent_hash)
        child_list = list(children)
        normal_children = child_list if limit_children is None else child_list[:limit_children]
        extra_children = [] if limit_children is None else child_list[limit_children:]
        for ch in normal_children:
            cid = make_node_id(prefix, "commit", ch)
            ekey = (parent_id, cid)
            if ekey not in drawn_edges:
                commit_sg.edge(parent_id, cid, minlen='2')
                drawn_edges.add(ekey)
        for ch in extra_children:
            cid = make_node_id(prefix, "commit", ch)
            ekey = (parent_id, cid)
            if ekey not in drawn_edges:
                commit_sg.edge(parent_id, cid, style='dotted', minlen='2')
                drawn_edges.add(ekey)

    # -------------- BRANCHES --------------
    branch_sg = create_subgraph(repo_cluster, f"cluster_{prefix}_branches", "Branches", "gray")
    for branch_name, commit_hash in repo_obj.branch_to_commit.items():
        branch_id = make_node_id(prefix, "branch", branch_name)
        attrs = {'shape': 'parallelogram'}
        if config['colors']:
            attrs['color'] = config['colors_map']['branch']
        branch_sg.node(branch_id, label=branch_name, **attrs)

        # Decide edge style
        style = 'solid'
        is_remote = ('/' in branch_name)  # naive: slash => remote
        dotted_groups = config['dotted']  # e.g. ['remote','local']
        if (is_remote and ('remote' in dotted_groups or 'all' in dotted_groups)) \
           or (not is_remote and ('local' in dotted_groups or 'all' in dotted_groups)):
            style = 'dotted'

        commit_id = make_node_id(prefix, "commit", commit_hash)
        ekey = (branch_id, commit_id)
        if ekey not in drawn_edges:
            branch_sg.edge(branch_id, commit_id, style=style, minlen='2')
            drawn_edges.add(ekey)

    # -------------- LEGEND --------------
    if config['colors']:
        legend = create_subgraph(repo_cluster, f"cluster_{prefix}_legend", "Legend", "black")
        for nodetype, color in config['colors_map'].items():
            legend.node(f"legend_{prefix}_{nodetype}",
                        label=nodetype.capitalize(),
                        shape='box',
                        style='filled',
                        color=color)

# -------------------------------------------------------------------
#                COMBINED REPO MODE ( --repo-base )
# -------------------------------------------------------------------

def generate_combined_repo_graph(repo_base_path: str, config: dict) -> None:
    """
    Recursively search for .git directories and create one PDF with:
      - A "project structure" subgraph
      - Subgraphs for each discovered repo
    """
    abs_base = os.path.abspath(repo_base_path)
    if abs_base in ['/', os.path.sep]:
        logging.error("Refusing to scan system root '%s'", abs_base)
        sys.exit(1)

    logging.info("Scanning for Git repos under '%s'", abs_base)
    found_repos = []
    for root, dirs, files in os.walk(abs_base, topdown=True, followlinks=False):
        # skip if root is outside of base (should not happen in normal os.walk)
        if not os.path.commonpath([abs_base, os.path.abspath(root)]) == abs_base:
            continue
        if '.git' in dirs:
            found_repos.append(root)
            # do not descend further
            dirs[:] = []

    if not found_repos:
        logging.error("No Git repos found under '%s'", abs_base)
        sys.exit(1)

    master = Digraph(comment="Combined Repository Graph", format='pdf')
    master.attr(compound='true', splines='true', overlap='false')

    # 1) Add the directory structure cluster
    proj_cluster = create_subgraph(master, "cluster_project_structure", "Project Structure", "black")

    node_id_counter = [0]

    def walk_dir(path: str, parent: str = None, depth: int = 0, maxd: int = config.get('max_depth', 10)):
        if depth > maxd:
            return
        try:
            items = os.listdir(path)
        except Exception as e:
            logging.warning("Cannot list directory '%s': %s", path, e)
            return
        label = os.path.basename(path) or path
        this_id = sanitize_id(os.path.abspath(path))
        proj_cluster.node(this_id, label=label, shape='folder', style='filled', color='lightblue')
        if parent:
            proj_cluster.edge(parent, this_id)
        for it in sorted(items):
            fullp = os.path.join(path, it)
            if os.path.isdir(fullp):
                walk_dir(fullp, this_id, depth+1, maxd)
            else:
                # file
                fid = sanitize_id(os.path.abspath(fullp)) + "_" + str(node_id_counter[0])
                node_id_counter[0] += 1
                proj_cluster.node(fid, label=it, shape='note', style='filled', color='lightyellow')
                proj_cluster.edge(this_id, fid)

    walk_dir(abs_base)

    # 2) Add the "Git Repositories" cluster
    repos_cluster = create_subgraph(master, "cluster_git_repos", "Git Repositories", "blue")
    for idx, rdir in enumerate(found_repos):
        relp = os.path.relpath(rdir, abs_base)
        prefix = f"repo{idx}_{sanitize_id(relp)}"
        logging.info("Processing repo: %s (relative: %s)", rdir, relp)
        try:
            repo_obj = GitRepo(rdir, local_only=config['local_only'])
            repo_obj.parse_dot_git_dir()
        except Exception as e:
            logging.error("Error loading repo '%s': %s", rdir, e)
            continue
        add_git_repo_subgraph(repos_cluster, repo_obj, config, prefix, repo_label=relp)

    # Render
    outfile = 'combined_repo.gv'
    try:
        logging.info("Rendering combined graph to %s.pdf", outfile)
        logging.debug("Graph source:\n%s", master.source)
        master.render(outfile, view=(not config['output_only']))
    except Exception as e:
        logging.error("Error rendering combined repo graph: %s", e)
        sys.exit(1)

# -------------------------------------------------------------------
#          SIMPLE CLASS TO GENERATE SINGLE-REPO GRAPH
# -------------------------------------------------------------------

class GraphGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_graph(self, repo_obj) -> None:
        """
        Generate a PDF named 'git.gv.pdf' for a single repository object.
        """
        master = Digraph(comment='Git graph', format='pdf')
        master.attr(compound='true', splines='true', overlap='false')

        try:
            add_git_repo_subgraph(master, repo_obj, self.config,
                                  prefix="single", repo_label="Repository")
        except Exception as e:
            logging.error("Error building subgraph: %s", e)
            sys.exit(1)

        outname = 'git.gv'
        try:
            logging.info("Rendering graph to '%s.pdf'", outname)
            logging.debug("Graph source:\n%s", master.source)
            master.render(outname, view=(not self.config['output_only']))
        except Exception as e:
            logging.error("Error rendering graph: %s", e)
            sys.exit(1)

# -------------------------------------------------------------------
#                       CLI / MAIN
# -------------------------------------------------------------------

def check_dependencies() -> None:
    """Ensure 'dot' is in PATH."""
    if not shutil.which('dot'):
        logging.error('Graphviz "dot" command not found. Please install Graphviz.')
        sys.exit(1)

def get_git_repo_path(path: str) -> str:
    """Validate 'path' is a directory with .git inside."""
    if not os.path.isdir(path):
        logging.error("Invalid path for repository: '%s'", path)
        sys.exit(1)
    dotgit = os.path.join(path, '.git')
    if not os.path.isdir(dotgit):
        logging.error("No .git directory found in '%s'", path)
        sys.exit(1)
    return path

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate a Graphviz diagram of a Git repository (single or combined), "
                    "or dump the repo structure to a JSON file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('path', nargs='?', default=os.getcwd(),
                        help='Path to the Git repo (default current dir).')
    parser.add_argument('--colors', action='store_true',
                        help='Enable distinct colors for nodes plus a legend.')
    parser.add_argument('-o', '--output-only', action='store_true',
                        help='Save .pdf but do not open in viewer.')
    parser.add_argument('--local', action='store_true',
                        help='Show only local branches (skip remotes).')
    parser.add_argument('-P', '--predecessors', type=int, default=None,
                        help='Max parents per commit (others dotted).')
    parser.add_argument('-S', '--successors', type=int, default=None,
                        help='Max children per commit (others dotted).')
    parser.add_argument('--dotted', nargs='*', choices=['remote','local','all'], default=[],
                        help='Draw edges from these branch types in dotted style.')
    parser.add_argument('-M', '--metadata', type=str, default='all',
                        help='Comma-separated metadata for commits: e.g. author,flags,gitstat. Default=all')
    parser.add_argument('--group', action='store_true',
                        help='Enable grouping nodes with similar labels.')
    parser.add_argument('--group-types', type=str, default='blob,tree',
                        help='Node types to group: blob,tree,commit,branch (comma-separated).')
    parser.add_argument('--group-condense-level', type=int, default=1,
                        help='Condensation of grouped nodes. Default=1')
    parser.add_argument('--repo-base', type=str,
                        help='If given, scan for multiple repos within that folder, generate combined PDF.')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Directory traversal depth (repo-base mode).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dump', action='store_true',
                       help='Dump repository data to JSON, do not generate PDF.')
    group.add_argument('--read-dump', type=str,
                       help='Read data from a previously created JSON and generate a PDF.')
    parser.add_argument('--dump-file', type=str, default='git_repo_dump.json',
                        help='Where to write or read JSON dump (default git_repo_dump.json).')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging.')

    args = parser.parse_args()

    # Build config
    meta_val = args.metadata.strip()
    if meta_val != "all":
        meta_val = {m.strip().lower() for m in meta_val.split(',')}

    cfg = {
        'colors': args.colors,
        'output_only': args.output_only,
        'local_only': args.local,
        'predecessor_limit': args.predecessors,
        'successor_limit': args.successors,
        'dotted': args.dotted,
        'metadata': meta_val,
        'group_enabled': args.group,
        'group_types': [t.strip().lower() for t in args.group_types.split(',')] if args.group_types else [],
        'group_condense_level': args.group_condense_level,
        'max_depth': args.max_depth,
        'colors_map': {
            'blob': 'lightyellow',
            'tree': 'lightgreen',
            'commit': 'lightblue',
            'branch': 'orange'
        }
    }
    return {
        'repo_path': args.path,
        'repo_base': args.repo_base,
        'dump': args.dump,
        'read_dump': args.read_dump,
        'dump_file': args.dump_file,
        'verbose': args.verbose,
        'config': cfg
    }

def main() -> None:
    # Parse
    args = parse_arguments()
    # Set up logging
    loglevel = logging.DEBUG if args['verbose'] else logging.INFO
    logging.basicConfig(level=loglevel, format="%(levelname)s: %(message)s")

    # Check for 'dot'
    check_dependencies()

    # Decide mode
    try:
        if args['dump']:
            # Dump the single repo data
            repo_path = get_git_repo_path(args['repo_path'])
            repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            repo.parse_dot_git_dir()
            dump_git_repo(repo, args['dump_file'])

        elif args['read_dump']:
            # Load from JSON, then graph
            dr = load_git_repo_dump(args['read_dump'])
            GraphGenerator(args['config']).generate_graph(dr)

        elif args['repo_base']:
            # Combined multi-repo
            base = os.path.abspath(args['repo_base'])
            if not os.path.isdir(base):
                logging.error("Invalid --repo-base: '%s'", base)
                sys.exit(1)
            generate_combined_repo_graph(base, args['config'])

        else:
            # Single repo graph
            repo_path = get_git_repo_path(args['repo_path'])
            repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            repo.parse_dot_git_dir()
            GraphGenerator(args['config']).generate_graph(repo)

    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
