#!/usr/bin/env python3
"""
Git Graph Visualizer with Multi–Level Visual Grouping, Dump/Read–Dump,
and Alternative Interactive Visualization (HTML/vis-network).

Version: 99.102

Modes:
  1) Default single–repo mode (static PDF via Graphviz)
  2) Combined repository mode (--repo-base)
  3) Dump mode (--dump) and Read-dump mode (--read-dump)
  4) Alternative interactive visualization (--interactive)
  
Requirements:
  - Python 3.x
  - pip install graphviz
  - Graphviz “dot” in PATH for PDF generation
  - For interactive mode, the output HTML uses the vis-network JS library from CDN.
  
Usage examples:
  python3 git_graph.py /path/to/repo
  python3 git_graph.py --interactive /path/to/repo
  python3 git_graph.py --dump --dump-file repo_dump.json
  python3 git_graph.py --read-dump repo_dump.json --interactive
  python3 git_graph.py --repo-base /path/to/project --verbose
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

__version__ = "99.102"

# ---------------------- GIT OBJECT TYPES -------------------------
Branch = collections.namedtuple('Branch', 'name commit remote')
Commit = collections.namedtuple('Commit', 'hash tree parents author')
Tree   = collections.namedtuple('Tree',   'hash name trees blobs')
Blob   = collections.namedtuple('Blob',   'hash name')
Hash   = str

# ---------------------- HELPER FUNCTIONS -------------------------

def sanitize_id(s: str) -> str:
    """Sanitize a string for use as a Graphviz or data node ID."""
    return re.sub(r'\W+', '_', s)

def make_node_id(prefix: str, typ: str, identifier: str) -> str:
    """Create a canonical node ID, e.g. prefix_commit_abcd."""
    return f"{prefix}_{typ}_{sanitize_id(identifier)}"

def get_distinct_color(index: int, total: int) -> str:
    """Return a distinct hex color based on index/total using HLS."""
    if total <= 0:
        total = 1
    hue = (index / total) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def create_subgraph(parent: Digraph, name: str, label: str = None, color: str = None) -> Digraph:
    """Create and return a new subgraph inside parent."""
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
    Group nodes by their common label; draw a single record node for groups with multiple elements.
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

# ---------------------- GIT REPO MODEL & PARSER -------------------------

class GitRepo:
    """
    A parsed Git repository; caches commits, trees, blobs and reference relationships.
    """
    def __init__(self, git_repo_path: str, local_only: bool = False):
        self.git_repo_path = os.path.abspath(git_repo_path)
        self.dot_git_dir = os.path.join(self.git_repo_path, '.git')
        self.local_only = local_only

        self.cache: Dict[Hash, object] = {}
        self.branches: List[Branch] = []
        self.branch_to_commit: Dict[str, Hash] = {}
        self.commit_to_branches: Dict[Hash, List[str]] = defaultdict(list)
        self.commit_to_parents: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.commit_to_children: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.commit_to_tree: Dict[Hash, Hash] = {}
        self.tree_to_trees: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.tree_to_blobs: Dict[Hash, Set[Hash]] = defaultdict(set)
        self.blobs: Dict[Hash, Blob] = {}
        self.commit_gitstat: Dict[Hash, str] = {}
        self._all_refs: Dict[str, str] = {}

    def parse_dot_git_dir(self) -> None:
        logging.info("Parsing .git directory at '%s'", self.dot_git_dir)
        self._gather_references()

        self.branches.clear()
        for ref_name, commit_hash in self._all_refs.items():
            if not commit_hash:
                continue
            is_remote = ref_name.startswith("refs/remotes/")
            if self.local_only and is_remote:
                continue
            short_name = ref_name
            if ref_name.startswith("refs/heads/"):
                short_name = ref_name.replace("refs/heads/", "", 1)
            elif ref_name.startswith("refs/remotes/"):
                short_name = ref_name.replace("refs/remotes/", "", 1)
            self.branches.append(Branch(name=short_name, commit=commit_hash, remote=is_remote))
            self.branch_to_commit[short_name] = commit_hash
            self.commit_to_branches[commit_hash].append(short_name)

        if not self.branches:
            head_file = os.path.join(self.dot_git_dir, 'HEAD')
            if os.path.isfile(head_file):
                head_val = self._read_txt(head_file)
                if re.match(r'^[0-9a-f]{4,40}$', head_val):
                    logging.info("No refs found; using HEAD as a detached branch.")
                    self.branches.append(Branch(name='HEAD', commit=head_val, remote=False))
                    self.branch_to_commit['HEAD'] = head_val
                    self.commit_to_branches[head_val].append('HEAD')

        visited: Set[Hash] = set()
        for br in self.branches:
            try:
                self._traverse_history(br.commit, visited)
            except Exception as e:
                logging.warning("Unable to traverse branch %s: %s", br.name, e)

        self._build_commit_children()

    def _gather_references(self) -> None:
        self._all_refs.clear()
        head_file = os.path.join(self.dot_git_dir, 'HEAD')
        if os.path.isfile(head_file):
            head_content = self._read_txt(head_file)
            if head_content.startswith('ref: '):
                ref_path = head_content[5:].strip()
                commit_hash = self._resolve_ref(ref_path)
                if commit_hash:
                    self._all_refs[ref_path] = commit_hash
            else:
                if re.match(r'^[0-9a-fA-F]{4,40}$', head_content):
                    self._all_refs['HEAD'] = head_content
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
                            ref_hash, ref_full = parts
                            if re.match(r'^[0-9a-fA-F]{4,40}$', ref_hash):
                                self._all_refs[ref_full] = ref_hash
            except Exception as e:
                logging.warning("Error reading packed-refs: %s", e)

    def _resolve_ref(self, ref_path: str) -> Optional[str]:
        full_ref = os.path.join(self.dot_git_dir, ref_path)
        if os.path.isfile(full_ref):
            val = self._read_txt(full_ref)
            if re.match(r'^[0-9a-fA-F]{4,40}$', val):
                return val
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

    def _traverse_history(self, commit_hash: Hash, visited: Set[Hash]) -> None:
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
        for child, parents in self.commit_to_parents.items():
            for p in parents:
                self.commit_to_children[p].add(child)

    def get_commit(self, hash: Hash) -> Commit:
        if hash in self.cache and isinstance(self.cache[hash], Commit):
            return self.cache[hash]
        content = self._git_cat_file(hash)
        commit_obj = self._parse_commit(hash, content)
        self.cache[hash] = commit_obj
        self.commit_to_tree[commit_obj.hash] = commit_obj.tree
        try:
            self.get_tree(commit_obj.tree)
        except Exception as e:
            logging.warning("Could not load tree %s for commit %s: %s", commit_obj.tree, hash, e)
        return commit_obj

    def _parse_commit(self, hash: Hash, content: str) -> Commit:
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
        if hash in self.cache and isinstance(self.cache[hash], Tree):
            return self.cache[hash]
        content = self._git_cat_file(hash)
        tree_obj = self._parse_tree(hash, name, content)
        self.cache[hash] = tree_obj
        for b in tree_obj.blobs:
            self.tree_to_blobs[hash].add(b)
        for t in tree_obj.trees:
            self.tree_to_trees[hash].add(t)
        return tree_obj

    def _parse_tree(self, hash: Hash, name: str, content: str) -> Tree:
        subtrees = []
        blobs = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = re.split(r'\s+', line, maxsplit=3)
            if len(parts) < 4:
                logging.debug("Skipping malformed tree entry in %s: %r", hash, line)
                continue
            mode, obj_type, child_hash, child_name = parts
            if obj_type == 'tree':
                subtrees.append(child_hash)
                try:
                    self.get_tree(child_hash, child_name)
                except Exception as e:
                    logging.warning("Error reading subtree %s: %s", child_hash, e)
            elif obj_type == 'blob':
                blobs.append(child_hash)
                self.blobs[child_hash] = Blob(hash=child_hash, name=child_name)
        return Tree(hash=hash, name=name, trees=subtrees, blobs=blobs)

    def _git_cat_file(self, hash: Hash) -> str:
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

    def _read_txt(self, path: str) -> str:
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except Exception as e:
            logging.debug("Could not read file %s: %s", path, e)
            return ""

# ---------------------- DUMP / READ DUMP SUPPORT -------------------------

def dump_git_repo(git_repo: GitRepo, dump_file: str) -> None:
    data = {
        "branches": [b._asdict() for b in git_repo.branches],
        "commits": {h: c._asdict() for h, c in git_repo.cache.items() if isinstance(c, Commit)},
        "trees": {h: t._asdict() for h, t in git_repo.cache.items() if isinstance(t, Tree)},
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
    A lightweight repository object built from a JSON dump.
    Supports get_commit, get_tree, and get_git_stat.
    """
    def __init__(self, data: dict):
        self.branches = [SimpleNamespace(**b) for b in data.get("branches", [])]
        self.commits = data.get("commits", {})
        self.trees = data.get("trees", {})
        self.blobs = {h: SimpleNamespace(**b) for h, b in data.get("blobs", {}).items()}
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
            raise KeyError(f"DumpedRepo: commit {commit_hash} not found.")
        return SimpleNamespace(**c)

    def get_tree(self, tree_hash: str) -> SimpleNamespace:
        t = self.trees.get(tree_hash)
        if not t:
            raise KeyError(f"DumpedRepo: tree {tree_hash} not found.")
        return SimpleNamespace(**t)

    def get_git_stat(self, commit_hash: str) -> str:
        return self.commit_gitstat.get(commit_hash, "")

def load_git_repo_dump(dump_file: str) -> DumpedRepo:
    try:
        with open(dump_file, 'r') as f:
            data = json.load(f)
        logging.info("Loaded dump from '%s'", dump_file)
        return DumpedRepo(data)
    except Exception as e:
        logging.error("Failed loading dump file '%s': %s", dump_file, e)
        sys.exit(1)

# ---------------------- GRAPH GENERATION (STATIC PDF) -------------------------

def format_commit_label(commit, repo_obj, config: dict) -> str:
    label = commit.hash[:6]
    meta = config['metadata']
    def want(key: str) -> bool:
        return meta == 'all' or (isinstance(meta, set) and key in meta)
    if want("author"):
        label += "\n" + commit.author
    if want("flags"):
        flags = []
        if commit.hash in getattr(repo_obj, 'commit_to_branches', {}):
            for b in repo_obj.commit_to_branches[commit.hash]:
                flags.append("R" if '/' in b else "L")
            flags = sorted(set(flags))
            if flags:
                label += "\n(" + "/".join(flags) + ")"
    if want("gitstat"):
        stat = repo_obj.get_git_stat(commit.hash)
        if stat:
            label += "\n" + stat.strip()
    return label

def add_git_repo_subgraph(master: Digraph, repo_obj, config: dict,
                          prefix: str, repo_label: str) -> None:
    drawn_edges = set()
    repo_cluster = create_subgraph(master, f"cluster_repo_{sanitize_id(prefix)}", label=repo_label, color='blue')

    # BLOBS
    if config.get('group_enabled') and 'blob' in config.get('group_types', []):
        blob_list = []
        for blob_hash, blob in repo_obj.blobs.items():
            node_id = make_node_id(prefix, "blob", blob_hash)
            blob_label = f"{blob.name} {blob_hash[:6]}"
            blob_list.append({
                'id': node_id,
                'common': blob.name,
                'differentiator': blob_hash[:6],
                'full_label': blob_label,
                'attrs': {'shape': 'ellipse', 'style': 'filled',
                          'color': config['colors_map']['blob'] if config['colors'] else 'black'}
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

    # TREES
    if config.get('group_enabled') and 'tree' in config.get('group_types', []):
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
                'attrs': {'shape': 'triangle', 'color': config['colors_map']['tree'] if config['colors'] else 'black'}
            })
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_grouped_trees", "Trees", "gray")
        draw_grouped_nodes(sg, tree_nodes, "tree", prefix, config)
    else:
        sg = create_subgraph(repo_cluster, f"cluster_{prefix}_trees", "Trees", "gray")
        for th, blob_hashes in repo_obj.tree_to_blobs.items():
            try:
                tobj = repo_obj.get_tree(th)
            except Exception as e:
                logging.warning("Skipping tree %s: %s", th, e)
                continue
            tree_id = make_node_id(prefix, "tree", tobj.name if (config.get('group_enabled') and 'tree' in config.get('group_types', [])) else th)
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
        for th, subtree_hashes in repo_obj.tree_to_trees.items():
            try:
                tobj = repo_obj.get_tree(th)
            except Exception as e:
                logging.warning("Skipping tree %s: %s", th, e)
                continue
            tree_id = make_node_id(prefix, "tree", tobj.name if (config.get('group_enabled') and 'tree' in config.get('group_types', [])) else th)
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
                subtree_id = make_node_id(prefix, "tree", subtobj.name if (config.get('group_enabled') and 'tree' in config.get('group_types', [])) else sub_h)
                subtree_label = f"{subtobj.name} {subtobj.hash[:6]}"
                sg.node(subtree_id, label=subtree_label, **attrs)
                ekey = (tree_id, subtree_id)
                if ekey not in drawn_edges:
                    sg.edge(tree_id, subtree_id, minlen='2')
                    drawn_edges.add(ekey)

    # COMMITS
    commit_sg = create_subgraph(repo_cluster, f"cluster_{prefix}_commits", "Commits", "gray")
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
        try:
            tobj = repo_obj.get_tree(tree_hash)
            tree_id = make_node_id(prefix, "tree", tobj.name if (config.get('group_enabled') and 'tree' in config.get('group_types', [])) else tree_hash)
            ekey = (commit_id, tree_id)
            if ekey not in drawn_edges:
                commit_sg.edge(commit_id, tree_id, minlen='2')
                drawn_edges.add(ekey)
        except Exception as e:
            logging.warning("Could not link commit %s to tree %s: %s", commit_hash, tree_hash, e)
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
        for p in extra_parents:
            pid = make_node_id(prefix, "commit", p)
            ekey = (commit_id, pid)
            if ekey not in drawn_edges:
                commit_sg.edge(commit_id, pid, style='dotted', minlen='2')
                drawn_edges.add(ekey)
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
    # BRANCHES
    branch_sg = create_subgraph(repo_cluster, f"cluster_{prefix}_branches", "Branches", "gray")
    for branch_name, commit_hash in repo_obj.branch_to_commit.items():
        branch_id = make_node_id(prefix, "branch", branch_name)
        attrs = {'shape': 'parallelogram'}
        if config['colors']:
            attrs['color'] = config['colors_map']['branch']
        branch_sg.node(branch_id, label=branch_name, **attrs)
        style = 'solid'
        if (('/' in branch_name and ('remote' in config['dotted'] or 'all' in config['dotted'])) or
            ('/' not in branch_name and ('local' in config['dotted'] or 'all' in config['dotted']))):
            style = 'dotted'
        commit_id = make_node_id(prefix, "commit", commit_hash)
        ekey = (branch_id, commit_id)
        if ekey not in drawn_edges:
            branch_sg.edge(branch_id, commit_id, style=style, minlen='2')
            drawn_edges.add(ekey)
    if config['colors']:
        legend = create_subgraph(repo_cluster, f"cluster_{prefix}_legend", "Legend", "black")
        for nodetype, color in config['colors_map'].items():
            legend.node(f"legend_{prefix}_{nodetype}", label=nodetype.capitalize(),
                        shape='box', style='filled', color=color)

# ---------------------- INTERACTIVE VISUALIZATION (HTML) -------------------------

def build_graph_data(repo_obj, config: dict, prefix: str = "interactive") -> Dict[str, List[dict]]:
    """
    Build a dictionary with two lists: 'nodes' and 'edges', for interactive visualization.
    Each node is a dict with id, label, group, and optional color and shape.
    Each edge is a dict with 'from', 'to', and optionally 'dashes' (for dotted).
    This function is designed to work with both GitRepo and DumpedRepo objects.
    """
    nodes = {}
    edges = []

    # Helper to add a node if not already added.
    def add_node(node_id: str, label: str, group: str, color: Optional[str]=None, shape: Optional[str]=None):
        if node_id not in nodes:
            nodes[node_id] = {
                "id": node_id,
                "label": label,
                "group": group
            }
            if color:
                nodes[node_id]["color"] = color
            if shape:
                nodes[node_id]["shape"] = shape

    # BLOBS: iterate over repo_obj.blobs
    for blob_hash, blob in repo_obj.blobs.items():
        node_id = make_node_id(prefix, "blob", blob_hash)
        label = f"{blob.name} {blob_hash[:6]}"
        add_node(node_id, label, "blob", color=config['colors_map']['blob'] if config['colors'] else None, shape="ellipse")

    # TREES: iterate over trees; for GitRepo, trees are in cache; for DumpedRepo, in self.trees.
    if hasattr(repo_obj, "get_tree"):
        tree_ids = set()
        try:
            # Use keys from tree_to_blobs and tree_to_trees if available
            tree_ids = set(repo_obj.tree_to_blobs.keys()) | set(repo_obj.tree_to_trees.keys())
        except Exception:
            # Fallback: if repo_obj has attribute "trees"
            if hasattr(repo_obj, "trees"):
                tree_ids = set(repo_obj.trees.keys())
        for t in tree_ids:
            try:
                tobj = repo_obj.get_tree(t)
            except Exception as e:
                logging.debug("Skipping tree %s: %s", t, e)
                continue
            node_id = make_node_id(prefix, "tree", tobj.name)
            label = f"{tobj.name} {tobj.hash[:6]}"
            add_node(node_id, label, "tree", color=config['colors_map']['tree'] if config['colors'] else None, shape="triangle")

    # COMMITS: iterate over commit_to_tree keys
    for commit_hash, tree_hash in repo_obj.commit_to_tree.items():
        try:
            cobj = repo_obj.get_commit(commit_hash)
        except Exception as e:
            logging.debug("Skipping commit %s: %s", commit_hash, e)
            continue
        node_id = make_node_id(prefix, "commit", commit_hash)
        label = format_commit_label(cobj, repo_obj, config)
        add_node(node_id, label, "commit", color=config['colors_map']['commit'] if config['colors'] else None, shape="box")

    # BRANCHES: iterate over branch_to_commit
    for branch_name, commit_hash in repo_obj.branch_to_commit.items():
        node_id = make_node_id(prefix, "branch", branch_name)
        label = branch_name
        add_node(node_id, label, "branch", color=config['colors_map']['branch'] if config['colors'] else None, shape="dot")

    # EDGES:
    # Tree -> Blob edges:
    for tree_hash, blob_hashes in getattr(repo_obj, "tree_to_blobs", {}).items():
        try:
            tobj = repo_obj.get_tree(tree_hash)
        except Exception:
            continue
        tree_id = make_node_id(prefix, "tree", tobj.name)
        for blob_h in blob_hashes:
            blob_id = make_node_id(prefix, "blob", blob_h)
            edges.append({"from": tree_id, "to": blob_id})
    # Tree -> Tree edges:
    for tree_hash, subtree_hashes in getattr(repo_obj, "tree_to_trees", {}).items():
        try:
            tobj = repo_obj.get_tree(tree_hash)
        except Exception:
            continue
        tree_id = make_node_id(prefix, "tree", tobj.name)
        for sub_h in subtree_hashes:
            try:
                subtobj = repo_obj.get_tree(sub_h)
            except Exception:
                continue
            subtree_id = make_node_id(prefix, "tree", subtobj.name)
            edges.append({"from": tree_id, "to": subtree_id})
    # Commit -> Tree edges:
    for commit_hash, tree_hash in repo_obj.commit_to_tree.items():
        try:
            tobj = repo_obj.get_tree(tree_hash)
        except Exception:
            continue
        commit_id = make_node_id(prefix, "commit", commit_hash)
        tree_id = make_node_id(prefix, "tree", tobj.name)
        edges.append({"from": commit_id, "to": tree_id})
    # Commit -> Parent commit edges:
    for commit_hash, parents in repo_obj.commit_to_parents.items():
        commit_id = make_node_id(prefix, "commit", commit_hash)
        for p in parents:
            parent_id = make_node_id(prefix, "commit", p)
            edges.append({"from": commit_id, "to": parent_id})
    # Branch -> Commit edges:
    for branch_name, commit_hash in repo_obj.branch_to_commit.items():
        branch_id = make_node_id(prefix, "branch", branch_name)
        commit_id = make_node_id(prefix, "commit", commit_hash)
        style = {}
        if (('/' in branch_name and ('remote' in config['dotted'] or 'all' in config['dotted'])) or
            ('/' not in branch_name and ('local' in config['dotted'] or 'all' in config['dotted']))):
            style["dashes"] = True
        edge = {"from": branch_id, "to": commit_id}
        edge.update(style)
        edges.append(edge)
    return {"nodes": list(nodes.values()), "edges": edges}

def generate_interactive_visualization(repo_obj, config: dict,
                                       prefix: str = "interactive",
                                       repo_label: str = "Repository",
                                       output_file: str = "git_interactive.html",
                                       view: bool = True) -> None:
    """
    Generate an HTML file that uses vis-network to display the graph interactively.
    """
    try:
        graph_data = build_graph_data(repo_obj, config, prefix)
        # Build HTML with embedded JSON data and vis-network code.
        html_template = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Git Graph Interactive Visualization</title>
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    #network {{
      width: 100%;
      height: 100vh;
      border: 1px solid lightgray;
    }}
  </style>
</head>
<body>
  <div id="network"></div>
  <script type="text/javascript">
    const container = document.getElementById('network');
    const data = {json.dumps(graph_data)};
    const options = {{
      nodes: {{
        shape: 'dot',
        size: 16,
        font: {{
          size: 14,
          color: '#000'
        }}
      }},
      edges: {{
        width: 2,
        color: 'gray',
        smooth: {{
          type: 'continuous'
        }}
      }},
      physics: {{
        stabilization: false,
        barnesHut: {{
          gravitationalConstant: -8000,
          springConstant: 0.001,
          springLength: 200
        }}
      }},
      interaction: {{
        tooltipDelay: 200,
        hideEdgesOnDrag: true
      }}
    }};
    const network = new vis.Network(container, data, options);
  </script>
</body>
</html>
"""
        with open(output_file, 'w') as f:
            f.write(html_template)
        logging.info("Interactive visualization written to '%s'", output_file)
        if view:
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.realpath(output_file))
            except Exception as e:
                logging.warning("Could not open browser automatically: %s", e)
    except Exception as e:
        logging.error("Error generating interactive visualization: %s", e)
        sys.exit(1)

# ---------------------- GRAPH GENERATION (STATIC PDF) WRAPPER -------------------------

class GraphGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_graph(self, repo_obj) -> None:
        logging.info("Generating static Graphviz PDF for repository.")
        master = Digraph(comment='Git graph', format='pdf')
        master.attr(compound='true', splines='true', overlap='false')
        try:
            add_git_repo_subgraph(master, repo_obj, self.config, prefix="single", repo_label="Repository")
        except Exception as e:
            logging.error("Error building graph subcomponent: %s", e)
            sys.exit(1)
        output_name = 'git.gv'
        try:
            logging.info("Rendering graph to '%s.pdf'", output_name)
            logging.debug("Graph source:\n%s", master.source)
            master.render(output_name, view=(not self.config['output_only']))
        except Exception as e:
            logging.error("Error rendering PDF graph: %s", e)
            sys.exit(1)

# ---------------------- COMBINED REPO MODE -------------------------

def generate_combined_repo_graph(repo_base_path: str, config: dict) -> None:
    abs_base = os.path.abspath(repo_base_path)
    if abs_base in ['/', os.path.sep]:
        logging.error("Refusing to scan system root '%s'", abs_base)
        sys.exit(1)
    logging.info("Scanning for Git repositories under '%s'", abs_base)
    found_repos = []
    for root, dirs, _ in os.walk(abs_base, topdown=True, followlinks=False):
        if not os.path.commonpath([abs_base, os.path.abspath(root)]) == abs_base:
            continue
        if '.git' in dirs:
            found_repos.append(root)
            dirs[:] = []
    if not found_repos:
        logging.error("No Git repositories found under '%s'", abs_base)
        sys.exit(1)
    master = Digraph(comment="Combined Repository Graph", format='pdf')
    master.attr(compound='true', splines='true', overlap='false')
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
                fid = sanitize_id(os.path.abspath(fullp)) + "_" + str(node_id_counter[0])
                node_id_counter[0] += 1
                proj_cluster.node(fid, label=it, shape='note', style='filled', color='lightyellow')
                proj_cluster.edge(this_id, fid)
    walk_dir(abs_base)
    repos_cluster = create_subgraph(master, "cluster_git_repos", "Git Repositories", "blue")
    for idx, rdir in enumerate(found_repos):
        relp = os.path.relpath(rdir, abs_base)
        prefix = f"repo{idx}_{sanitize_id(relp)}"
        logging.info("Processing repository: %s (relative: %s)", rdir, relp)
        try:
            repo_obj = GitRepo(rdir, local_only=config['local_only'])
            repo_obj.parse_dot_git_dir()
        except Exception as e:
            logging.error("Error loading repository '%s': %s", rdir, e)
            continue
        add_git_repo_subgraph(repos_cluster, repo_obj, config, prefix, repo_label=relp)
    output_file = 'combined_repo.gv'
    try:
        logging.info("Rendering combined graph to '%s.pdf'", output_file)
        logging.debug("Graph source:\n%s", master.source)
        master.render(output_file, view=(not config['output_only']))
    except Exception as e:
        logging.error("Error rendering combined repository graph: %s", e)
        sys.exit(1)

# ---------------------- CLI & MAIN -------------------------

def check_dependencies() -> None:
    if not shutil.which('dot'):
        logging.error('Graphviz "dot" command not found. Please install Graphviz.')
        sys.exit(1)

def get_git_repo_path(path: str) -> str:
    if not os.path.isdir(path):
        logging.error("Invalid repository path: '%s'", path)
        sys.exit(1)
    dotgit = os.path.join(path, '.git')
    if not os.path.isdir(dotgit):
        logging.error("No .git directory found in '%s'", path)
        sys.exit(1)
    return path

def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(
        description="Generate a Graphviz (PDF) or interactive (HTML) visualization of a Git repository or combined repos.\n"
                    "Supports --dump and --read-dump modes as well.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('path', nargs='?', default=os.getcwd(), help='Path to the Git repo (default: current directory)')
    parser.add_argument('--colors', action='store_true', help='Enable distinct colors for nodes plus a legend.')
    parser.add_argument('-o', '--output-only', action='store_true', help='Save output file without automatically opening it.')
    parser.add_argument('--local', action='store_true', help='Show only local branches (skip remotes).')
    parser.add_argument('-P', '--predecessors', type=int, default=None, help='Maximum number of parent edges to show normally (others dotted).')
    parser.add_argument('-S', '--successors', type=int, default=None, help='Maximum number of child edges to show normally (others dotted).')
    parser.add_argument('--dotted', nargs='*', choices=['remote','local','all'], default=[], help='Force these branch edge types to be drawn dotted.')
    parser.add_argument('-M', '--metadata', type=str, default='all', help='Comma-separated metadata keys for commits (author,flags,gitstat). Default: all')
    parser.add_argument('--group', action='store_true', help='Enable grouping nodes with similar labels.')
    parser.add_argument('--group-types', type=str, default='blob,tree', help='Comma-separated node types to group (options: blob,tree,commit,branch).')
    parser.add_argument('--group-condense-level', type=int, default=1, help='Level of condensation for grouped nodes (default: 1).')
    parser.add_argument('--repo-base', type=str, help='If set, scan for multiple repos within this base folder and produce a combined diagram.')
    parser.add_argument('--max-depth', type=int, default=10, help='Maximum directory depth when scanning in repo-base mode (default: 10).')
    parser.add_argument('--dump', action='store_true', help='Dump repository data to JSON instead of visualizing.')
    parser.add_argument('--read-dump', type=str, help='Read repository data from JSON dump and visualize.')
    parser.add_argument('--dump-file', type=str, default='git_repo_dump.json', help='Filename for dump output/input (default: git_repo_dump.json).')
    parser.add_argument('--interactive', action='store_true', help='Generate an interactive HTML visualization instead of static PDF.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    args = parser.parse_args()

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
        'interactive': args.interactive,
        'verbose': args.verbose,
        'config': cfg
    }

def main() -> None:
    args = parse_arguments()
    log_level = logging.DEBUG if args['verbose'] else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    check_dependencies()
    try:
        if args['dump']:
            repo_path = get_git_repo_path(args['repo_path'])
            repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            repo.parse_dot_git_dir()
            dump_git_repo(repo, args['dump_file'])
        elif args['read_dump']:
            dumped_repo = load_git_repo_dump(args['read_dump'])
            if args['interactive']:
                generate_interactive_visualization(dumped_repo, args['config'], output_file="git_interactive.html", view=(not args['config']['output_only']))
            else:
                GraphGenerator(args['config']).generate_graph(dumped_repo)
        elif args['repo_base']:
            generate_combined_repo_graph(args['repo_base'], args['config'])
        else:
            repo_path = get_git_repo_path(args['repo_path'])
            repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            repo.parse_dot_git_dir()
            if args['interactive']:
                generate_interactive_visualization(repo, args['config'], output_file="git_interactive.html", view=(not args['config']['output_only']))
            else:
                GraphGenerator(args['config']).generate_graph(repo)
    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
