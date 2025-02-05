#!/usr/bin/env python3
"""
Git Graph Visualizer with Multi–Level Visual Grouping, Dump/Read‐Dump,
Dual PDF Engines, and Enhanced Interactive Visualization (minimal commit–chain),
with non-overlapping interactive mode, disabled "sway" (physics),
and multi–branch color edges.
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

# ---------------------- GIT OBJECT TYPES -------------------------
Branch = collections.namedtuple('Branch', 'name commit remote')
Commit = collections.namedtuple('Commit', 'hash tree parents author')
Tree   = collections.namedtuple('Tree',   'hash name trees blobs')
Blob   = collections.namedtuple('Blob',   'hash name')
Hash   = str

# ---------------------- HELPER FUNCTIONS -------------------------

def sanitize_id(s: str) -> str:
    """Sanitize a string for use as a node ID."""
    return re.sub(r'\W+', '_', s)

def make_node_id(prefix: str, typ: str, identifier: str) -> str:
    """Generate a namespaced node ID."""
    return f"{prefix}_{typ}_{sanitize_id(identifier)}"

def get_distinct_color(index: int, total: int) -> str:
    """Generate a distinct hex color based on HLS."""
    if total <= 0:
        total = 1
    hue = (index / total) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.6, 0.7)
    return '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))

def create_subgraph(parent: Digraph, name: str, label: str = None, color: str = None) -> Digraph:
    """Helper to create a subgraph in Graphviz."""
    sub = parent.subgraph(name=name)
    if hasattr(sub, '__enter__'):
        sub = sub.__enter__()
    if label:
        sub.attr(label=label)
    if color:
        sub.attr(color=color)
    return sub

# ---------------------- METADATA FORMATTING -------------------------

def format_commit_label(commit, repo_obj, config: dict) -> str:
    """Build a concise label for a commit node."""
    label = f"{commit.hash[:7]}"
    meta = config['metadata']
    def want(key: str) -> bool:
        return meta == 'all' or (isinstance(meta, set) and key in meta)
    if want("author"):
        label += f"\n{commit.author}"
    # Optionally show flags, stats, etc.
    if want("flags"):
        branches = repo_obj.commit_to_branches.get(commit.hash, [])
        if branches:
            label += f"\nBranches: {', '.join(branches)}"
    if want("gitstat"):
        stat = repo_obj.get_git_stat(commit.hash)
        if stat:
            label += f"\n{stat.strip()}"
    return label

def format_commit_tooltip(commit, repo_obj, config: dict) -> str:
    """Return an extended tooltip for a commit node."""
    tooltip = f"Commit: {commit.hash}\nAuthor: {commit.author}\n"
    branches = repo_obj.commit_to_branches.get(commit.hash, [])
    if branches:
        tooltip += f"Branches: {', '.join(branches)}\n"
    tooltip += f"Parents: {', '.join(commit.parents) if commit.parents else 'None'}\n"
    stat = repo_obj.get_git_stat(commit.hash)
    if stat:
        tooltip += f"Git Stats:\n{stat.strip()}"
    return tooltip

def format_tree_tooltip(tobj) -> str:
    """Return a tooltip for a tree node."""
    return f"Tree: {tobj.name}\nHash: {tobj.hash}\nSubtrees: {len(tobj.trees)}\nBlobs: {len(tobj.blobs)}"

def format_blob_tooltip(blob) -> str:
    """Return a tooltip for a blob node."""
    return f"Blob: {blob.name}\nHash: {blob.hash}"

# ---------------------- GIT REPO MODEL & PARSER -------------------------

class GitRepo:
    """
    Represents a parsed Git repository; caches commits, trees, blobs, and relationships.
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

        # If no branches found, check for detached HEAD
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

# ---------------------- INTERACTIVE VISUALIZATION (HTML) -------------------------
# Updated to:
#   - Use hierarchical layout WITHOUT live physics (nodes stay where they are).
#   - Color commits by branch membership. If a commit is in multiple branches,
#     it’s shown in gray with multiple edges of different colors.
#   - Non-overlapping arrangement so no elements overlap on initial load.

def build_graph_data(repo_obj, config: dict, prefix: str = "interactive") -> Dict[str, List[dict]]:
    """
    Build a minimal commit & branch graph for interactive mode, with multi-branch coloring:
      - If a commit is in exactly one branch, node color = that branch’s color.
      - If in multiple branches, node color = gray, edges show each branch color.
      - If commit & parent share multiple branches, multiple edges (one per color).
      - If commit & parent share no common branch, show a dashed black edge.
    """
    nodes = {}
    edges = []

    # Branch -> unique color
    branch_colors = {}
    branch_list = list(repo_obj.branch_to_commit.keys())
    total_branches = len(branch_list)
    for i, branch_name in enumerate(branch_list):
        branch_colors[branch_name] = get_distinct_color(i, total_branches)

    def add_node(node_id: str, label: str, group: str,
                 title: Optional[str] = None,
                 color: Optional[str] = None,
                 shape: Optional[str] = None):
        if node_id not in nodes:
            n = {"id": node_id, "label": label, "group": group}
            if title:
                n["title"] = title
            if color:
                n["color"] = color
            if shape:
                n["shape"] = shape
            nodes[node_id] = n

    # Create commit nodes
    all_commits = repo_obj.commit_to_tree.keys()
    for commit_hash in all_commits:
        try:
            commit = repo_obj.get_commit(commit_hash)
        except Exception:
            continue
        node_id = make_node_id(prefix, "commit", commit_hash)
        label = format_commit_label(commit, repo_obj, config)
        title = format_commit_tooltip(commit, repo_obj, config)

        # Determine the commit color based on how many branches it’s in
        commit_branches = repo_obj.commit_to_branches.get(commit_hash, [])
        if len(commit_branches) == 1:
            # Single-branch commit => color from that one branch
            node_color = branch_colors[commit_branches[0]]
        elif len(commit_branches) > 1:
            # Multi-branch commit => neutral color
            node_color = "#cccccc"
        else:
            # No branch found => fallback
            node_color = config["colors_map"].get("commit", "#bfbfbf")

        add_node(node_id, label, "commit", title=title, color=node_color, shape="box")

    # Create edges from commit to parent(s), one edge per shared branch color
    for commit_hash in all_commits:
        commit_branches = set(repo_obj.commit_to_branches.get(commit_hash, []))
        try:
            commit = repo_obj.get_commit(commit_hash)
        except Exception:
            continue
        node_id = make_node_id(prefix, "commit", commit_hash)

        for parent_hash in commit.parents:
            parent_id = make_node_id(prefix, "commit", parent_hash)
            parent_branches = set(repo_obj.commit_to_branches.get(parent_hash, []))
            shared_br = commit_branches.intersection(parent_branches)
            if not shared_br:
                # If no common branch, make a dashed black edge
                edges.append({
                    "from": node_id,
                    "to": parent_id,
                    "color": "#000000",
                    "dashes": True
                })
            else:
                # One edge per shared branch, each with that branch’s color
                for b in shared_br:
                    edges.append({
                        "from": node_id,
                        "to": parent_id,
                        "color": branch_colors[b]
                    })

    # Create branch "heads" as nodes and link them to their HEAD commit
    for branch_name, head_hash in repo_obj.branch_to_commit.items():
        node_id = make_node_id(prefix, "branch", branch_name)
        label = branch_name
        title = f"Branch: {branch_name}\nHEAD => {head_hash}"
        branch_color = branch_colors[branch_name]
        add_node(node_id, label, "branch", title=title, color=branch_color, shape="dot")

        head_node_id = make_node_id(prefix, "commit", head_hash)
        edges.append({
            "from": node_id,
            "to": head_node_id,
            "color": branch_color
        })

    return {"nodes": list(nodes.values()), "edges": edges}

import json, os, logging
from typing import Dict, List, Optional
import webbrowser

def generate_interactive_visualization(repo_obj,
                                       config: dict,
                                       prefix: str = "interactive",
                                       repo_label: str = "Repository",
                                       output_file: str = "git_interactive.html",
                                       view: bool = True) -> None:
    """
    Generate an interactive HTML file using vis-network with:
      - hierarchical layout,
      - physics disabled after initial layout so nodes won't drift,
      - multi-branch color edges,
      - ability to drag nodes,
      - minimal overlap,
      - right-click context menu on commits (highlight, collapse, uncollapse),
      - dark mode toggle.
    """
    try:
        # 1) Build the minimal node/edge data
        graph_data = build_graph_data(repo_obj, config, prefix)

        # 2) We'll also build a simple adjacency list (for descendants) so we can highlight/collapse
        #    For simplicity, we consider edges from parent -> child as "descendant" edges.
        #    If you want ancestors, you’d invert this.
        adjacency_map = {}
        for edge in graph_data["edges"]:
            src = edge["to"]     # parent
            dst = edge["from"]   # child
            adjacency_map.setdefault(src, []).append(dst)

        # 3) Choose layout direction (vertical or horizontal)
        direction = "UD" if config.get("layout_orientation", "vertical") == "vertical" else "LR"

        # 4) The HTML template with added right-click context menu + dark mode toggle
        html_template = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Git Graph Interactive Visualization</title>
  <!-- Load vis-network from a CDN -->
  <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style type="text/css">
    html, body {{
      margin: 0; padding: 0;
      width: 100%; height: 100%;
      font-family: sans-serif;
      overflow: hidden;
    }}
    #network {{
      width: 100%; height: 100%;
      border: 1px solid lightgray;
      position: relative;
    }}

    /* Context menu styling */
    #contextMenu {{
      position: absolute;
      z-index: 9999;
      display: none;
      background: #f8f8f8;
      border: 1px solid #ccc;
      min-width: 150px;
      box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
    }}
    #contextMenu ul {{
      list-style-type: none;
      margin: 0; padding: 0;
    }}
    #contextMenu li {{
      padding: 8px 12px;
      cursor: pointer;
    }}
    #contextMenu li:hover {{
      background: #e0e0e0;
    }}

    /* Dark mode support: toggled by adding 'dark-mode' to <body> */
    body.dark-mode {{
      background-color: #222;
      color: #eee;
    }}
    body.dark-mode #network {{
      border-color: #444;
    }}
    body.dark-mode #contextMenu {{
      background: #333;
      color: #eee;
      border-color: #666;
    }}
    body.dark-mode #contextMenu li:hover {{
      background: #444;
    }}
  </style>
</head>
<body>
  <button id="darkModeToggle" style="position:absolute; top:8px; left:8px; z-index:9999;">
    Toggle Dark Mode
  </button>
  <div id="network"></div>

  <!-- Our custom context menu -->
  <div id="contextMenu">
    <ul>
      <li id="highlightTreeBtn">Highlight Tree</li>
      <li id="collapseOthersBtn">Collapse All Others</li>
      <li id="uncollapseBtn">Uncollapse All</li>
    </ul>
  </div>

  <script type="text/javascript">
    // --- Data from Python:
    const graphData = {json.dumps(graph_data)};
    const adjacency = {json.dumps(adjacency_map)};

    // Vis-network init
    const container = document.getElementById('network');
    const options = {{
      layout: {{
        hierarchical: {{
          enabled: true,
          direction: '{direction}',
          sortMethod: 'directed',
          nodeSpacing: 150,
          levelSeparation: 150
        }}
      }},
      physics: {{
        enabled: false  // after initial layout, no drifting
      }},
      interaction: {{
        dragNodes: true,
        hover: true,
        tooltipDelay: 200,
        hideEdgesOnDrag: false
      }},
      nodes: {{
        shape: 'dot',
        size: 16,
        font: {{
          size: 12,
          color: '#000'
        }}
      }},
      edges: {{
        smooth: {{
          type: 'cubicBezier',
          roundness: 0.3
        }},
        width: 2,
        color: '#333'
      }}
    }};

    const network = new vis.Network(container, graphData, options);
    network.fit();

    // For convenience, keep track of hidden nodes so we can uncollapse later
    let hiddenNodes = new Set();

    // For context menu logic:
    let currentNode = null;      // which node was right-clicked
    let clickPos = {{x: 0, y: 0}};

    const contextMenu = document.getElementById("contextMenu");
    const highlightBtn = document.getElementById("highlightTreeBtn");
    const collapseBtn = document.getElementById("collapseOthersBtn");
    const uncollapseBtn = document.getElementById("uncollapseBtn");

    // Right-click (contextmenu) event on the network:
    network.on("oncontext", function(params) {{
      params.event.preventDefault();

      // Hide context menu by default
      contextMenu.style.display = "none";

      // Get the pointer/touch position
      clickPos.x = params.pointer.DOM.x;
      clickPos.y = params.pointer.DOM.y;

      // Check if a node is right-clicked
      const nodeIds = network.getNodeAt({{x: clickPos.x, y: clickPos.y}});
      if (nodeIds) {{
        currentNode = nodeIds;
        // Show the menu at the cursor
        contextMenu.style.left = params.event.clientX + "px";
        contextMenu.style.top = params.event.clientY + "px";
        contextMenu.style.display = "block";
      }} else {{
        currentNode = null;
      }}
    }});

    // Hide the context menu if user left-clicks anywhere
    container.addEventListener("click", () => {{
      contextMenu.style.display = "none";
    }});

    // Utility: get subtree (descendants) starting at 'node'
    // You can invert it if you want ancestors, or do both if you want the entire lineage.
    function getSubtreeNodes(node) {{
      let result = new Set();
      let stack = [node];
      while(stack.length > 0) {{
        let n = stack.pop();
        if(!result.has(n)) {{
          result.add(n);
          if(adjacency[n]) {{
            adjacency[n].forEach(child => {{
              stack.push(child);
            }});
          }}
        }}
      }}
      return result;
    }}

    // 1) Highlight the subtree
    highlightBtn.onclick = function() {{
      contextMenu.style.display = "none";
      if(!currentNode) return;

      // We'll gather all subtree nodes and give them a special color
      let subtree = getSubtreeNodes(currentNode);

      // Reset all nodes to default first
      let allNodeIds = network.body.data.nodes.getIds();
      allNodeIds.forEach(id => {{
        network.body.data.nodes.update({{ id, color: undefined }});
      }});

      // Now highlight the subtree in, say, gold
      subtree.forEach(id => {{
        network.body.data.nodes.update({{ id, color: {{
          background: "gold", border: "#cc0"
        }} }});
      }});
    }}

    // 2) Collapse all others => hide nodes not in the subtree
    collapseBtn.onclick = function() {{
      contextMenu.style.display = "none";
      if(!currentNode) return;

      let subtree = getSubtreeNodes(currentNode);
      let allNodeIds = network.body.data.nodes.getIds();

      // We'll hide every node *not* in that subtree
      let updates = [];
      allNodeIds.forEach(id => {{
        if(!subtree.has(id)) {{
          hiddenNodes.add(id);
          updates.push({{ id, hidden: true }});
        }}
      }});
      if(updates.length > 0) {{
        network.body.data.nodes.update(updates);
      }}
    }}

    // 3) Uncollapse all => simply show every node again
    uncollapseBtn.onclick = function() {{
      contextMenu.style.display = "none";
      if(hiddenNodes.size === 0) return;

      let updates = [];
      hiddenNodes.forEach(id => {{
        updates.push({{ id, hidden: false }});
      }});
      network.body.data.nodes.update(updates);
      hiddenNodes.clear();
    }}

    // Dark mode toggle
    const darkModeToggle = document.getElementById("darkModeToggle");
    darkModeToggle.onclick = function() {{
      document.body.classList.toggle("dark-mode");

      // Adjust default font color for nodes if in dark mode
      let isDark = document.body.classList.contains("dark-mode");
      let newFontColor = isDark ? "#eee" : "#000";
      let newEdgeColor = isDark ? "#aaa" : "#333";

      // Update all node fonts
      let allNodeIds = network.body.data.nodes.getIds();
      allNodeIds.forEach(id => {{
        // We only update the 'font.color'; you might also want
        // to adjust background color if it was white/black.
        network.body.data.nodes.update({{ id, font: {{ color: newFontColor }} }});
      }});

      // Update all edges color
      let allEdges = network.body.data.edges.get();
      allEdges.forEach(edge => {{
        edge.color = newEdgeColor;
      }});
      network.body.data.edges.update(allEdges);
    }};
  </script>
</body>
</html>
"""

        # 5) Write out the file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)

        logging.info("Interactive visualization (with context menu & dark mode) written to '%s'", output_file)
        if view:
            webbrowser.open('file://' + os.path.realpath(output_file))

    except Exception as e:
        logging.error("Error generating interactive visualization: %s", e)
        import sys
        sys.exit(1)
# ---------------------- STATIC GRAPH (PDF) ENGINE: NETWORKX -------------------------

def generate_static_pdf_networkx(repo_obj, config: dict, output_file: str = "git_static.pdf") -> None:
    """
    Minimal demonstration of using the same commit/branch graph data in NetworkX
    and saving as a PDF (requires matplotlib).
    """
    try:
        data = build_graph_data(repo_obj, config, prefix="static")
        import networkx as nx
        G = nx.DiGraph()
        for node in data["nodes"]:
            G.add_node(node["id"], label=node["label"], color=node.get("color", "lightblue"))
        for edge in data["edges"]:
            G.add_edge(edge["from"], edge["to"], color=edge.get("color", "gray"), dashes=edge.get("dashes", False))

        # Try Graphviz layout if available, fallback to spring layout
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except Exception:
            pos = nx.spring_layout(G, seed=42)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))

        # Color nodes
        node_colors = [G.nodes[n].get("color", "lightblue") for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)

        # Node labels
        labels = {n: G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

        # Edges with color / dashed
        edges_normal = []
        edges_dashed = []
        edges_colors_normal = []
        edges_colors_dashed = []
        for (u, v, d) in G.edges(data=True):
            if d.get("dashes"):
                edges_dashed.append((u, v))
                edges_colors_dashed.append(d.get("color", "gray"))
            else:
                edges_normal.append((u, v))
                edges_colors_normal.append(d.get("color", "gray"))

        nx.draw_networkx_edges(G, pos, edgelist=edges_normal,
                               edge_color=edges_colors_normal, arrows=True,
                               arrowstyle="->", arrowsize=10, width=2)
        nx.draw_networkx_edges(G, pos, edgelist=edges_dashed,
                               edge_color=edges_colors_dashed, style='dashed',
                               arrows=True, arrowstyle="->", arrowsize=10, width=2)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_file, format="pdf")
        plt.close()
        logging.info("Static PDF generated (NetworkX engine) saved to '%s'", output_file)
    except Exception as e:
        logging.error("Error generating static PDF with NetworkX: %s", e)
        sys.exit(1)

# ---------------------- STATIC GRAPH (PDF) ENGINE: GRAPHVIZ -------------------------

class GraphGenerator:
    def __init__(self, config: dict):
        self.config = config

    def generate_graph(self, repo_obj) -> None:
        """
        Produce a static PDF using Graphviz, forcing a hierarchical layout
        and no overlap. Branch/commit coloring is included.
        """
        logging.info("Generating static Graphviz PDF for repository.")
        master = Digraph(comment='Git Graph', format='pdf')
        rankdir = 'TB' if self.config.get("layout_orientation", "vertical") == "vertical" else "LR"
        master.attr(compound='true', splines='true', overlap='false', rankdir=rankdir)

        graph_data = build_graph_data(repo_obj, self.config, prefix="graphviz")

        # Add nodes with coloring
        for node in graph_data["nodes"]:
            master.node(
                node["id"],
                label=node["label"],
                tooltip=node.get("title", ""),
                shape=node.get("shape", "box"),
                style="filled",
                fillcolor=node.get("color", "lightblue"),
            )

        # Add edges
        for edge in graph_data["edges"]:
            attrs = {}
            if edge.get("dashes"):
                attrs["style"] = "dashed"
            if "color" in edge:
                attrs["color"] = edge["color"]
            master.edge(edge["from"], edge["to"], **attrs)

        output_name = 'git_graph'
        try:
            master.render(output_name, view=(not self.config['output_only']))
            logging.info("Graphviz PDF rendered to '%s.pdf'", output_name)
        except Exception as e:
            logging.error("Error rendering Graphviz PDF: %s", e)
            sys.exit(1)

# ---------------------- COMBINED REPO MODE -------------------------

def generate_combined_repo_graph(repo_base_path: str, config: dict) -> None:
    """
    Example stub for scanning multiple repos in a directory tree
    and generating a combined output or separate subgraphs.
    (Keeps minimal usage for demonstration.)
    """
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
    # Example of building a big combined diagram. 
    # Here, we simply iterate each found repo and generate a separate PDF:
    for idx, rdir in enumerate(found_repos):
        relp = os.path.relpath(rdir, abs_base)
        logging.info("Processing repository %d: %s (relative: %s)", idx, rdir, relp)
        try:
            repo_obj = GitRepo(rdir, local_only=config['local_only'])
            repo_obj.parse_dot_git_dir()
        except Exception as e:
            logging.error("Error loading repository '%s': %s", rdir, e)
            continue
        out_name = f"combined_repo_{idx}.pdf"
        if config.get("pdf_engine", "networkx") == "networkx":
            generate_static_pdf_networkx(repo_obj, config, output_file=out_name)
        else:
            GraphGenerator(config).generate_graph(repo_obj)

# ---------------------- CLI & MAIN -------------------------

def check_dependencies() -> None:
    if not shutil.which('dot'):
        logging.warning('Graphviz "dot" command not found. Legacy PDF engine may fail.')

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
        description="Generate a Git graph visualization (static PDF or interactive HTML) from a repository or base folder.\nAlso supports dump/read-dump modes.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('path', nargs='?', default=os.getcwd(), help='Path to the Git repository (default: current directory)')
    parser.add_argument('--colors', action='store_true', help='Enable distinct colors for nodes plus a legend.')
    parser.add_argument('-o', '--output-only', action='store_true', help='Save output file without automatically opening it.')
    parser.add_argument('--local', action='store_true', help='Show only local branches (skip remotes).')
    parser.add_argument('-P', '--predecessors', type=int, default=None, help='(Unused in this minimal example).')
    parser.add_argument('-S', '--successors', type=int, default=None, help='(Unused in this minimal example).')
    parser.add_argument('--dotted', nargs='*', choices=['remote','local','all'], default=[], help='(Unused in minimal example).')
    parser.add_argument('-M', '--metadata', type=str, default='all', help='Comma-separated metadata keys for commits (author,flags,gitstat). Default: all')
    parser.add_argument('--group', action='store_true', help='(Unused in minimal example).')
    parser.add_argument('--group-types', type=str, default='blob,tree', help='(Unused in minimal example).')
    parser.add_argument('--group-condense-level', type=int, default=1, help='(Unused in minimal example).')
    parser.add_argument('--layout-orientation', type=str, choices=['vertical','horizontal'], default='vertical', help='Orientation for interactive and static layouts (default: vertical).')
    parser.add_argument('--repo-base', type=str, help='If set, scan for multiple repos within this base folder for combined diagrams.')
    parser.add_argument('--max-depth', type=int, default=10, help='Maximum directory depth in repo-base mode (demo usage).')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dump', action='store_true', help='Dump repository data to JSON instead of visualizing.')
    group.add_argument('--read-dump', type=str, help='Read repository data from JSON dump and visualize.')
    parser.add_argument('--dump-file', type=str, default='git_repo_dump.json', help='Filename for dump output/input (default: git_repo_dump.json).')
    parser.add_argument('--interactive', action='store_true', help='Generate an interactive HTML visualization instead of static PDF.')
    parser.add_argument('--pdf-engine', type=str, choices=['networkx','graphviz'], default='networkx', help='PDF engine for static visualization (default: networkx).')
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
        'layout_orientation': args.layout_orientation,
        'colors_map': {
            'blob': 'lightyellow',
            'tree': 'lightgreen',
            'commit': 'lightblue',
            'branch': 'orange'
        },
        'pdf_engine': args.pdf_engine
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
                generate_interactive_visualization(dumped_repo, args['config'],
                                                  output_file="git_interactive.html",
                                                  view=(not args['config']['output_only']))
            else:
                if args['config'].get("pdf_engine", "networkx") == "networkx":
                    generate_static_pdf_networkx(dumped_repo, args['config'], output_file="git_static.pdf")
                else:
                    GraphGenerator(args['config']).generate_graph(dumped_repo)

        elif args['repo_base']:
            generate_combined_repo_graph(args['repo_base'], args['config'])

        else:
            repo_path = get_git_repo_path(args['repo_path'])
            repo = GitRepo(repo_path, local_only=args['config']['local_only'])
            repo.parse_dot_git_dir()

            if args['interactive']:
                generate_interactive_visualization(repo, args['config'],
                                                  output_file="git_interactive.html",
                                                  view=(not args['config']['output_only']))
            else:
                if args['config'].get("pdf_engine", "networkx") == "networkx":
                    generate_static_pdf_networkx(repo, args['config'], output_file="git_static.pdf")
                else:
                    GraphGenerator(args['config']).generate_graph(repo)

    except Exception as e:
        logging.exception("Unhandled error: %s", e)
        sys.exit(1)

if __name__ == '__main__':
    main()
