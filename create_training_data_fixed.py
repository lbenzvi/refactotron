#!/usr/bin/env python3
"""
Fixed degradation script that produces SYNTACTICALLY VALID Python code.
Uses proper AST transformations instead of regex hacks.
"""
import json
import random
import ast
import re
from typing import Tuple, List

class SafeCodeDegradation:
    """
    Degradation that GUARANTEES syntactically valid Python output.
    """

    def __init__(self, seed=42):
        random.seed(seed)

    def degrade_code(self, code: str) -> Tuple[str, bool]:
        """
        Apply safe degradations that maintain syntax validity.
        Returns: (degraded_code, was_changed)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # If input is already invalid, return as-is
            return code, False

        changed = False

        # Apply AST-based degradations
        tree, removed_docstrings = self.remove_docstrings(tree)
        tree, removed_type_hints = self.remove_type_hints(tree)
        tree, renamed_vars = self.rename_variables(tree)

        changed = removed_docstrings or removed_type_hints or renamed_vars

        # Convert back to code
        try:
            degraded = ast.unparse(tree)
        except:
            # If unparsing fails, return original
            return code, False

        # Apply safe string-based degradations
        degraded = self.degrade_formatting(degraded)

        # Final validation - ensure output is valid Python
        try:
            ast.parse(degraded)
            return degraded, changed
        except SyntaxError:
            # If our degradation broke syntax, return original
            print("⚠️ Degradation broke syntax, returning original")
            return code, False

    def remove_docstrings(self, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """Remove function docstrings while preserving syntax."""
        changed = False

        class DocstringRemover(ast.NodeTransformer):
            def __init__(self):
                self.changed = False

            def visit_FunctionDef(self, node):
                # Remove docstring if present
                if (node.body and
                    isinstance(node.body[0], ast.Expr) and
                    isinstance(node.body[0].value, ast.Constant) and
                    isinstance(node.body[0].value.value, str)):
                    node.body.pop(0)
                    self.changed = True
                    # Ensure function isn't empty
                    if not node.body:
                        node.body = [ast.Pass()]

                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node):
                return self.visit_FunctionDef(node)

        remover = DocstringRemover()
        tree = remover.visit(tree)
        return tree, remover.changed

    def remove_type_hints(self, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """Remove type hints using AST (not regex!)."""
        changed = False

        class TypeHintRemover(ast.NodeTransformer):
            def __init__(self):
                self.changed = False

            def visit_FunctionDef(self, node):
                # Remove return type annotation
                if node.returns is not None:
                    node.returns = None
                    self.changed = True

                # Remove parameter type annotations
                for arg in node.args.args:
                    if arg.annotation is not None:
                        arg.annotation = None
                        self.changed = True

                # Handle kwonly args
                for arg in node.args.kwonlyargs:
                    if arg.annotation is not None:
                        arg.annotation = None
                        self.changed = True

                # Handle vararg and kwarg
                if node.args.vararg and node.args.vararg.annotation:
                    node.args.vararg.annotation = None
                    self.changed = True

                if node.args.kwarg and node.args.kwarg.annotation:
                    node.args.kwarg.annotation = None
                    self.changed = True

                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node):
                return self.visit_FunctionDef(node)

            def visit_AnnAssign(self, node):
                """Convert annotated assignment to regular assignment."""
                self.changed = True
                # Convert: x: int = 5  →  x = 5
                if node.value is not None:
                    return ast.Assign(
                        targets=[node.target],
                        value=node.value,
                        lineno=node.lineno,
                        col_offset=node.col_offset
                    )
                else:
                    # No value, just remove the statement
                    return None

        remover = TypeHintRemover()
        tree = remover.visit(tree)
        return tree, remover.changed

    def rename_variables(self, tree: ast.AST) -> Tuple[ast.AST, bool]:
        """Rename variables to generic names."""
        changed = False

        class VariableRenamer(ast.NodeTransformer):
            def __init__(self):
                self.name_map = {}
                self.counter = 0
                self.changed = False
                # Don't rename these
                self.protected = {'self', 'cls', 'True', 'False', 'None'}

            def get_new_name(self, old_name):
                """Generate a generic name."""
                if old_name in self.name_map:
                    return self.name_map[old_name]

                if old_name in self.protected:
                    return old_name

                # Use generic names
                new_name = f"var{self.counter}"
                self.name_map[old_name] = new_name
                self.counter += 1
                self.changed = True
                return new_name

            def visit_FunctionDef(self, node):
                # Don't rename function names (too aggressive)
                # Only rename parameters and local variables

                # Rename function arguments
                for arg in node.args.args:
                    if arg.arg not in self.protected:
                        arg.arg = self.get_new_name(arg.arg)

                self.generic_visit(node)
                return node

            def visit_Name(self, node):
                # Only rename stores (variable assignments)
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self.protected:
                        node.id = self.get_new_name(node.id)
                # Rename loads if we've already mapped this name
                elif isinstance(node.ctx, ast.Load):
                    if node.id in self.name_map:
                        node.id = self.name_map[node.id]

                return node

        renamer = VariableRenamer()
        tree = renamer.visit(tree)
        return tree, renamer.changed

    def degrade_formatting(self, code: str) -> str:
        """Apply safe formatting degradations (whitespace only)."""
        lines = code.splitlines()
        new_lines = []

        for line in lines:
            # Randomly remove spaces around operators (but keep valid syntax)
            if random.random() > 0.7:
                # Remove spaces around = in assignments
                line = re.sub(r'\s+=\s+', '=', line)

            if random.random() > 0.7:
                # Remove spaces after commas
                line = re.sub(r',\s+', ',', line)

            new_lines.append(line)

        return '\n'.join(new_lines)


def main():
    """Generate training data with safe degradations."""

    print("🔧 Fixed Code Degradation Script")
    print("=" * 60)

    # Load clean functions
    print("\n📂 Loading clean functions...")
    with open('./data/clean_functions_optimized.jsonl', 'r') as f:
        functions = [json.loads(line) for line in f if line.strip()]

    print(f"✅ Loaded {len(functions)} functions")

    # Create degrader
    degrader = SafeCodeDegradation()

    # Create degraded pairs
    print("\n🔄 Creating degraded pairs...")
    pairs = []
    syntax_errors = 0

    for i, func_data in enumerate(functions):
        clean_code = func_data['code']
        degraded_code, was_changed = degrader.degrade_code(clean_code)

        # Verify degraded code is valid
        try:
            ast.parse(degraded_code)
        except SyntaxError:
            syntax_errors += 1
            print(f"⚠️ Syntax error in sample {i+1}, skipping")
            continue

        # Only include if code was actually changed
        if was_changed and degraded_code != clean_code:
            pair = {
                'input': f"### Refactor the following Python code to improve quality:\n\n{degraded_code}\n\n### Refactored code:",
                'output': clean_code
            }
            pairs.append(pair)

    print(f"✅ Created {len(pairs)} valid pairs")
    print(f"⚠️ Skipped {syntax_errors} pairs due to syntax errors")

    # Shuffle and split
    print("\n📊 Creating dataset splits...")
    random.seed(42)
    random.shuffle(pairs)

    n_train = int(len(pairs) * 0.8)
    n_val = int(len(pairs) * 0.1)

    splits = {
        'train': pairs[:n_train],
        'validation': pairs[n_train:n_train + n_val],
        'test': pairs[n_train + n_val:]
    }

    # Write files
    print("\n💾 Writing files...")
    for name, data in splits.items():
        filepath = f'./data/{name}.jsonl'
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

        print(f"  ✅ {name}.jsonl: {len(data)} samples")

    # Validation check - verify all examples are valid
    print("\n🔍 Validating generated data...")
    validation_errors = 0

    for name in ['train', 'validation', 'test']:
        filepath = f'./data/{name}.jsonl'
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                # Extract code from input
                input_code = data['input'].split('### Refactored code:')[0]
                input_code = input_code.split('### Refactor the following Python code to improve quality:')[1].strip()

                try:
                    ast.parse(input_code)
                except SyntaxError:
                    validation_errors += 1
                    if validation_errors <= 3:
                        print(f"  ⚠️ {name}[{i}]: Syntax error found")

    if validation_errors == 0:
        print("  ✅ All examples are syntactically valid!")
    else:
        print(f"  ⚠️ Found {validation_errors} syntax errors")

    print("\n" + "=" * 60)
    print("🎉 Done! Training data is ready.")
    print("\nNext steps:")
    print("  1. Upload train.jsonl, validation.jsonl, test.jsonl to Colab")
    print("  2. Run the optimized training script")
    print("  3. Expected validation loss: 0.45-0.50 (vs 0.68 previously)")


if __name__ == '__main__':
    main()
