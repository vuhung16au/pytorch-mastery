#!/usr/bin/env python3
"""
Validate that Python code follows the OOP and Helper Functions policies.

This script can be used to check if code in the repository follows
the GitHub Copilot instruction policies for:
1. OOP implementation preference
2. Helper functions usage
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any


class CodeStyleValidator(ast.NodeVisitor):
    """Validator to check OOP and helper functions usage."""
    
    def __init__(self):
        self.issues = []
        self.classes_found = 0
        self.functions_found = 0
        self.helper_functions = 0
        self.oop_patterns = 0
    
    def visit_ClassDef(self, node):
        """Check for OOP patterns in class definitions."""
        self.classes_found += 1
        
        # Check for docstring
        if not ast.get_docstring(node):
            self.issues.append(f"Class '{node.name}' lacks docstring (line {node.lineno})")
        
        # Check if inherits from appropriate base classes
        if any(base.id in ['nn.Module', 'Dataset', 'ABC'] for base in node.bases if hasattr(base, 'id')):
            self.oop_patterns += 1
        
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        """Check for helper functions and proper patterns."""
        self.functions_found += 1
        
        # Check for docstring
        if not ast.get_docstring(node):
            self.issues.append(f"Function '{node.name}' lacks docstring (line {node.lineno})")
        
        # Check for helper function indicators
        if (node.name.startswith('_') or 
            'helper' in node.name.lower() or 
            any(keyword in node.name.lower() for keyword in ['create', 'build', 'validate', 'format', 'detect'])):
            self.helper_functions += 1
        
        self.generic_visit(node)
    
    def get_report(self) -> Dict[str, Any]:
        """Generate validation report."""
        return {
            'classes_found': self.classes_found,
            'functions_found': self.functions_found,
            'helper_functions': self.helper_functions,
            'oop_patterns': self.oop_patterns,
            'issues': self.issues,
            'oop_ratio': self.oop_patterns / max(self.classes_found, 1),
            'helper_ratio': self.helper_functions / max(self.functions_found, 1)
        }


def validate_file(file_path: Path) -> Dict[str, Any]:
    """
    Validate a single Python file for OOP and helper function policies.
    
    Args:
        file_path: Path to Python file to validate
        
    Returns:
        Validation report dictionary
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        validator = CodeStyleValidator()
        validator.visit(tree)
        
        report = validator.get_report()
        report['file'] = str(file_path)
        report['valid'] = True
        
        return report
        
    except Exception as e:
        return {
            'file': str(file_path),
            'valid': False,
            'error': str(e),
            'classes_found': 0,
            'functions_found': 0,
            'helper_functions': 0,
            'oop_patterns': 0,
            'issues': [f"Failed to parse file: {e}"],
            'oop_ratio': 0,
            'helper_ratio': 0
        }


def validate_directory(directory: Path, pattern: str = "*.py") -> List[Dict[str, Any]]:
    """
    Validate all Python files in a directory.
    
    Args:
        directory: Directory to scan
        pattern: File pattern to match
        
    Returns:
        List of validation reports
    """
    reports = []
    
    for file_path in directory.rglob(pattern):
        if file_path.is_file():
            report = validate_file(file_path)
            reports.append(report)
    
    return reports


def print_summary(reports: List[Dict[str, Any]]) -> None:
    """Print validation summary."""
    total_files = len(reports)
    valid_files = sum(1 for r in reports if r['valid'])
    total_classes = sum(r['classes_found'] for r in reports)
    total_functions = sum(r['functions_found'] for r in reports)
    total_helpers = sum(r['helper_functions'] for r in reports)
    total_oop = sum(r['oop_patterns'] for r in reports)
    
    print(f"\n📊 VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Files analyzed: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Total classes: {total_classes}")
    print(f"Total functions: {total_functions}")
    print(f"Helper functions: {total_helpers} ({total_helpers/max(total_functions, 1):.1%})")
    print(f"OOP patterns: {total_oop} ({total_oop/max(total_classes, 1):.1%})")
    
    # Show files with issues
    files_with_issues = [r for r in reports if r['issues']]
    if files_with_issues:
        print(f"\n⚠️  FILES WITH ISSUES:")
        for report in files_with_issues:
            print(f"\n📄 {report['file']}:")
            for issue in report['issues'][:5]:  # Show first 5 issues
                print(f"   • {issue}")
            if len(report['issues']) > 5:
                print(f"   ... and {len(report['issues']) - 5} more issues")
    
    # Policy compliance check
    helper_compliance = total_helpers / max(total_functions, 1) >= 0.3  # 30% helper functions
    oop_compliance = total_oop / max(total_classes, 1) >= 0.5  # 50% OOP patterns
    
    print(f"\n🎯 POLICY COMPLIANCE:")
    print(f"Helper Functions Policy: {'✅ PASS' if helper_compliance else '❌ FAIL'}")
    print(f"OOP Implementation Policy: {'✅ PASS' if oop_compliance else '❌ FAIL'}")


def main():
    """Main validation function."""
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = Path(".")
    
    print(f"🔍 Validating OOP and Helper Functions policies in: {target.absolute()}")
    
    if target.is_file():
        reports = [validate_file(target)]
    else:
        reports = validate_directory(target)
    
    if not reports:
        print("❌ No Python files found to validate")
        return
    
    print_summary(reports)
    
    # Show examples of good practices
    good_examples = [r for r in reports if r['valid'] and r['oop_ratio'] > 0.5 and r['helper_ratio'] > 0.3]
    if good_examples:
        print(f"\n✨ EXAMPLES OF GOOD PRACTICES:")
        for report in good_examples[:3]:  # Show top 3
            print(f"   📄 {report['file']} - OOP: {report['oop_ratio']:.1%}, Helpers: {report['helper_ratio']:.1%}")


if __name__ == "__main__":
    main()