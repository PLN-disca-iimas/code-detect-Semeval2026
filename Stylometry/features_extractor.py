import pandas as pd
import numpy as np
import re

def tokenizer(code):
    tokens_re = re.compile(r"[A-Za-z_]+|\d+|[^\w\s]")
    tokens = tokens_re.findall(code)
    return tokens

def length_features(code):
    lengths = []
    lines = code.split('\n')
    for line in lines:
        lengths.append(len(line))
    avg = float(np.mean(lengths)) if lengths else 0.0
    return avg

def indentation_features(code, tab_size=4):
    indents = []
    for line in code.splitlines():
        if line.strip() == "":
            continue
        expanded = line.replace("\t", " " * tab_size)
        spaces = len(expanded) - len(expanded.lstrip(" "))
        indents.append(spaces)
    if not indents:
        return 0.0, 0.0, 0
    indent_mean = float(np.mean(indents))
    indent_std = float(np.std(indents))
    indent_levels = len(set(indents))
    return indent_mean, indent_std, indent_levels

def comment_features(code):
    prefixes = ('#','//','/*','--',';')
    prefixes_in_line_re = re.compile(r'(\s?#|\s?//|\s?/\*|\s?--|\s?;)')
    lines = code.split('\n')
    total_comments = 0
    for line in lines:
        is_start = line.lstrip().startswith(prefixes)
        if is_start:
            total_comments += 1
        else:
            is_in_line = prefixes_in_line_re.search(line)
            if is_in_line:
                pos = is_in_line.end()
                tail = line[pos:].strip()
                if tail:
                    total_comments += 1
    comment_density = total_comments / len(lines) if lines else 0
    return total_comments, comment_density

def extract_all_features(code):
    tokens = tokenizer(code)
    unique_tokens = len(set(tokens))
    total_tokens = len(tokens)
    if total_tokens == 0:
        return (float('nan'),) * 10
    token_ratio = unique_tokens / total_tokens
    avg_length_line = length_features(code)
    ident_mean, ident_std, ident_levels = indentation_features(code)
    code_length = len(code)
    total_comments, comments_density = comment_features(code)
    return (unique_tokens, total_tokens, token_ratio, avg_length_line, 
            ident_mean, ident_std, ident_levels, code_length, 
            total_comments, comments_density)