#!/usr/bin/python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import argparse

def sync_cache_dir(src_cache_dir : str, dst_cache_dir : str):
    if not os.path.exists(src_cache_dir):
        return

    if not os.path.exists(dst_cache_dir):
        os.system(f"mkdir -p {dst_cache_dir}")

    compiler_version_list = os.listdir(src_cache_dir)
    for compiler_version in compiler_version_list:
        if compiler_version.find("neuronxcc") != 0:
            continue

        src_compiler_dir = os.path.join(src_cache_dir, compiler_version)
        dst_compiler_dir = os.path.join(dst_cache_dir, compiler_version)
        if not os.path.exists(dst_compiler_dir):
            os.makedirs(dst_compiler_dir)

        src_module_list = os.listdir(src_compiler_dir)
        dst_module_list = os.listdir(dst_compiler_dir)
        for module in src_module_list:
            if module.find("MODULE") != 0:
                continue

            if module not in dst_module_list:
                src_module_dir = os.path.join(src_compiler_dir, module)
                print(f"cp -r {src_module_dir} {dst_compiler_dir}")
                os.system(f"cp -r {src_module_dir} {dst_compiler_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog=sys.argv[0],
                        description='sync two neuron cache dir')
    parser.add_argument("src_dir", help="source dir")
    parser.add_argument("dst_dir", help="destibatuib dir")
    parser.add_argument("-b", "--bidirectional", action="store_true", help="whether to udpate source using destination", default=False)
    args = parser.parse_args()
    sync_cache_dir(args.src_dir, args.dst_dir)
    if args.bidirectional:
        sync_cache_dir(args.dst_dir, args.src_dir)
