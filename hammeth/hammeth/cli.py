#!/usr/bin/env python3

import argparse

from hammeth.commands.prepare import run_prepare
from hammeth.commands.hamdist import run_hamdist
from hammeth.commands.matrix import run_matrix


def main():
    parser = argparse.ArgumentParser(prog="hammeth")
    subparsers = parser.add_subparsers(dest="command")

    # =========================
    # prepare
    # =========================
    parser_prepare = subparsers.add_parser("prepare", help="prepare PAT/cell inputs")
    parser_prepare.add_argument("-i", required=True, help="输入 BAM 目录或输入目录")
    parser_prepare.add_argument("-m", type=int, default=4, help="并行进程数")
    parser_prepare.add_argument("-r", required=True, help="region 文件目录")
    parser_prepare.add_argument("-cp", required=True, help="CpG 参考目录")
    parser_prepare.add_argument("-g", required=True, help="基因组名")
    parser_prepare.add_argument("-cl", nargs="+", required=True, help="处理染色体列表")
    parser_prepare.add_argument("-o", required=False, help="输出目录（可选）")
    parser_prepare.set_defaults(func=run_prepare)

    # =========================
    # hamdist
    # =========================
    parser_hamdist = subparsers.add_parser("hamdist", help="run Hamming-distance pipeline")
    parser_hamdist.add_argument("-i", required=True, help="输入 pat 目录")
    parser_hamdist.add_argument("-o", required=True, help="输出目录")
    parser_hamdist.add_argument("-bs", type=int, required=True, help="每批采样细胞数")
    parser_hamdist.add_argument("-bn", type=int, required=True, help="采样批次数")
    parser_hamdist.add_argument("-m", type=int, default=4, help="并行进程数")
    parser_hamdist.add_argument("-n", type=int, required=True, help="选取 top N bins")
    parser_hamdist.add_argument("-r", required=True, help="region 文件目录")
    parser_hamdist.add_argument("-cp", required=True, help="CpG 参考目录")
    parser_hamdist.add_argument("-g", required=True, help="基因组名")
    parser_hamdist.add_argument("-cl", nargs="+", required=True, help="处理染色体列表")

    parser_hamdist.add_argument("--binsize", type=int, default=5000, help="bin size")
    parser_hamdist.add_argument("--seed", type=int, default=1234, help="随机种子")
    parser_hamdist.add_argument("--min_pairs", type=int, default=15, help="最小 HammingCount")
    parser_hamdist.add_argument("--min_cpg", type=int, default=1, help="bin 内最少 CpG 数")
    parser_hamdist.add_argument("--max_dis", type=int, default=2000, help="read 回溯距离")
    parser_hamdist.add_argument("--max_possible_read", type=int, default=200000, help="每个 bin 最大 read 数")
    parser_hamdist.add_argument("--basedon0", type=int, default=0, help="是否转为 0-based")
    parser_hamdist.add_argument("--use_gpu", action="store_true", help="启用 GPU 加速（需安装 CuPy）")
    parser_hamdist.set_defaults(func=run_hamdist)

    # =========================
    # matrix
    # =========================
    parser_matrix = subparsers.add_parser("matrix", help="build ratio beds and Hamming distance matrix")
    parser_matrix.add_argument("-i", required=True, help="bed 文件目录（hamdist 生成的 bed 目录）")
    parser_matrix.add_argument("-r", "--region_bed", required=True,help="目标区域 BED，例如 middle_res/diff_result/SD_hmtop20000.bed")
    parser_matrix.add_argument("-t", "--tmp_dir", default="tmp",help="临时目录，默认 tmp")
    parser_matrix.add_argument(    "-ol", "--out_long", default="tutorial_hamming_long_v4.tsv",help="长表输出文件")
    parser_matrix.add_argument("-om", "--out_matrix", default="tutorial_hamming_distance_matrix_v4.tsv",help="矩阵输出文件")
    parser_matrix.add_argument("-mm", "--memmap_file", default="X_memmap.dat",help="memmap 临时文件")
    parser_matrix.add_argument("-p", "--nproc", type=int, default=20,help="并行进程数")
    parser_matrix.add_argument("-b", "--batch_size", type=int, default=5000,help="pair 分批大小")
    parser_matrix.add_argument("--overwrite", action="store_true",help="若输出已存在则覆盖重算")
    parser_matrix.add_argument("--keep_tmp", action="store_true",help="保留 tmp 和 memmap 文件")
    parser_matrix.set_defaults(func=run_matrix)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
