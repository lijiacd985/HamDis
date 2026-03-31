#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import io
import math
import gzip
import glob
import argparse
import random
import shutil
import subprocess
import warnings
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


# =========================
# 基础工具函数
# =========================

def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def infer_output_root(pat_dir, output_dir=None):
    if output_dir:
        safe_makedirs(output_dir)
        return os.path.abspath(output_dir)
    pat_dir = os.path.abspath(pat_dir)
    parent = os.path.dirname(pat_dir)
    return parent


def natural_chr_key(chr_name):
    x = str(chr_name).replace("chr", "")
    if x == "X":
        return 23
    if x == "Y":
        return 24
    try:
        return int(x)
    except ValueError:
        return 10**9


def process_chr_name(chr_name):
    chr_num = str(chr_name).replace('chr', '').upper()
    if chr_num.isdigit():
        try:
            int(chr_num)
            return chr_num
        except ValueError:
            return None
    elif chr_num in ['X', 'Y']:
        return chr_num
    else:
        return None


def is_file_empty(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) == 0


# =========================
# step1: 数据采样与合并
# =========================

def sample_pat(pat_dir, chr_list, batchsize, batchnum, seed=1234):
    """
    尽量对齐原脚本逻辑：
    - 只从 chr_list[0] 读一次 pat 名称
    - 每个 batch 生成一组样本名
    - 所有染色体共用同一组样本名
    返回:
      merge_files: [(chr_name, batch_idx, out_pat), ...]
      sampled_files_map: {(chr_name, batch_idx): [pat.gz paths]}
    """
    if not chr_list:
        return [], {}

    rng = random.Random(seed)
    chr0 = chr_list[0]
    chr0_dir = os.path.join(pat_dir, chr0)
    if not os.path.isdir(chr0_dir):
        raise FileNotFoundError(f"采样参考目录不存在: {chr0_dir}")

    pat_names = [x for x in os.listdir(chr0_dir) if x.endswith("pat.gz")]
    if len(pat_names) == 0:
        raise FileNotFoundError(f"{chr0_dir} 下没有 pat.gz 文件")

    sampled_samples = []
    for _ in range(batchnum):
        names = pat_names[:]
        rng.shuffle(names)
        sampled_samples_batch = names[:min(batchsize, len(names))]
        sampled_samples.append(sampled_samples_batch)

    merge_root = os.path.join(infer_output_root(pat_dir), "middle_res", "mergepats")
    safe_makedirs(merge_root)

    sampled_files_map = {}
    merge_files = []
    for chr_name in chr_list:
        for bi, sampled_batch in enumerate(sampled_samples):
            out_pat = os.path.join(merge_root, f"{chr_name}_merge_{bi}.pat")
            sampled_files = [os.path.join(pat_dir, chr_name, x) for x in sampled_batch]
            sampled_files_map[(chr_name, bi)] = sampled_files
            merge_files.append((chr_name, bi, out_pat))

    return merge_files, sampled_files_map


def merge_index_chr(args):
    """
    尽量对齐原脚本 merge 逻辑：
    - 如果对应 mecon 已存在且非空，则跳过 merge
    - 使用 pandas 读 pat.gz
    - 若原文件只有4列，则新增 source_file；否则改写第5列为 source_file
    """
    chr_name, batch_idx, out_pat, pat_files = args
    safe_makedirs(os.path.dirname(out_pat))
    meconfile = out_pat.replace("mergepats", "metrics").replace(".pat", "_MeConcord.txt")
    if os.path.exists(meconfile) and os.path.getsize(meconfile) > 0:
        print("meconcord exists, skip merge pats:", meconfile)
        return out_pat

    with open(out_pat, 'w'):
        pass

    for file_name in pat_files:
        if os.path.exists(file_name):
            with gzip.open(file_name, 'rt') as f:
                df = pd.read_csv(f, delim_whitespace=True, header=None)
                source_file = os.path.basename(file_name).split('.')[0]
                if len(df.columns) == 4:
                    df['source_file'] = source_file
                else:
                    df[4] = df[4].astype(str).str.split('.').str[0]
                df.to_csv(out_pat, mode='a', header=False, sep='\t', index=False)
        else:
            print(f"file {file_name} not exist")
            continue
    return out_pat


# =========================
# step2: 汉明距离指标计算
# =========================

def compress_bits_zero_one_nan_extended(matrix: np.ndarray):
    num_rows, num_cols = matrix.shape
    num_chunks = (num_cols + 63) // 64
    val_bits = np.zeros((num_rows, num_chunks), dtype=np.uint64)
    mask_bits = np.zeros((num_rows, num_chunks), dtype=np.uint64)

    for i in range(num_rows):
        row_data = matrix[i]
        for chunk_idx in range(num_chunks):
            start_col = chunk_idx * 64
            end_col = min((chunk_idx + 1) * 64, num_cols)
            vb = np.uint64(0)
            mb = np.uint64(0)
            for col in range(start_col, end_col):
                val = row_data[col]
                if not math.isnan(val):
                    shift_amount = col - start_col
                    shifted_val = 1 << shift_amount
                    mb |= np.uint64(shifted_val)
                    if val == 1:
                        vb |= np.uint64(shifted_val)
            val_bits[i, chunk_idx] = vb
            mask_bits[i, chunk_idx] = mb
    return val_bits, mask_bits


def popcount_u64(x) -> int:
    val = int(x)
    if val < 0:
        raise ValueError(f"popcount_u64: x不能是负数, x={val}")
    count = 0
    while val:
        val &= (val - 1)
        count += 1
    return count


def _gpu_popcount_rows_u64(arr_u64):
    """
    对 uint64 数组按行统计 bit 数（GPU）。
    输入 shape: (rows, chunks), 输出 shape: (rows,)
    """
    if arr_u64.ndim == 1:
        arr_u64 = arr_u64.reshape(1, -1)
    bits = cp.unpackbits(arr_u64.view(cp.uint8), axis=-1)
    return bits.sum(axis=-1, dtype=cp.int32)


def pairwise_hamming_ignore_nan_extended(val_bits: np.ndarray, mask_bits: np.ndarray, use_gpu=False):
    n, num_chunks = val_bits.shape
    distance_matrix = np.zeros((n, n), dtype=float)
    hamming_counts = 0
    reads_bin_idx = []

    if use_gpu and _CUPY_AVAILABLE:
        try:
            _ = cp.cuda.runtime.getDeviceCount()
            val_gpu = cp.asarray(val_bits, dtype=cp.uint64)
            mask_gpu = cp.asarray(mask_bits, dtype=cp.uint64)
            for i_ in range(n - 1):
                common_mask = cp.bitwise_and(mask_gpu[i_], mask_gpu[i_ + 1:])
                diff_bits = cp.bitwise_and(cp.bitwise_xor(val_gpu[i_], val_gpu[i_ + 1:]), common_mask)
                total_common = _gpu_popcount_rows_u64(common_mask).astype(cp.float32)
                total_diff = _gpu_popcount_rows_u64(diff_bits).astype(cp.float32)
                dist = cp.where(
                    total_common == 0,
                    cp.nan,
                    cp.where(total_diff == 0, 1e-10, total_diff / total_common),
                )
                dist_cpu = cp.asnumpy(dist)
                distance_matrix[i_, i_ + 1:] = dist_cpu
                distance_matrix[i_ + 1:, i_] = dist_cpu
                valid = ~np.isnan(dist_cpu)
                if np.any(valid):
                    valid_j = np.where(valid)[0] + i_ + 1
                    hamming_counts += int(valid_j.size)
                    reads_bin_idx.extend([i_] * int(valid_j.size))
                    reads_bin_idx.extend(valid_j.tolist())
            return distance_matrix, hamming_counts, reads_bin_idx
        except Exception as e:
            print(f"[WARN] GPU 计算失败，自动回退 CPU: {e}")

    for i_ in range(n):
        for j_ in range(i_ + 1, n):
            total_diff = 0
            total_common = 0
            for chunk_idx in range(num_chunks):
                common_mask = mask_bits[i_, chunk_idx] & mask_bits[j_, chunk_idx]
                if common_mask != 0:
                    diff_bits = (val_bits[i_, chunk_idx] ^ val_bits[j_, chunk_idx]) & common_mask
                    popc = popcount_u64(diff_bits)
                    total_diff += popc
                    total_common += popcount_u64(common_mask)

            if total_common == 0:
                dist = np.nan
            else:
                if total_diff == 0:
                    dist = 1e-10
                else:
                    dist = float(total_diff) / total_common

            distance_matrix[i_, j_] = dist
            distance_matrix[j_, i_] = dist
            if not np.isnan(dist):
                hamming_counts += 1
                reads_bin_idx.append(i_)
                reads_bin_idx.append(j_)

    return distance_matrix, hamming_counts, reads_bin_idx


def merge_reads(result_1, cell_ids):
    cell_ids = np.array(cell_ids)
    unique_cells = np.unique(cell_ids)
    merged_reads_list = []
    merged_cell_ids = []

    for cell in unique_cells:
        indices = np.where(cell_ids == cell)[0]
        if len(indices) <= 1:
            for idx in indices:
                merged_reads_list.append(result_1[idx])
                merged_cell_ids.append(cell)
            continue

        cell_reads = result_1[indices, :]
        n_reads = cell_reads.shape[0]
        processed = [False] * n_reads

        for i in range(n_reads):
            if processed[i]:
                continue
            current_group = [i]
            processed[i] = True

            while True:
                added_new = False
                for j in range(n_reads):
                    if processed[j]:
                        continue
                    has_overlap = False
                    for idx in current_group:
                        read_i = cell_reads[idx]
                        read_j = cell_reads[j]
                        valid_i = ~np.isnan(read_i)
                        valid_j = ~np.isnan(read_j)
                        overlap = np.logical_and(valid_i, valid_j)
                        if np.any(overlap):
                            has_overlap = True
                            break
                    if has_overlap:
                        current_group.append(j)
                        processed[j] = True
                        added_new = True
                if not added_new:
                    break

            if len(current_group) > 1:
                group_reads = cell_reads[current_group]
                n_sites = cell_reads.shape[1]
                consensus = np.full(n_sites, np.nan)
                for j in range(n_sites):
                    site_vals = group_reads[:, j]
                    valid_vals = site_vals[~np.isnan(site_vals)]
                    if valid_vals.size > 0:
                        count_0 = np.sum(valid_vals == 0)
                        count_1 = np.sum(valid_vals == 1)
                        if count_0 > count_1:
                            consensus[j] = 0
                        elif count_1 > count_0:
                            consensus[j] = 1
                        else:
                            consensus[j] = np.nan
                merged_reads_list.append(consensus)
                merged_cell_ids.append(cell)
            else:
                merged_reads_list.append(cell_reads[current_group[0]])
                merged_cell_ids.append(cell)

    merged_array = np.array(merged_reads_list)
    return merged_array, merged_cell_ids


def get_unique_cells(cell_id):
    unique_celllist = list(set(cell_id))
    count = len(unique_celllist)
    return unique_celllist, count


def pat_to_Hamming(merge_pat_file, cpg_file, region_file, out_file,
                   binsize=5000, basedon0=0, min_cpg=1, max_dis=2000,
                   use_gpu=False,
                   max_possible_read=200000):
    """
    这里尽量按用户提供的原版脚本逻辑实现，使 _MeConcord.txt 的列意义与输出形式一致：
    interval chrom start end ReadNum_b ReadNum_a CpGNum MeCpG TotalCpG DNAme HammingCount HammingSum Readcounts Cellnum MajNum SubNum
    """
    chr_name = os.path.basename(cpg_file).replace("CpG_", "").replace(".bed", "")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return out_file

    CG_bed = pd.read_csv(cpg_file, sep='\t', header=None)
    region = pd.read_csv(region_file, sep='\t', header=None)
    pat_data = pd.read_csv(merge_pat_file, sep='\t', header=None)

    if pat_data.shape[1] < 8:
        pat_data_sorted = pat_data.sort_values(by=[0, 1])
        pat_data_sorted[5] = pat_data_sorted[1] + pat_data_sorted[2].astype(str).str.len() - 1
        cpg_dict = dict(zip(CG_bed[2], CG_bed[1]))
        pat_data_sorted[6] = pat_data_sorted[1].map(cpg_dict)
        pat_data_sorted[7] = pat_data_sorted[5].map(cpg_dict) + 1
        pat_data = pat_data_sorted

    reads_chr = pat_data.sort_values(by=[6, 7], ascending=[True, True])
    bin_chrom = region.loc[region.iloc[:, 0] == chr_name, :].copy()
    bin_sort = bin_chrom.sort_values(by=[2], ascending=[True])

    with open(out_file, 'w') as outdata:
        outdata.write(
            'interval\tchrom\tstart\tend\tReadNum_b\tReadNum_a\tCpGNum\tMeCpG\tTotalCpG\tDNAme\t'
            'HammingCount\tHammingSum\tReadcounts\tCellnum\tMajNum\tSubNum\n'
        )

    if bin_sort.shape[0] == 0:
        return out_file

    i = 0
    idx = 0
    split_num = int(math.ceil((bin_sort.iloc[i, 2] - bin_sort.iloc[i, 1] + 1) * 1.0 / binsize))
    if basedon0 == 1:
        interval_name = chr_name + '_' + str(bin_sort.iloc[i, 1] - 1) + '_' + str(bin_sort.iloc[i, 2])
    else:
        interval_name = chr_name + '_' + str(bin_sort.iloc[i, 1]) + '_' + str(bin_sort.iloc[i, 2])

    results = []
    bed_pos_array = CG_bed.iloc[:, 1].values
    reads_chr_shape0 = reads_chr.shape[0] - 1
    reads_chr_6 = reads_chr.iloc[:, 6].values
    reads_chr_7 = reads_chr.iloc[:, 7].values
    reads_chr_2 = reads_chr.iloc[:, 2].astype(str).tolist()
    reads_chr_1 = reads_chr.iloc[:, 1].values

    for j in range(0, split_num):
        pos1 = int(bin_sort.iloc[i, 1] + j * binsize)
        pos2 = int(min(bin_sort.iloc[i, 2], bin_sort.iloc[i, 1] + (j + 1) * binsize - 1))

        mask_wanted = (bed_pos_array >= pos1) & (bed_pos_array <= pos2)
        mask_help = (bed_pos_array >= pos1 - 2000) & (bed_pos_array <= pos2)

        cpg_pos_wanted = CG_bed[mask_wanted]
        cpg_pos_help = CG_bed[mask_help]
        cell_ids = []

        if cpg_pos_wanted.shape[0] >= min_cpg:
            bin_reads_me = np.zeros((max_possible_read, cpg_pos_wanted.shape[0]), dtype=int)
            bin_reads_unme = np.zeros((max_possible_read, cpg_pos_wanted.shape[0]), dtype=int)
            current_index = 0

            while idx > 0 and pos1 - reads_chr.iloc[idx, 7] < max_dis:
                idx -= 1
            while idx < (reads_chr.shape[0] - 1) and pos1 - reads_chr.iloc[idx, 7] > max_dis:
                idx += 1

            cpg_positions_wanted = cpg_pos_wanted.iloc[:, 1].values
            cpg_positions_help = cpg_pos_help.iloc[:, 1].values
            cpg_pos_wanted02 = cpg_pos_wanted.iloc[0, 2]

            while idx < reads_chr_shape0 and reads_chr_6[idx] < pos2:
                bin_reads_me_read = np.zeros((1, cpg_pos_wanted.shape[0]), int)
                bin_reads_unme_read = np.zeros((1, cpg_pos_wanted.shape[0]), int)
                if len(reads_chr_2[idx]) >= 1:
                    read_start = reads_chr_6[idx]
                    read_end = reads_chr_7[idx]

                    if np.sum((cpg_positions_wanted >= read_start) & (cpg_positions_wanted < read_end)) >= 1:
                        lost_num = np.sum((read_start <= cpg_positions_help) & (cpg_positions_wanted[0] > cpg_positions_help))
                        methy_reads = reads_chr_2[idx][lost_num:]
                        reads_CG = len(methy_reads)

                        reads_start_index = reads_chr_1[idx] + lost_num
                        bin_start_index = cpg_pos_wanted02

                        if lost_num > 0 and reads_CG <= cpg_pos_wanted.shape[0]:
                            bin_CG = np.zeros((1, cpg_pos_wanted.shape[0]), dtype=str)
                            bin_CG[0, :reads_CG] = list(methy_reads)
                            bin_CG[0, reads_CG:] = '0'
                        elif lost_num > 0 and reads_CG > cpg_pos_wanted.shape[0]:
                            bin_CG = np.zeros((1, cpg_pos_wanted.shape[0]), dtype=str)
                            bin_CG[0, :cpg_pos_wanted.shape[0]] = list(methy_reads[:cpg_pos_wanted.shape[0]])
                        else:
                            index_minus = reads_start_index - bin_start_index
                            bin_CG = np.zeros((1, cpg_pos_wanted.shape[0]), dtype=str)
                            available_space = cpg_pos_wanted.shape[0] - index_minus
                            if reads_CG <= available_space:
                                try:
                                    bin_CG[0, index_minus:index_minus + reads_CG] = list(methy_reads)
                                except Exception:
                                    idx += 1
                                    continue
                            else:
                                try:
                                    bin_CG[0, index_minus:index_minus + available_space] = list(methy_reads[:available_space])
                                except Exception:
                                    break

                        bin_reads_me_end = np.zeros((1, cpg_pos_wanted.shape[0]), int)
                        bin_reads_me_end[bin_CG == 'C'] = 1
                        bin_reads_unme_end = np.zeros((1, cpg_pos_wanted.shape[0]), int)
                        bin_reads_unme_end[bin_CG == 'T'] = 1
                        bin_reads_me_read += bin_reads_me_end
                        bin_reads_unme_read += bin_reads_unme_end

                if bin_reads_me_read.sum() + bin_reads_unme_read.sum() >= 2:
                    if current_index >= max_possible_read:
                        idx += 1
                        current_index += 1
                        continue
                    bin_reads_me[current_index] = bin_reads_me_read
                    bin_reads_unme[current_index] = bin_reads_unme_read
                    cell_ids.append(reads_chr.iloc[idx, 4])
                    current_index += 1
                idx += 1

            if current_index >= 1:
                bin_reads_me = bin_reads_me[:current_index, :]
                bin_reads_unme = bin_reads_unme[:current_index, :]
                bin_reads_total = bin_reads_me + bin_reads_unme
                methy_level = round(bin_reads_me.sum() * 1.0 / bin_reads_total.sum(), 3)
                total_me = bin_reads_me.sum()
                total_cpg = bin_reads_total.sum()
            else:
                bin_reads_me = np.zeros((0, cpg_pos_wanted.shape[0]), dtype=int)
                bin_reads_unme = np.zeros((0, cpg_pos_wanted.shape[0]), dtype=int)
                methy_level = np.nan
                total_me = 0
                total_cpg = 0

            if current_index >= 1:
                bin_reads_me = bin_reads_me.astype(float)
                bin_reads_unme = bin_reads_unme.astype(float)
                bin_reads_me[bin_reads_me == 0] = np.nan
                bin_reads_unme[bin_reads_unme == 0] = np.nan
                bin_reads_unme[bin_reads_unme == 1] = 0
                result = np.where(
                    np.isnan(bin_reads_me) & np.isnan(bin_reads_unme),
                    np.nan,
                    np.nansum([bin_reads_me, bin_reads_unme], axis=0)
                )

                result_1, celllist = merge_reads(result, cell_ids)
                unique_cells, cellcount = get_unique_cells(celllist)
                num_rows = result_1.shape[0]
                distance_matrix = np.full((num_rows, num_rows), np.nan)
                hamming_counts = 0
                reads_bin_idx = []

                val_bits, mask_bits = compress_bits_zero_one_nan_extended(result_1)
                distance_matrix, hamming_counts, reads_bin_idx = pairwise_hamming_ignore_nan_extended(
                    val_bits,
                    mask_bits,
                    use_gpu=use_gpu,
                )
                distance_sum = np.nansum(distance_matrix)
                read_counts = len(set(reads_bin_idx))
                cell_num = cellcount
            else:
                hamming_counts = 0
                distance_sum = 0
                read_counts = current_index
                cell_num = 0
                result_1 = bin_reads_me

            start_out = pos1 - 1 if basedon0 == 1 else pos1
            maj_num = 0
            sub_num = 0
            results.append(
                f"{interval_name}\t{chr_name}\t{start_out}\t{pos2}\t"
                f"{bin_reads_me.shape[0]}\t{result_1.shape[0]}\t{bin_reads_me.shape[1]}\t"
                f"{total_me}\t{total_cpg}\t{methy_level}\t"
                f"{hamming_counts}\t{distance_sum}\t{read_counts}\t"
                f"{cell_num}\t{maj_num}\t{sub_num}\n"
            )

        if len(results) >= 10000:
            with open(out_file, 'a') as f:
                f.writelines(results)
            results = []

    if len(results) > 0:
        with open(out_file, 'a') as f:
            f.writelines(results)

    print(f"hamming dist compute finished saved in {out_file}")
    return out_file


def compute_metric(args):
    merge_pat_file, cpg_file, region_file, metrics_dir, binsize, hm_params = args
    base = os.path.basename(merge_pat_file).replace(".pat", "")
    out_file = os.path.join(metrics_dir, f"{base}_MeConcord.txt")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return out_file
    return pat_to_Hamming(
        merge_pat_file,
        cpg_file,
        region_file,
        out_file,
        binsize=binsize,
        basedon0=hm_params['basedon0'],
        min_cpg=hm_params['min_cpg'],
        max_dis=hm_params['max_dis'],
        use_gpu=hm_params['use_gpu'],
        max_possible_read=hm_params['max_possible_read'],
    )


# =========================
# step3: 统计与区域筛选（内嵌原子脚本逻辑）
# =========================

def build_hmdistance_mean_matrix(metrics_dir, output_file, min_pairs=15):
    """
    对齐原脚本：
    - 跳过 header
    - 使用第11/12列（1-based）即 HammingCount / HammingSum
    - mean = col12 / col11
    - col11 < min_pairs 记为 NaN
    - 主键使用 interval/chrom/start/end，避免行数不一致时错位
    """
    files = sorted(glob.glob(os.path.join(metrics_dir, "*_MeConcord.txt")))
    if not files:
        pd.DataFrame(columns=["interval", "chrom", "start", "end"]).to_csv(output_file, sep='\t', index=False)
        return output_file

    merged = None
    valid_file_n = 0
    for fp in files:
        if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
            print(f"[WARN] 空文件，跳过: {fp}")
            continue
        try:
            df = pd.read_csv(fp, sep='\t')
        except pd.errors.EmptyDataError:
            print(f"[WARN] 无有效内容，跳过: {fp}")
            continue
        except Exception as e:
            print(f"[WARN] 读取失败，跳过: {fp} ({e})")
            continue
        need_cols = ['interval', 'chrom', 'start', 'end', 'HammingCount', 'HammingSum']
        if not set(need_cols).issubset(df.columns):
            print(f"[WARN] 文件缺少必要列，跳过: {fp}")
            continue

        sub = df[['interval', 'chrom', 'start', 'end', 'HammingCount', 'HammingSum']].copy()
        sub['HammingCount'] = pd.to_numeric(sub['HammingCount'], errors='coerce')
        sub['HammingSum'] = pd.to_numeric(sub['HammingSum'], errors='coerce')
        sub['meanHM'] = sub['HammingSum'] / sub['HammingCount']
        sub.loc[sub['HammingCount'] < min_pairs, 'meanHM'] = np.nan

        sample_name = os.path.basename(fp).replace('_MeConcord.txt', '')
        sub = sub[['interval', 'chrom', 'start', 'end', 'meanHM']].drop_duplicates(subset=['interval', 'chrom', 'start', 'end'])
        sub = sub.rename(columns={'meanHM': sample_name})

        if merged is None:
            merged = sub
        else:
            merged = merged.merge(sub, on=['interval', 'chrom', 'start', 'end'], how='outer')
        valid_file_n += 1

    if merged is None or valid_file_n == 0:
        print(f"[WARN] 目录下没有可用的 *_MeConcord.txt: {metrics_dir}")
        merged = pd.DataFrame(columns=['interval', 'chrom', 'start', 'end'])
    else:
        merged['chr_order'] = merged['chrom'].map(natural_chr_key)
        merged['start'] = pd.to_numeric(merged['start'], errors='coerce')
        merged['end'] = pd.to_numeric(merged['end'], errors='coerce')
        merged = merged.sort_values(['chr_order', 'start', 'end']).drop(columns=['chr_order'])

    merged.to_csv(output_file, sep='\t', index=False)
    return output_file


def stat_hmdistance_mean_matrix(mean_matrix_file, stat_file):
    df = pd.read_csv(mean_matrix_file, sep='\t')
    if df.shape[1] <= 4:
        out = df.iloc[:, :4].copy()
        out.columns = ['interval', 'chr', 'start', 'end']
        out['Mean'] = np.nan
        out['Sum'] = np.nan
        out['CV'] = np.nan
        out['Sum_CV'] = np.nan
        out['SD'] = np.nan
        out['ValidN'] = 0
        out.to_csv(stat_file, sep='\t', index=False)
        return stat_file

    val = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce')
    valid_ratio = val.notna().sum(axis=1) / max(val.shape[1], 1)
    mean_ = val.mean(axis=1, skipna=True)
    sum_ = val.sum(axis=1, skipna=True)
    sd_ = val.std(axis=1, skipna=True, ddof=1)
    cv_ = sd_ / mean_
    sum_cv_ = sum_ * cv_
    mask = valid_ratio > 0.8
    mean_[~mask] = np.nan
    sum_[~mask] = np.nan
    sd_[~mask] = np.nan
    cv_[~mask] = np.nan
    sum_cv_[~mask] = np.nan

    out = pd.DataFrame({
        'interval': df.iloc[:, 0],
        'chr': df.iloc[:, 1],
        'start': df.iloc[:, 2],
        'end': df.iloc[:, 3],
        'Mean': mean_,
        'Sum': sum_,
        'CV': cv_,
        'Sum_CV': sum_cv_,
        'SD': sd_,
        'ValidN': val.notna().sum(axis=1),
    })
    out.to_csv(stat_file, sep='\t', index=False)
    return stat_file


def get_stat_value(metrics_root, hm_dir, chr_list, min_pairs=15):
    safe_makedirs(hm_dir)
    all_stat_dfs = []
    for chr_name in chr_list:
        chr_metrics_dir = os.path.join(metrics_root, chr_name)
        if not os.path.isdir(chr_metrics_dir):
            print(f"[WARN] metrics 目录不存在，跳过: {chr_metrics_dir}")
            continue
        mean_matrix_file = os.path.join(hm_dir, f"hmdistance_mean_{chr_name}.txt")
        stat_file = os.path.join(hm_dir, f"hmdistance_stat_{chr_name}.tsv")
        build_hmdistance_mean_matrix(chr_metrics_dir, mean_matrix_file, min_pairs=min_pairs)
        try:
            stat_hmdistance_mean_matrix(mean_matrix_file, stat_file)
            sdf = pd.read_csv(stat_file, sep='\t')
            if sdf.shape[0] > 0:
                all_stat_dfs.append(sdf)
        except Exception as e:
            print(f"[WARN] 统计失败，跳过 {chr_name}: {e}")
            continue

    merged_stat = os.path.join(hm_dir, 'hmdistance_stat_allchr.tsv')
    if all_stat_dfs:
        pd.concat(all_stat_dfs, axis=0, ignore_index=True).to_csv(merged_stat, sep='\t', index=False)
    else:
        pd.DataFrame(columns=['interval', 'chr', 'start', 'end', 'Mean', 'Sum', 'CV', 'Sum_CV', 'SD', 'ValidN']).to_csv(merged_stat, sep='\t', index=False)
    return merged_stat


def extract_ref_cpg(interval_file, cpg_file_path, output_prefix, output_folder, chr_list):
    safe_makedirs(output_folder)
    interval_df = pd.read_csv(interval_file, sep='\t', header=None, names=['chr', 'start', 'end'])
    produced = []
    for chr_name in chr_list:
        sub = interval_df[interval_df['chr'] == chr_name].copy()
        if sub.empty:
            continue
        cpg_file = os.path.join(cpg_file_path, f"CpG_{chr_name}.bed")
        if not os.path.exists(cpg_file):
            print(f"[WARN] CpG 文件不存在，跳过: {cpg_file}")
            continue
        cpg_df = pd.read_csv(cpg_file, sep='\t', header=None)
        if cpg_df.shape[1] < 3:
            continue
        cpg_df = cpg_df.iloc[:, :3].copy()
        cpg_df.columns = ['chr', 'start', 'end']
        out_rows = []
        cpg_pos = cpg_df['start'].values
        for _, row in sub.iterrows():
            mask = (cpg_pos >= int(row['start'])) & (cpg_pos < int(row['end']))
            if mask.any():
                out_rows.append(cpg_df.loc[mask, ['chr', 'start', 'end']])
        if out_rows:
            out_df = pd.concat(out_rows, axis=0).drop_duplicates().sort_values(['chr', 'start', 'end'])
        else:
            out_df = pd.DataFrame(columns=['chr', 'start', 'end'])
        out_file = os.path.join(output_folder, f"{output_prefix}{chr_name}.txt")
        out_df.to_csv(out_file, sep='\t', header=False, index=False)
        produced.append(out_file)
    return produced


def run_commands_1(size, hm_dir, stat_filename, diff_result, cpg_file_dir, chr_list):
    safe_makedirs(diff_result)
    stat_file = os.path.join(hm_dir, stat_filename)
    if not os.path.exists(stat_file):
        raise FileNotFoundError(f"统计文件不存在: {stat_file}")
    df = pd.read_csv(stat_file, sep='\t')
    required = {'chr', 'start', 'end', 'SD'}
    if not required.issubset(df.columns):
        raise ValueError(f"统计文件缺少必要列 {required}，实际列: {list(df.columns)}")
    df = df.sort_values('SD', ascending=False).head(size).copy()
    df['chr_order'] = df['chr'].map(natural_chr_key)
    df = df.sort_values(['chr_order', 'start', 'end']).drop(columns=['chr_order'])
    interval_file = os.path.join(diff_result, f"SD_hmtop{size}.bed")
    df[['chr', 'start', 'end']].to_csv(interval_file, sep='\t', header=False, index=False)
    print(f"已生成 top{size} 区域的 BED 文件: {interval_file}")

    out_prefix = f"cpg_pos_hmtop{size}_"
    produced = extract_ref_cpg(interval_file, cpg_file_dir, out_prefix, diff_result, chr_list)
    cpg_pos = os.path.join(diff_result, f"cpg_pos_hmtop{size}_allchr.txt")
    with open(cpg_pos, 'w') as fout:
        for fp in sorted(produced, key=lambda x: natural_chr_key(re.search(r'(chr[^./]+)', os.path.basename(x)).group(1)) if re.search(r'(chr[^./]+)', os.path.basename(x)) else 10**9):
            if os.path.exists(fp):
                with open(fp) as fin:
                    shutil.copyfileobj(fin, fout)
    print(f"已合并所有染色体的 CpG 数据: {cpg_pos}")
    return interval_file, cpg_pos


# =========================
# step4: 数据格式转换与对齐
# =========================

def pat2bed(pat_gz_file, out_bed_file, genome=None):
    import glob
    safe_makedirs(os.path.dirname(out_bed_file))
    tmp_dir = out_bed_file + ".tmpdir"
    safe_makedirs(tmp_dir)
    cmd1 = f"wgbstools pat2beta -o {tmp_dir}"
    if genome:
        cmd1 += f" --genome {genome}"
    cmd1 += f" {pat_gz_file}"
    subprocess.run(cmd1, shell=True, check=True)
    beta_candidates = sorted(glob.glob(os.path.join(tmp_dir, "*.beta")))
    if not beta_candidates:
        beta_candidates = sorted(glob.glob(os.path.join(tmp_dir, "*.beta.gz")))
    if not beta_candidates:
        raise FileNotFoundError(f"pat2beta 未在 {tmp_dir} 生成 beta 文件")
    beta_file = beta_candidates[0]
    cmd2 = f"wgbstools beta2bed {beta_file} --outpath {out_bed_file}"
    subprocess.run(cmd2, shell=True, check=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_bed_file


def merge_cell_chr_beds(chr_bed_files, merged_bed_file):
    safe_makedirs(os.path.dirname(merged_bed_file))
    dfs = []
    for fp in chr_bed_files:
        if (not os.path.exists(fp)) or os.path.getsize(fp) == 0:
            continue
        try:
            df = pd.read_csv(fp, sep='\t', header=None)
        except Exception as e:
            print(f"[WARN] 读取 chr bed 失败，跳过 {fp}: {e}")
            continue
        if df.shape[0] == 0 or df.shape[1] < 5:
            continue
        df = df.iloc[:, :5].copy()
        dfs.append(df)
    if not dfs:
        open(merged_bed_file, 'w').close()
        return merged_bed_file
    merged = pd.concat(dfs, axis=0, ignore_index=True).drop_duplicates()
    merged.columns = ['chr', 'start', 'end', 'meth', 'total']
    merged['start'] = pd.to_numeric(merged['start'], errors='coerce')
    merged['end'] = pd.to_numeric(merged['end'], errors='coerce')
    merged['chr_order'] = merged['chr'].map(natural_chr_key)
    merged = merged.sort_values(['chr_order', 'start', 'end']).drop(columns=['chr_order'])
    merged.to_csv(merged_bed_file, sep='\t', header=False, index=False)
    return merged_bed_file


def process_bed_file(bed_file, ref_cpg_file, out_file):
    ref = pd.read_csv(ref_cpg_file, sep='\t', header=None, names=['chr', 'start', 'end'])
    if os.path.getsize(bed_file) == 0:
        out = ref.copy()
        out['value'] = np.nan
        out.to_csv(out_file, sep='\t', header=False, index=False)
        return out_file
    bed = pd.read_csv(bed_file, sep='\t', header=None)
    if bed.shape[1] < 5:
        raise ValueError(f"bed 文件列数不足，至少需要5列: {bed_file}")
    bed = bed.iloc[:, :5].copy()
    bed.columns = ['chr', 'start', 'end', 'meth', 'total']
    bed['meth'] = pd.to_numeric(bed['meth'], errors='coerce')
    bed['total'] = pd.to_numeric(bed['total'], errors='coerce')
    bed['ratio'] = bed['meth'] / bed['total']
    bed['value'] = np.where(bed['ratio'] > 0.5, 1, 0)
    bed = bed[['chr', 'start', 'end', 'value']].drop_duplicates(subset=['chr', 'start', 'end'])
    out = ref.merge(bed, on=['chr', 'start', 'end'], how='left')
    out.to_csv(out_file, sep='\t', header=False, index=False)
    return out_file

# =========================
# 主流程函数
# =========================

def run_hamdist(
    pat_dir,
    output_dir,
    batch_size,
    batch_num,
    m_process,
    bin_num,
    region_dir,
    cpg_file_dir,
    genome,
    chr_list,
    binsize=5000,
    seed=1234,
    min_pairs=15,
    min_cpg=1,
    max_dis=2000,
    max_possible_read=200000,
    basedon0=0,
    use_gpu=False,
):
    pat_dir = os.path.abspath(pat_dir)
    output_root = infer_output_root(pat_dir, output_dir)

    merge_root = os.path.join(output_root, "middle_res", "mergepats")
    metrics_root = os.path.join(output_root, "middle_res", "metrics")
    diff_result = os.path.join(output_root, "middle_res", "diff_result")
    hm_dir = os.path.join(diff_result, f"hm_{genome}_{batch_size}-{batch_num}")
    bed_dir = os.path.join(output_root, "bed")
    bed_new_dir = os.path.join(output_root, "bed_new")

    for d in [merge_root, metrics_root, diff_result, hm_dir, bed_dir, bed_new_dir]:
        safe_makedirs(d)
    for chr_name in chr_list:
        safe_makedirs(os.path.join(metrics_root, chr_name))

    print("[1/4] 数据采样与合并...")
    merge_files, sampled_files_map = sample_pat(
        pat_dir=pat_dir,
        chr_list=chr_list,
        batchsize=batch_size,
        batchnum=batch_num,
        seed=seed,
    )

    merge_jobs = [
        (chr_name, bi, out_pat, sampled_files_map[(chr_name, bi)])
        for chr_name, bi, out_pat in merge_files
    ]

    with Pool(processes=m_process) as pool:
        merged_pat_files = pool.map(merge_index_chr, merge_jobs)

    print("[2/4] 计算每个 merge pat 的局部汉明距离指标...")
    hm_params = {
        "basedon0": basedon0,
        "min_cpg": min_cpg,
        "max_dis": max_dis,
        "use_gpu": use_gpu,
        "max_possible_read": max_possible_read,
    }

    metric_jobs = []
    for merge_pat_file in merged_pat_files:
        m = re.match(r"(chr[^_]+)_merge_\d+\.pat$", os.path.basename(merge_pat_file))
        if not m:
            continue
        chr_name = m.group(1)
        cpg_file = os.path.join(cpg_file_dir, f"CpG_{chr_name}.bed")
        region_file = os.path.join(region_dir, f"region_{chr_name}.bed")
        chr_metrics_dir = os.path.join(metrics_root, chr_name)
        metric_jobs.append(
            (merge_pat_file, cpg_file, region_file, chr_metrics_dir, binsize, hm_params)
        )

    with Pool(processes=m_process) as pool:
        pool.map(compute_metric, metric_jobs)

    print("[3/4] 统计并筛选 hmtop 区域...")
    stat_filename = os.path.basename(
        get_stat_value(metrics_root, hm_dir, chr_list, min_pairs=min_pairs)
    )

    interval_file, cpg_pos = run_commands_1(
        size=bin_num,
        hm_dir=hm_dir,
        stat_filename=stat_filename,
        diff_result=diff_result,
        cpg_file_dir=cpg_file_dir,
        chr_list=chr_list,
    )

    print("[4/4] pat 转 bed，按 cell 合并所有 chr，再与目标 CpG 对齐...")
    cell_to_chr_pat = defaultdict(list)

    for chr_name in chr_list:
        chr_dir = os.path.join(pat_dir, chr_name)
        if not os.path.isdir(chr_dir):
            continue
        for pat_gz in sorted(glob.glob(os.path.join(chr_dir, "*.pat.gz"))):
            cell_name = os.path.basename(pat_gz).replace(".pat.gz", "")
            cell_to_chr_pat[cell_name].append((chr_name, pat_gz))

    for cell_name, chr_pat_list in cell_to_chr_pat.items():
        cell_tmp_dir = os.path.join(bed_dir, "_chr_tmp", cell_name)
        safe_makedirs(cell_tmp_dir)

        chr_bed_files = []
        for chr_name, pat_gz in chr_pat_list:
            chr_bed_file = os.path.join(cell_tmp_dir, f"{cell_name}.{chr_name}.bed")

            if (not os.path.exists(chr_bed_file)) or os.path.getsize(chr_bed_file) == 0:
                try:
                    pat2bed(pat_gz, chr_bed_file, genome=genome)
                except subprocess.CalledProcessError as e:
                    print(f"[WARN] pat2bed 失败，跳过 {pat_gz}: {e}")
                    continue
                except Exception as e:
                    print(f"[WARN] pat2bed 异常，跳过 {pat_gz}: {e}")
                    continue

            if os.path.exists(chr_bed_file) and os.path.getsize(chr_bed_file) > 0:
                chr_bed_files.append(chr_bed_file)

        merged_bed_file = os.path.join(bed_dir, f"{cell_name}.bed")
        try:
            merge_cell_chr_beds(chr_bed_files, merged_bed_file)
        except Exception as e:
            print(f"[WARN] 合并 cell bed 失败，跳过 {cell_name}: {e}")
            continue

        bed_new_file = os.path.join(bed_new_dir, f"{cell_name}.bed")
        try:
            process_bed_file(merged_bed_file, cpg_pos, bed_new_file)
        except Exception as e:
            print(f"[WARN] process_bed_file 失败，跳过 {merged_bed_file}: {e}")

    print("\n处理完成。关键输出：")
    print(f"- top bins BED: {interval_file}")
    print(f"- 目标 CpG 列表: {cpg_pos}")
    print(f"- 对齐后的单细胞 bed 目录: {bed_new_dir}")


# =========================
# 命令行入口
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="简化版 2-hamDist.py：核心原理尽量对齐原脚本，输出路径沿用 v4"
    )
    parser.add_argument("-i", "--pat_dir", required=True, help="输入 pat 目录，结构需为 pat/chr1/*.pat.gz")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="每批采样细胞数")
    parser.add_argument("-bn", "--batch_num", type=int, required=True, help="采样批次数")
    parser.add_argument("-m", "--m_process", type=int, default=4, help="并行进程数")
    parser.add_argument("-n", "--bin_num", type=int, required=True, help="选取 top N bins")
    parser.add_argument("-r", "--region_dir", required=True, help="region 文件目录，如 region_chr1.bed 所在目录")
    parser.add_argument("-cp", "--cpg_file_dir", required=True, help="CpG 参考目录，如 CpG_chr1.bed 所在目录")
    parser.add_argument("-g", "--genome", required=True, help="基因组名，仅用于命名")
    parser.add_argument("-cl", "--chr_list", nargs="+", required=True, help="处理的染色体列表")
    parser.add_argument("-od", "--output_dir", default=None, help="输出根目录，可选；默认写到 pat 的上级目录")
    parser.add_argument("--binsize", type=int, default=5000, help="pat_to_Hamming 划分 bin 大小")
    parser.add_argument("--seed", type=int, default=1234, help="随机采样种子")
    parser.add_argument("--min_pairs", type=int, default=15, help="构建 hamming mean matrix 时要求的最小 HammingCount")
    parser.add_argument("--min_cpg", type=int, default=1, help="原始 pat_to_Hamming 的 min_cpg")
    parser.add_argument("--max_dis", type=int, default=2000, help="原始 pat_to_Hamming 的 max_dis")
    parser.add_argument("--max_possible_read", type=int, default=200000, help="原始 pat_to_Hamming 的 max_possible_read")
    parser.add_argument("--basedon0", type=int, default=0, choices=[0, 1], help="原始 pat_to_Hamming 的 basedon0")

    args = parser.parse_args()

    run_hamdist(
        pat_dir=args.pat_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        batch_num=args.batch_num,
        m_process=args.m_process,
        bin_num=args.bin_num,
        region_dir=args.region_dir,
        cpg_file_dir=args.cpg_file_dir,
        genome=args.genome,
        chr_list=args.chr_list,
        binsize=args.binsize,
        seed=args.seed,
        min_pairs=args.min_pairs,
        min_cpg=args.min_cpg,
        max_dis=args.max_dis,
        max_possible_read=args.max_possible_read,
        basedon0=args.basedon0,
    )


if __name__ == "__main__":
    main()
