from hammeth.scripts.hamdist_core import run_hamdist as hamdist_core


def run_hamdist(args):
    hamdist_core(
        pat_dir=args.i,
        output_dir=args.o,
        batch_size=args.bs,
        batch_num=args.bn,
        m_process=args.m,
        bin_num=args.n,
        region_dir=args.r,
        cpg_file_dir=args.cp,
        genome=args.g,
        chr_list=args.cl,
        binsize=getattr(args, "binsize", 5000),
        seed=getattr(args, "seed", 1234),
        min_pairs=getattr(args, "min_pairs", 15),
        min_cpg=getattr(args, "min_cpg", 1),
        max_dis=getattr(args, "max_dis", 2000),
        max_possible_read=getattr(args, "max_possible_read", 200000),
        basedon0=getattr(args, "basedon0", 0),
        use_gpu=getattr(args, "use_gpu", False),
    )
