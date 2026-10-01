import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def import_std():
    import functools
    import pathlib
    import re
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    warnings.filterwarnings("ignore")
    return ThreadPoolExecutor, functools, pathlib, re


@app.cell
def import_pkg():
    import boto3
    import botocore
    from iterpop import iterpop as ip
    import marimo as mo
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats
    import seaborn as sns
    from teeplot import teeplot as tp
    from watermark import watermark

    from dishpylib.pyhelpers import fit_control_t_distns

    return (
        boto3,
        botocore,
        fit_control_t_distns,
        ip,
        mo,
        pd,
        scipy_stats,
        sns,
        tp,
        watermark,
    )


@app.cell(hide_code=True)
def do_watermark(mo, watermark):
    mo.md(
        f"""
    ```Text
    {watermark(
        current_date=True,
        iso8601=True,
        machine=True,
        updated=True,
        python=True,
        iversions=True,
        globals_=globals(),
    )}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def intro(mo):
    mo.md("""
    # Genotype Complexity Trajectories: Replays of Replicate 16005

    Genotype complexity is calculated on the fly (rather than through the full collation/processing system) as the number of sites at which a nop-out knockout (`variant-competitions`) is significantly deleterious (`Is Less Fit`), following the OEE4 binder notebooks (e.g., `2025-12-09-bb-complexity`).
    Each replay bucket is plotted in its own facet, with the trajectory of the original run (bucket `prq49`, series 16005) shown in gray for reference.
    """)
    return


@app.cell
def set_teeplot_subdir(pathlib):
    teeplot_subdir = pathlib.Path(__file__).stem
    return (teeplot_subdir,)


@app.cell
def define_buckets():
    original_bucket = "prq49"
    replay_buckets = [
        "prq49-replay-stint014-rep000",
        "prq49-replay-stint014-rep001",
        "prq49-replay-stint014-rep002",
        "prq49-replay-stint014-rep003",
        "prq49-replay-stint014-rep004",
        "prq49-replay-stint015-rep000",
        "prq49-replay-stint015-rep001",
        "prq49-replay-stint015-rep002",
        "prq49-replay-stint015-rep003",
        "prq49-replay-stint015-rep004",
        "prq49-replay-stint016-rep000",
        "prq49-replay-stint016-rep001",
        "prq49-replay-stint016-rep002",
        "prq49-replay-stint016-rep003",
        "prq49-replay-stint016-rep004",
        "prq49-replay-stint020-rep000",
        "prq49-replay-stint020-rep001",
        "prq49-replay-stint020-rep002",
        "prq49-replay-stint020-rep003",
        "prq49-replay-stint020-rep004",
        "prq49-replay-stint030-rep000",
        "prq49-replay-stint030-rep001",
        "prq49-replay-stint030-rep002",
        "prq49-replay-stint030-rep003",
        "prq49-replay-stint030-rep004",
    ]
    endeavor = 16
    series = 16005
    return endeavor, original_bucket, replay_buckets, series


@app.cell
def define_analysis(
    boto3,
    botocore,
    fit_control_t_distns,
    functools,
    ip,
    pd,
    re,
    scipy_stats,
):
    s3_handle = boto3.resource(
        "s3",
        region_name="us-east-2",
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
        ),
    )

    def list_stint_keys(bucket, prefix):
        # one collated file per stint
        bucket_handle = s3_handle.Bucket(bucket)
        return {
            int(re.search(r"/stint=(\d+)/", x.key).group(1)): x.key
            for x in bucket_handle.objects.filter(Prefix=prefix)
        }

    @functools.lru_cache
    def get_control_t_distns(bucket, control_key):
        control_df = pd.read_csv(f"s3://{bucket}/{control_key}")

        return fit_control_t_distns(
            control_df[control_df["Root ID"] == 1].copy()
        )

    def preprocess_competition_fitnesses(competitions_df, control_fits_df):
        # preprocess data
        @functools.lru_cache
        def h0_fit(series):
            return ip.popsingleton(
                control_fits_df[control_fits_df["Series"] == series].to_dict(
                    orient="records",
                )
            )

        competitions_df["p"] = competitions_df.apply(
            lambda row: scipy_stats.t.cdf(
                row["Fitness Differential"],
                h0_fit(row["genome series"])["Fit Degrees of Freedom"],
                loc=h0_fit(row["genome series"])["Fit Loc"],
                scale=h0_fit(row["genome series"])["Fit Scale"],
            ),
            axis=1,
        )
        competitions_df["Is Less Fit"] = competitions_df["p"] < 1.0 / 40
        competitions_df["Is More Fit"] = competitions_df["p"] > (1.0 - 1.0 / 40)
        competitions_df["Is Neutral"] = ~(
            competitions_df["Is Less Fit"] | competitions_df["Is More Fit"]
        )
        competitions_df["Relative Fitness"] = competitions_df.apply(
            lambda row: (
                "Significantly Advantageous"
                if row["Is More Fit"]
                else (
                    "Significantly Deleterious"
                    if row["Is Less Fit"]
                    else "Neutral"
                )
            ),
            axis=1,
        )

        return competitions_df

    return (
        get_control_t_distns,
        list_stint_keys,
        preprocess_competition_fitnesses,
    )


@app.cell(hide_code=True)
def delimit_get_data(mo):
    mo.md("""
    ## Get Data
    """)
    return


@app.cell
def get_data(
    ThreadPoolExecutor,
    endeavor,
    get_control_t_distns,
    list_stint_keys,
    original_bucket,
    pd,
    preprocess_competition_fitnesses,
    replay_buckets,
    series,
):
    def tabulate_stint(bucket, stint, variant_key, control_key):
        df = pd.read_csv(f"s3://{bucket}/{variant_key}", compression="xz")
        df = df[df["Competition Series"] == series]
        df = df[df["genome variation"] != "master"].copy()
        df = df.groupby("genome variation").mean(numeric_only=True).reset_index()
        control_fits_df = get_control_t_distns(bucket, control_key)
        df = preprocess_competition_fitnesses(df, control_fits_df)
        return {
            "bucket": bucket,
            "Stint": stint,
            "Genotype Complexity": df["Is Less Fit"].sum(),
            "Num Advantageous": df["Is More Fit"].sum(),
            "Num Neutral": df["Is Neutral"].sum(),
            "Num Sites Tested": len(df),
        }

    jobs = []
    for bucket in [original_bucket, *replay_buckets]:
        variant_keys = list_stint_keys(
            bucket,
            f"endeavor={endeavor}/variant-competitions/stage=3+what=collated/",
        )
        control_keys = list_stint_keys(
            bucket,
            f"endeavor={endeavor}/control-competitions/stage=2+what=collated/",
        )
        for stint, variant_key in sorted(variant_keys.items()):
            jobs.append((bucket, stint, variant_key, control_keys[stint]))

    with ThreadPoolExecutor(max_workers=16) as pool:
        df_complexity = pd.DataFrame(
            pool.map(lambda args: tabulate_stint(*args), jobs)
        )
    return (df_complexity,)


@app.cell
def prep_data(df_complexity, original_bucket):
    df_complexity["replay"] = df_complexity["bucket"].str.replace(
        "prq49-replay-", "", regex=False
    )
    df_complexity.loc[
        df_complexity["bucket"] == original_bucket, "replay"
    ] = "original"

    df_original = df_complexity[
        df_complexity["bucket"] == original_bucket
    ].copy()
    df_replays = df_complexity[
        df_complexity["bucket"] != original_bucket
    ].copy()
    df_replays["replay start"] = df_replays["replay"].str.extract(
        r"(stint\d+)"
    )
    df_replays["replicate"] = df_replays["replay"].str.extract(r"(rep\d+)")
    return df_original, df_replays


@app.cell
def peek_data(df_original, df_replays, pd):
    pd.concat([df_original.head(), df_replays.head(), df_replays.tail()])
    return


@app.cell(hide_code=True)
def delimit_original(mo):
    mo.md("""
    ## Original Trajectory (bucket `prq49`, series 16005)
    """)
    return


@app.cell
def plot_original(df_original, sns, teeplot_subdir, tp):
    with tp.teed(
        sns.relplot,
        data=df_original,
        x="Stint",
        y="Genotype Complexity",
        kind="line",
        marker="o",
        color=sns.color_palette("Dark2")[0],
        height=2.5,
        aspect=2.5,
        teeplot_subdir=teeplot_subdir,
        teeplot_show=True,
    ) as _g:
        _g.set_titles("prq49 (original)")
    return


@app.cell(hide_code=True)
def delimit_replays(mo):
    mo.md("""
    ## Replay Trajectories
    Each facet is one replay bucket.
    Points above the y-axis limit are flagged with a red triangle.
    The original trajectory is underlaid in gray.
    """)
    return


@app.cell
def plot_replays(df_original, df_replays, sns, teeplot_subdir, tp):
    ymax = 120  # outliers above this are flagged with a red triangle
    with tp.teed(
        sns.relplot,
        data=df_replays,
        x="Stint",
        y="Genotype Complexity",
        col="replay",
        col_wrap=5,
        kind="line",
        marker="o",
        color=sns.color_palette("Dark2")[0],
        height=1.6,
        aspect=1.5,
        teeplot_subdir=teeplot_subdir,
        teeplot_show=True,
    ) as _g:
        _g.set_titles("{col_name}", size=8)
        _g.set_axis_labels("Stint", "")
        _g.figure.supylabel("Genotype Complexity", x=0.0, fontsize=12)
        for _name, _ax in _g.axes_dict.items():
            _ax.plot(
                df_original["Stint"],
                df_original["Genotype Complexity"],
                color="0.75",
                zorder=-1,
            )
            _ax.set_ylim(0, ymax)
            _clipped = df_replays[
                (df_replays["replay"] == _name)
                & (df_replays["Genotype Complexity"] > ymax)
            ]
            _ax.scatter(
                _clipped["Stint"],
                [ymax] * len(_clipped),
                marker="^",
                color="red",
                clip_on=False,
                zorder=3,
            )
    return


if __name__ == "__main__":
    app.run()
