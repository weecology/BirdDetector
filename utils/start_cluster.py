"""
Create a cluster of GPU nodes to perform parallel prediction of tiles
"""
import argparse
import sys
import socket
from dask_jobqueue import SLURMCluster
from dask.distributed import Client, wait
import gc

def collect():
    gc.collect()

def args():
    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--debug',
                        help='Run local version without GPU',
                        action='store_true')
    parser.add_argument('--workers', help='Number of dask workers', default="4")
    parser.add_argument('--memory_worker', help='GB memory per worker', default="10")

def start_tunnel():
    """
    Start a juypter session and ssh tunnel to view task progress
    """
    host = socket.gethostname()
    print("To tunnel into dask dashboard:")
    print("For GPU dashboard: ssh -N -L 8787:%s:8787 -l b.weinstein hpg2.rc.ufl.edu" %
          (host))
    print("For CPU dashboard: ssh -N -L 8781:%s:8781 -l b.weinstein hpg2.rc.ufl.edu" %
          (host))

    #flush system
    sys.stdout.flush()


def start(cpus=0, gpus=0, mem_size="10GB"):
    #################
    # Setup dask cluster
    #################

    if cpus > 0:
        #job args
        extra_args = [
            "--error=/orange/idtrees-collab/logs/dask-worker-%j.err", "--account=ewhite",
            "--output=/orange/idtrees-collab/logs/dask-worker-%j.out"
        ]

        cluster = SLURMCluster(queue='hpg2-compute',
                               memory=mem_size,
                               walltime='1:00:00',
                               job_extra=extra_args,
                               scheduler_options={"dashboard_address": ":8781"},
                               local_directory="/orange/idtrees-collab/tmp/",
                               death_timeout=300)

        print(cluster.job_script())
        cluster.scale(cpus)

    if gpus:
        #job args
        extra_args = [
            "--error=/orange/idtrees-collab/logs/dask-worker-%j.err", "--account=ewhite",
            "--output=/orange/idtrees-collab/logs/dask-worker-%j.out", "--partition=gpu",
            "--gpus=1",
            "--cpus-per-task=2",
            "--nodes=1"
        ]

        cluster = SLURMCluster(cores=1,
                               memory=mem_size,
                               walltime='24:00:00',
                               job_extra=extra_args,
                               scheduler_options={"dashboard_address": ":8787"},
                               local_directory="/orange/idtrees-collab/tmp/",
                               death_timeout=300)

        cluster.scale(gpus)

    dask_client = Client(cluster)

    #Start dask
    dask_client.run_on_scheduler(start_tunnel)

    return dask_client
