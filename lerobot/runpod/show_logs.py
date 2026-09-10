import argparse
import sys

import runpod

from launch_training import POD_NAME, REMOTE_DIR, configure, run_ssh


def find_pod():
    for pod in runpod.get_pods():
        if pod["name"] == POD_NAME:
            return pod
    sys.exit(f"No pod named {POD_NAME} running.")


def ssh_endpoint(pod):
    runtime = pod.get("runtime") or {}
    for mapping in runtime.get("ports") or []:
        if mapping.get("privatePort") == 22 and mapping.get("isIpPublic"):
            return mapping["ip"], mapping["publicPort"]
    sys.exit(f"Pod {pod['id']} has no public SSH endpoint yet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lines", type=int, default=100)
    parser.add_argument("-f", "--follow", action="store_true")
    args = parser.parse_args()

    configure()
    pod = find_pod()
    ip, port = ssh_endpoint(pod)
    follow = "-f " if args.follow else ""
    try:
        run_ssh(ip, port, f"tail -n {args.lines} {follow}{REMOTE_DIR}/train.log")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
