#!/usr/bin/env bash
# Remove what VM runs leave behind: exited benchmark containers and their anonymous volumes (the
# osworld-docker image declares VOLUME /storage; closing an env removes the container only).
# Refuses while any benchmark container is running.
set -euo pipefail
IMAGES=(happysixd/osworld-docker@sha256:0e6497a9295647cf05bf2b2af522fdd79bdeba2737595259cab310a3bcf6baa9
        agentslastexam/ale-qemu:0.2.0)
filters=()
for image in "${IMAGES[@]}"; do filters+=(--filter "ancestor=$image"); done
running=$(docker ps -q "${filters[@]}" | wc -l)
if [[ $running -gt 0 ]]; then echo "$running benchmark containers running; not cleaning" >&2; exit 1; fi
for image in "${IMAGES[@]}"; do
  ids=$(docker ps -aq --filter status=exited --filter status=created --filter ancestor="$image")
  [[ -z $ids ]] || docker rm -v $ids >/dev/null
done
before=$(docker volume ls -qf dangling=true | wc -l)
docker volume prune -f >/dev/null
echo "removed $before dangling volumes"
docker system df
